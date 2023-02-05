import os
import argparse
import gc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np
import pkbar
from apmeter import APMeter

import x3d as resnet_x3d

from kinetics_multigrid import Kinetics
from kinetics import Kinetics as Kinetics_val

from kinetics_multigrid import kin_collate_fn as collate_fn_tr
from kinetics import kin_collate_fn as collate_fn_val

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled, RandomEqualize
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import cycle_batch_sampler as cbs
import dataloader as DL

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



KINETICS_TRAIN_ROOT = '/dataset/train_frames'
KINETICS_TRAIN_ANNO = '/dataset/kinetics400/train.json'
KINETICS_VAL_ROOT = '/dataset/valid_frames'
KINETICS_VAL_ANNO = '/dataset/kinetics400/validate.json'
KINETICS_CLASS_LABELS = '/dataset/kinetics400/labels.txt'


KINETICS_MEAN = [110.63666788/255, 103.16065604/255, 96.29023126/255]
KINETICS_STD = [38.7568578/255, 37.88248729/255, 40.02898126/255]
KINETICS_DATASET_SIZE = {'train':220000, 'val':17500}

BS = 8
BS_UPSCALE = 16*2//4 #2 #8 #16 # CHANGE WITH GPU AVAILABILITY
INIT_LR = (1.6/1024)*(BS*BS_UPSCALE) /2 #/10
SCHEDULE_SCALE = 2 #4
EPOCHS = (60000 * 1024 * 1.5)/220000 #(~420)

LONG_CYCLE = [8, 4, 2, 1]
LONG_CYCLE_LR_SCALE = [8, 0.5, 0.5, 0.5]
GPUS = 2*2 #1 #4
BASE_BS_PER_GPU = BS * BS_UPSCALE // GPUS # FOR SPLIT BN
CONST_BN_SIZE = 8

X3D_VERSION = 'M' # ['S', 'M', 'XL']


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def setup_data(batch_size, num_steps_per_update, epochs, iterations_per_epoch, cur_iterations, crop_size, resize_size, num_frames, gamma_tau):

  torch.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  num_iterations = int(epochs * iterations_per_epoch)
  #schedule = [int(i*num_iterations) for i in [0, 0.4, 0.65, 0.85, 1]]
  schedule = [int(i*num_iterations) for i in [0, 0.025, 0.075, 0.2, 1]]
  print(schedule)

  train_transforms = {
      'spatial':  Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                           #RandomEqualize(p=1.),
                           RandomHorizontalFlip(),
                           ToTensor(255),
                           Normalize(KINETICS_MEAN, KINETICS_STD)
                           ]),
      'temporal': TemporalRandomCrop(num_frames, gamma_tau),
      'target':   ClassLabel()
  }

  dataset = Kinetics(
          KINETICS_TRAIN_ROOT,
          KINETICS_TRAIN_ANNO,
          KINETICS_CLASS_LABELS,
          'train',
          spatial_transform=train_transforms['spatial'],
          temporal_transform=train_transforms['temporal'],
          target_transform=train_transforms['target'],
          sample_duration=num_frames)

  drop_last = False
  shuffle = True
  if shuffle:
    sampler = cbs.RandomEpochSampler(dataset, epochs=epochs)
  else:
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)

  batch_sampler = cbs.CycleBatchSampler(sampler, batch_size, drop_last,
                                        schedule=schedule,
                                        cur_iterations = cur_iterations,
                                        long_cycle_bs_scale=LONG_CYCLE)
  dataloader = DL.DataLoader(dataset, num_workers=12, batch_sampler=batch_sampler, worker_init_fn=worker_init_fn,
                            pin_memory=True)#, collate_fn=collate_fn_tr)

  schedule[-2] = (schedule[-2]+schedule[-1])//2 # FINE TUNE LAST PHASE, HALF WITH PREV_LR AND HALF WITH REDUCED_LR

  return dataloader, dataset, schedule[1:]



max_epochs = int(EPOCHS/SCHEDULE_SCALE) #*3 #+ 10 # 120 --> 210

def run(init_lr=INIT_LR, warmup_steps=8000, max_epochs=max_epochs, batch_size=BS*BS_UPSCALE):

    frames=80
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5*1, 'XL':5}[X3D_VERSION] # 'M':5 FOR LONGER SCHEDULE, NUM OF GPUS INCREASE

    steps = load_steps = st_steps = 500000
    epochs = 150
    num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = KINETICS_DATASET_SIZE['train']//batch_size
    val_batch_size = batch_size//2
    val_iterations_per_epoch = KINETICS_DATASET_SIZE['val']//val_batch_size
    max_steps = iterations_per_epoch * max_epochs

    last_long = -2

    dataloader, dataset, lr_schedule = setup_data(batch_size, num_steps_per_update, max_epochs, iterations_per_epoch,
                                                cur_iterations, crop_size, resize_size, frames, gamma_tau)

    lr_schedule = [i//num_steps_per_update for i in lr_schedule]

    validation_transforms = {
        'spatial':  Compose([CenterCropScaled(crop_size),
                             ToTensor(255),
                             Normalize(KINETICS_MEAN, KINETICS_STD)
                             ]),
        'temporal': TemporalRandomCrop(frames, gamma_tau),
        'target':   ClassLabel()
    }

    val_dataset = Kinetics_val(
            KINETICS_VAL_ROOT,
            KINETICS_VAL_ANNO,
            KINETICS_CLASS_LABELS,
            'validate',
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=frames,
            gamma_tau=gamma_tau,
            crops=3)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                        worker_init_fn=worker_init_fn, num_workers=12)#, collate_fn=collate_fn_val)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', lr_schedule[-1]*num_steps_per_update, 'Total steps:', lr_schedule[-1])
    print('datasets created')


    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400+0, n_input_channels=3,
                                    dropout=0.5, base_bn_splits=BASE_BS_PER_GPU//CONST_BN_SIZE//2, task='loc') #*2 //6 'dual-loc'
    load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    x3d.load_state_dict(load_ckpt['model_state_dict'])

    #x3d.replace_logits(400+1)


    #save_model = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_freeze_')
    save_model = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_mixup_')
    #save_model = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_cutmix_')



    RESTART = False
    if steps>0:
        load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_freeze_'+str(load_steps).zfill(6)+'.pt')
        cur_long_ind = -1 #load_ckpt['long_ind']
        x3d.load_state_dict(load_ckpt['model_state_dict'])
        last_long = cur_long_ind
        RESTART = True

    x3d.cuda()
    x3d = nn.DataParallel(x3d)
    print('model loaded')

    lr = init_lr
    print ('INIT LR: %f'%lr)

    PRED =['module.fc1.weight',
            'module.fc2.weight',
            'module.fc2.bias']
    backbone_params=[]; pred_params=[];
    for name,para in x3d.named_parameters():
        if name in PRED: pred_params.append(para); print('pred_para {}'.format(name))
        else: backbone_params.append(para)

    optimizer = optim.SGD([{'params': pred_params}, {'params': backbone_params, 'lr': lr/10}], lr=lr, momentum=0.9, weight_decay=5e-5)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)
    if steps>0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion = nn.CrossEntropyLoss(reduction='none')

    val_apm = APMeter()
    tr_apm = APMeter()

    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in 2*['train']+['val']: #['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)
            if phase == 'train':
                x3d.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                x3d.train(False)  # Set model to evaluate mode
                _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS
                torch.autograd.set_grad_enabled(False)

            tot_loss = 0.0
            tot_cls_loss = 0.0
            tot_loc_loss = 0.0
            tot_acc = 0.0
            tot_corr = 0.0
            tot_dat = 0.0
            num_iter = 0

            tot_BG, tot_NBG, tot_FP, tot_FN = 0., 0., 0., 0.

            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for i,data in enumerate(dataloaders[phase]):
                num_iter += 1
                bar.update(i)
                if phase == 'train':
                    if i> iterations_per_epoch:
                        break
                    IS_BG_AUG = False #True if torch.rand(1)<0.5 else False

                    inputs, labels, long_ind, stats, vid_id = data
                    # inputs B C T H W labels B T
                    inputs = inputs.cuda() # B 3 T H W
                    labels = labels.cuda() # B T
                    b,c,t,h,w = inputs.shape

                else:
                    IS_BG_AUG = False #True if num_iter%2 == 0 else False

                    inputs, labels, vid_id, _ = data
                    b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
                    bn = b*n
                    inputs = inputs.view(b*n,c,t,h,w)

                    labels = labels.view(b*n,t)
                    inputs = inputs.cuda() # B 3 T H W
                    labels = labels.cuda()

                ################################################# START ###########################################################################
                ###################################################################################################################################

                AUG_SWITCH = torch.rand(3).view(1,-1).repeat(4,1) #random.randint(0,3) if phase == 'train' else i%4

                logits, labels, rand_gammas, rand_mask = x3d(inputs, labels, AUG_SWITCH) # B C 1 --> B C T
                labels, rand_gammas, rand_mask = labels.transpose(0,1), rand_gammas.transpose(0,1), rand_mask.transpose(0,1)

                _, preds = torch.max(logits, 1)

                # logits B C T labels M B T
                cls_loss = 0.
                correct = 0.
                if IS_BG_AUG:
                    cls_loss += torch.mean(criterion(logits, labels))
                    correct += torch.sum(preds == labels.data).double()
                else:
                    for mix_ind in range(labels.shape[0]):
                        cls_loss += torch.sum(rand_gammas[mix_ind] * criterion(logits, labels[mix_ind])) / torch.sum(rand_mask[mix_ind]) # .view(1,-1)
                        correct += torch.sum(rand_gammas[mix_ind] * (preds == labels[mix_ind].data)).double() # .view(1,-1)
                tot_corr += correct#.double()
                tot_dat += labels.shape[-1] * labels.shape[-2]
                tot_cls_loss += cls_loss.item()
                loss = cls_loss/num_steps_per_update
                tot_loss += loss.item()

                ################################################# END #############################################################################
                ###################################################################################################################################


                if phase == 'train':
                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    lr_warmup(lr, steps-st_steps, warmup_steps, optimizer) # USE ONLY AT THE START, AVOID OVERLAP WITH LONG_CYCLE CHANGES
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    s_times = iterations_per_epoch//10 #2
                    if (steps-load_steps) % s_times == 0:
                        tot_acc = tot_corr/tot_dat
                        print (' Epoch:{} {} steps: {} Cls Loss: {:.4f} Tot Loss: {:.4f} acc: {:.4f}'.format(epochs, phase,
                            steps, tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tot_acc))#, tot_acc, tot_bg_fp, tot_bg_fn)) Acc: {:.4f} BG_FP: {:.4f} BG_FN: {:.4f}
                        tot_loss = tot_cls_loss = tot_loc_loss = tot_acc = tot_corr = tot_dat = 0.
                        tot_BG, tot_NBG, tot_FP, tot_FN = 0., 0., 0., 0.
                    if steps % (1000*2) == 0:
                        ckpt = {'model_state_dict': x3d.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_sched.state_dict(),
                                'long_ind': long_ind}
                        torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                tot_acc = tot_corr/tot_dat
                print (' Epoch:{} {} Cls Loss: {:.4f} Tot Loss: {:.4f} acc: {:.4f}'.format(epochs, phase,
                    tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, tot_acc))#, tot_acc, tot_bg_fp, tot_bg_fn)) Acc: {:.4f} BG_FP: {:.4f} BG_FN: {:.4f}
                tot_loss = tot_cls_loss = tot_loc_loss = tot_acc = tot_corr = tot_dat = 0.
                tot_BG, tot_NBG, tot_FP, tot_FN = 0., 0., 0., 0.

            del inputs, labels, logits
            gc.collect()
            torch.cuda.empty_cache()



def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr


if __name__ == '__main__':
    run()
