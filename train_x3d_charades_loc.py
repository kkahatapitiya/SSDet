import os
import argparse

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
from barbar import Bar
import pkbar
from apmeter import APMeter

import x3d as resnet_x3d

from charades import Charades
from charades import custom_collate_fn as collate_fn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


BS = 16
BS_UPSCALE = 1 #2
INIT_LR = 0.02 * BS_UPSCALE #/ 10.
GPUS = 1 #2

X3D_VERSION = 'M'

CHARADES_ROOT = '/dataset/Charades_v1_rgb'
CHARADES_ANNO = 'data/charades.json'
CHARADES_DATASET_SIZE = {'train':7900, 'val':1850}
CHARADES_MEAN = [0.413, 0.368, 0.338]
CHARADES_STD = [0.131, 0.125, 0.132] # CALCULATED ON CHARADES TRAINING SET FOR FRAME-WISE MEANS
# ON VAL SET MEAN:[0.415 0.384 0.366], STD:[0.146 0.140 0.137]

# warmup_steps=0
def run(init_lr=INIT_LR, max_epochs=100, root=CHARADES_ROOT, anno=CHARADES_ANNO, batch_size=BS*BS_UPSCALE):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS

    steps = load_steps = st_steps = 0
    epochs = 0
    num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = CHARADES_DATASET_SIZE['train']//batch_size
    val_iterations_per_epoch = CHARADES_DATASET_SIZE['val']//(batch_size//2)
    max_steps = iterations_per_epoch * max_epochs


    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(CHARADES_MEAN, CHARADES_STD)])

    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(CHARADES_MEAN, CHARADES_STD)])

    dataset = Charades(anno, 'training', root, train_spatial_transforms,
                        task='loc', frames=80, gamma_tau=gamma_tau, crops=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=8, pin_memory=True, collate_fn=collate_fn)

    val_dataset = Charades(anno, 'testing', root, val_spatial_transforms,
                            task='loc', frames=80, gamma_tau=gamma_tau, crops=10)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False,
                                                num_workers=8, pin_memory=True, collate_fn=collate_fn)


    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')


    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400+0, n_input_channels=3, task='loc', dropout=0.5, base_bn_splits=2) # 400+1

    load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_freeze.pt')
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_mixup.pt')
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_cutmixpt')


    state_to_load = load_ckpt['model_state_dict']
    state = x3d.state_dict()
    #state.update(load_ckpt['model_state_dict'])
    for k in state_to_load:
        if state_to_load[k].shape != state[k].shape:
            if 'running_mean' in k or 'running_var' in  k:
                n = state_to_load[k].shape[0]
                state[k] = F.adaptive_avg_pool1d(state_to_load[k].view(1,1,n), n//2*4).view(-1)
            elif 'fc' in k:
                state[k] = state_to_load[k].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                print(k, state_to_load[k].shape, state[k].shape)
            #state[k] = state_to_load[k].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        else:
            state[k] = state_to_load[k]
    x3d.load_state_dict(state)

    save_model = 'models/x3d_charades_loc_rgb_sgd_'
    x3d.replace_logits(157)

    if steps>0:
        load_ckpt = torch.load('models/x3d_charades_loc_rgb_sgd_'+str(load_steps).zfill(6)+'.pt')
        x3d.load_state_dict(load_ckpt['model_state_dict'])

    x3d.cuda()
    x3d = nn.DataParallel(x3d)
    print('model loaded')

    lr = init_lr
    print ('INIT LR: %f'%lr)


    optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=True) #2
    if steps>0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion = nn.BCEWithLogitsLoss()#pos_weight=pos_weight[1]) # for CLS task
    criterion2 = nn.BCEWithLogitsLoss()#pos_weight=pos_weight[0].view(-1,1)) # for LOC task

    val_apm = APMeter()
    tr_apm = APMeter()

    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in 2*['train']+['val']:
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
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for i,data in enumerate(dataloaders[phase]):
                num_iter += 1
                if i>=bar_st:
                    continue
                bar.update(i)
                if phase == 'train':
                    inputs, labels, masks, _ = data
                else:
                    inputs, labels, masks, _ = data

                inputs = inputs.cuda() # B 3 T W H
                tl = labels.size(2)
                #labels = torch.max(labels, dim=2)[0] # B C T --> B C
                labels = labels.cuda() # B C TL
                masks = masks.cuda() # B TL
                valid_t = torch.sum(masks, dim=1).int()

                per_frame_logits = x3d(inputs) # B C T
                per_frame_logits = F.interpolate(per_frame_logits, tl, mode='linear')

                probs = F.sigmoid(per_frame_logits) * masks.unsqueeze(1)

                cls_loss = criterion(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item()

                loc_loss = criterion2(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item()

                if phase == 'train':
                    for b in range(labels.shape[0]):
                        tr_apm.add(probs[b][:,:valid_t[b].item()].transpose(0,1).detach().cpu().numpy(),
                                    labels[b][:,:valid_t[b].item()].transpose(0,1).cpu().numpy())
                else:
                    for b in range(labels.shape[0]):
                        val_apm.add(probs[b][:,:valid_t[b].item()].transpose(0,1).detach().cpu().numpy(),
                                    labels[b][:,:valid_t[b].item()].transpose(0,1).cpu().numpy())


                loss = (cls_loss + loc_loss)/(2 * num_steps_per_update)
                tot_loss += loss.item()

                if phase == 'train':
                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    #lr_warmup(lr, steps-st_steps, warmup_steps, optimizer)
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    #lr_sched.step()
                    s_times = iterations_per_epoch//2
                    if (steps-load_steps) % s_times == 0:
                        tr_map = tr_apm.value().mean()
                        tr_apm.reset()
                        print (' Epoch:{} {} steps: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                            steps, tot_loc_loss/(s_times*num_steps_per_update), tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tr_map))#, tot_acc/(s_times*num_steps_per_update)))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
                    if steps % (1000) == 0:
                        #tr_apm.reset()
                        ckpt = {'model_state_dict': x3d.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_sched.state_dict()}
                        torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                val_map = val_apm.value().mean()
                lr_sched.step(tot_loss)
                val_apm.reset()
                print (' Epoch:{} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                    tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))#, tot_acc/num_iter))
                tot_loss = tot_loc_loss = tot_cls_loss = 0.


def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr


if __name__ == '__main__':
    run()
