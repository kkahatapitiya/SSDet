import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, task='loc'):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.task = task

    def forward(self, x):
        if self.task == 'dual-loc':
            b,c,t,h,w = x.shape
            #print(x.shape)
            lx = x[...,:w//2]
            rx = x[...,w//2:]
            lx = F.conv3d(lx, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            rx = F.conv3d(rx, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return torch.cat([lx,rx], dim=-1)
        else:
            return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SubBatchNorm3d(nn.Module):
    """ FROM SLOWFAST """
    def __init__(self, num_splits, **args):
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        self.num_features = args["num_features"]
        # Keep only one set of weight and bias.
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = self.num_features * self.num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n
            + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """Synchronize running_mean, and running_var. Call this before eval."""
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x


class Swish(nn.Module):
    """ FROM SLOWFAST """
    """Swish activation function: x * sigmoid(x)."""
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    """ FROM SLOWFAST """
    """Swish activation function: x * sigmoid(x)."""
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


def conv3x3x3(in_planes, out_planes, stride=1, task='loc'):
    return Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=(1,stride,stride),
                     padding=1,
                     bias=False,
                     groups=in_planes,
                     task=task
                     )


def conv1x1x1(in_planes, out_planes, stride=1, task='loc'):
    return Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=(1,stride,stride),
                     bias=False,
                     task=task
                     )


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None, index=0, base_bn_splits=8, task='loc'):
        super(Bottleneck, self).__init__()

        self.index = index
        self.base_bn_splits = base_bn_splits
        self.conv1 = conv1x1x1(in_planes, planes[0], task=task)
        self.bn1 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[0], affine=True) #nn.BatchNorm3d(planes[0])
        self.conv2 = conv3x3x3(planes[0], planes[0], stride, task=task)
        self.bn2 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[0], affine=True) #nn.BatchNorm3d(planes[0])
        self.conv3 = conv1x1x1(planes[0], planes[1], task=task)
        self.bn3 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[1], affine=True) #nn.BatchNorm3d(planes[1])
        self.swish = Swish() #nn.Hardswish()
        self.relu = nn.ReLU(inplace=True)
        self.task = task
        if self.index % 2 == 0:
            width = self.round_width(planes[0])
            self.global_pool = nn.AdaptiveAvgPool3d((1,1,2)) if task == 'dual-loc' else nn.AdaptiveAvgPool3d((1,1,1))
            self.fc1 = Conv3d(planes[0], width, kernel_size=1, stride=1, task=task)
            self.fc2 = Conv3d(width, planes[0], kernel_size=1, stride=1, task=task)
            self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def round_width(self, width, multiplier=0.0625, min_width=8, divisor=8):
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # Squeeze-and-Excitation
        if self.index % 2 == 0:
            se_w = self.global_pool(out)
            se_w = self.fc1(se_w)
            se_w = self.relu(se_w)
            se_w = self.fc2(se_w)
            se_w = self.sigmoid(se_w)
            if self.task == 'dual-loc':
                w_ = out.shape[-1]
                out_l = out[...,:w_//2] * se_w[...,0].unsqueeze(-1)
                out_r = out[...,w_//2:] * se_w[...,1].unsqueeze(-1)
                out = torch.cat([out_l, out_r], dim=-1)
            else:
                out = out * se_w
        out = self.swish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 shortcut_type='B',
                 widen_factor=1.0,
                 dropout=0.5,
                 n_classes=400,
                 base_bn_splits=8,
                 task='class'):
        super(ResNet, self).__init__()

        block_inplanes = [(int(x * widen_factor),int(y * widen_factor)) for x,y in block_inplanes]
        self.index = 0
        self.base_bn_splits = base_bn_splits
        self.task = task
        self.mixup_ind = 0
        self.counter = 0
        self.AUG_SWITCH = torch.rand(3) #0

        self.in_planes = block_inplanes[0][1]

        self.conv1_s = Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, 2, 2),
                               padding=(0, 1, 1),
                               bias=False,
                               task=task)
        self.conv1_t = Conv3d(self.in_planes,
                               self.in_planes,
                               kernel_size=(5, 1, 1),
                               stride=(1, 1, 1),
                               padding=(2, 0, 0),
                               bias=False,
                               groups=self.in_planes,
                               task=task)
        self.bn1 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=self.in_planes, affine=True) #nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type,
                                       stride=2,
                                       task=task)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.conv5 = Conv3d(block_inplanes[3][1],
                               block_inplanes[3][0],
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=(0, 0, 0),
                               bias=False)
        self.bn5 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=block_inplanes[3][0], affine=True) #nn.BatchNorm3d(block_inplanes[3][0])
        if task == 'class':
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.task == 'loc' or self.task == 'dual-loc':
            self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        #elif task == 'dual-loc':
        #    self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 2))
        self.fc1 = Conv3d(block_inplanes[3][0], 2048, bias=False, kernel_size=1, stride=1)
        self.fc2 = nn.Linear(2048, n_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, task='loc'):
        downsample = None
        if stride != 1 or self.in_planes != planes[1]:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes[1],
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes[1], stride, task=task),
                    SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[1], affine=True) #nn.BatchNorm3d(planes[1])
                    )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  index=self.index,
                  base_bn_splits=self.base_bn_splits,
                  task=task))
        self.in_planes = planes[1]
        self.index += 1
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, index=self.index, base_bn_splits=self.base_bn_splits,  task=task))
            self.index += 1

        self.index = 0
        return nn.Sequential(*layers)


    def replace_logits(self, n_classes):
        self.fc2 = nn.Linear(2048, n_classes)


    def freeze(self):
        TRAINABLE =['fc1.weight',
                    'fc2.weight',
                    'fc2.bias']
        for name, param in self.named_parameters():
            if name in TRAINABLE: # or 'Mixed_4f' in name :
                continue
            param.requires_grad = False


    def update_bn_splits_long_cycle(self, long_cycle_bn_scale):
        for m in self.modules():
            if isinstance(m, SubBatchNorm3d):
                m.num_splits = self.base_bn_splits * long_cycle_bn_scale
                m.split_bn = nn.BatchNorm3d(num_features=m.num_features*m.num_splits, affine=False).to(m.weight.device)
        return self.base_bn_splits * long_cycle_bn_scale


    def aggregate_sub_bn_stats(self):
        """find all SubBN modules and aggregate sub-BN stats."""
        count = 0
        for m in self.modules():
            if isinstance(m, SubBatchNorm3d):
                m.aggregate_stats()
                count += 1
        return count


    def forward(self, inputs, labels, AUG_SWITCH):

        ################################################# START ###########################################################################
        ###################################################################################################################################

        '''if self.training:
            mixup_level = random.randint(0,4)
            self.mixup_ind = 0
        else:
            self.mixup_ind += 1
            mixup_level = self.mixup_ind % 5'''

        mixup_level = 0

        self.AUG_SWITCH = AUG_SWITCH.view(-1).detach()
        if mixup_level == 0: inputs, labels, rand_gammas, rand_mask = self.perform_aug(inputs, labels)

        x = self.conv1_s(inputs)
        x = self.conv1_t(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        #if mixup_level == 1: x, labels, rand_gammas, rand_mask = self.perform_aug(x, labels)

        x = self.layer2(x)
        #if mixup_level == 2: x, labels, rand_gammas, rand_mask = self.perform_aug(x, labels)

        x = self.layer3(x)
        #if mixup_level == 3: x, labels, rand_gammas, rand_mask = self.perform_aug(x, labels)

        x = self.layer4(x)
        #if mixup_level == 4: x, labels, rand_gammas, rand_mask = self.perform_aug(x, labels)

        ################################################# END #############################################################################
        ###################################################################################################################################


        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.avgpool(x)

        x = self.fc1(x)
        x = self.relu(x) # B C T 1 1 or B C T 1 2

        if self.task == 'class':
            x = x.squeeze(4).squeeze(3).squeeze(2) # B C
            x = self.dropout(x)
            x = self.fc2(x).unsqueeze(2) # B C 1
        if self.task == 'loc' or self.task == 'dual-loc':
            x = x.squeeze(4).squeeze(3).permute(0,2,1) # B T C
            x = self.dropout(x)
            x = self.fc2(x).permute(0,2,1) # B C T

        return x, labels.transpose(0,1), rand_gammas.transpose(0,1), rand_mask.transpose(0,1)


    ################################################# START ###########################################################################
    ###################################################################################################################################

    def perform_aug(self, inputs, labels):
        b,c,t,h,w = inputs.shape

        x = inputs
        if self.AUG_SWITCH[0] <= 0.5:
            x, labels = self.bg_freeze(x, labels)
        labels = labels.unsqueeze(0) # 1 B T
        rand_gammas, rand_mask = torch.ones(1,b,t).cuda(), torch.ones(1,b,t).cuda()
        if self.AUG_SWITCH[1] <= 0.5:
            x, labels, rand_gammas, rand_mask = self.mixup(x, labels, rand_gammas, rand_mask)
        if self.AUG_SWITCH[2] <= 0.5:
            rand_gammas, rand_mask = rand_gammas.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,w), rand_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,w)
            x, labels, rand_gammas, rand_mask = self.cutmix_v2(x, labels, rand_gammas, rand_mask)

        return x, labels, rand_gammas, rand_mask


    def bg_freeze(self, inputs, labels):

        b,c,t,h,w = inputs.shape

        if self.training:
            rand_len_bg = torch.randint(1,t//2,(b,)).cuda() # B
            rand_st_bg = torch.tensor([torch.randint(ind_st_bg+1,(1,)) for ind_st_bg in t-rand_len_bg.view(-1)]).view(rand_len_bg.shape).cuda() # B
            for_mask_bg = torch.arange(t).view(1,-1).repeat(b,1).cuda() # B T

        else:
            rand_len_bg = (torch.ones(b) * t//4).long().cuda() # BN
            rand_st_bg = (torch.ones(b) * t//8).long().cuda()
            rand_st_bg[torch.arange(b)%2 != 0] = 5*t//8
            for_mask_bg = torch.arange(t).view(1,-1).repeat(b,1).cuda() # BN T

        freeze_copy_mask_bg = ((for_mask_bg == rand_st_bg.view(-1,1)) * 1.) # B T
        freeze_paste_mask_bg = ((for_mask_bg >= rand_st_bg.view(-1,1)) * (for_mask_bg < (rand_st_bg+rand_len_bg).view(-1,1)) * 1.) # B T
        rest_copy_mask_bg = ((for_mask_bg > rand_st_bg.view(-1,1)) * (for_mask_bg <= t-rand_len_bg.view(-1,1)) * 1.) # B T
        rest_paste_mask_bg = ((for_mask_bg>=(rand_st_bg+rand_len_bg).view(-1,1)) * 1.) # B T

        inputs_to_replicate_bg = inputs[freeze_copy_mask_bg.view(b,1,t,1,1).repeat(1,c,1,h,w).bool()].view(b,c,1,h,w).repeat(1,1,t,1,1)
        inputs[rest_paste_mask_bg.view(b,1,t,1,1).repeat(1,c,1,h,w).bool()] = inputs[rest_copy_mask_bg.view(b,1,t,1,1).repeat(1,c,1,h,w).bool()]
        inputs[freeze_paste_mask_bg.view(b,1,t,1,1).repeat(1,c,1,h,w).bool()] = inputs_to_replicate_bg[freeze_paste_mask_bg.view(b,1,t,1,1).repeat(1,c,1,h,w).bool()]

        labels_to_replicate_bg = torch.zeros(b,t).long().cuda() #labels[freeze_copy_mask_bg.bool()].view(b,1).repeat(1,t)
        labels[rest_paste_mask_bg.bool()] = labels[rest_copy_mask_bg.bool()]
        labels[freeze_paste_mask_bg.bool()] = labels_to_replicate_bg[freeze_paste_mask_bg.bool()]

        return inputs, labels

    '''
    def cutmix_v3(self, inputs, labels, prev_gammas, prev_mask):

        b,c,t,h,w = inputs.shape
        MIX = 2

        if self.training:
            rand_len = torch.randint(t//2,t+1,(MIX,b)).cuda() # t//4,3*t//4+1 # M B
            rand_st = torch.tensor([torch.randint(ind_st+1,(1,)) for ind_st in t-rand_len.view(-1)]).view(rand_len.shape).cuda() # M B
            rand_mask = torch.arange(t).view(1,1,-1).repeat(MIX,b,1).cuda() # M B T
            rand_mask = ((rand_mask >= rand_st.unsqueeze(-1)) * (rand_mask < (rand_st+rand_len).unsqueeze(-1)) * 1.)# M B T

        else:
            rand_len = torch.randint(3*t//4,3*t//4+1,(MIX,)).view(-1,1).repeat(1,b).cuda() # 3*t//4,3*t//4+1 # M BN
            rand_st = torch.tensor([0, t//4]).view(-1,1).repeat(1,b).cuda() # t//12, t//6, t//4 # M BN
            rand_mask = torch.arange(t).view(1,1,-1).repeat(MIX,b,1).cuda() # M BN T
            rand_mask = ((rand_mask >= rand_st.unsqueeze(-1)) * (rand_mask < (rand_st+rand_len).unsqueeze(-1)) * 1.) # M BN T


        rand_overlap = rand_mask[0]*rand_mask[1] # B T
        window_step = 1./(torch.sum(rand_overlap, dim=-1, keepdim=True)+1) # B 1

        window_inc = (torch.arange(1,t+1).cuda().view(1,-1) - torch.max(rand_st,dim=0)[0].view(-1,1)) * window_step * rand_overlap * w
        window_dec = torch.clamp(((w - window_inc) + w//16) * rand_overlap, min=0, max=w)
        window_inc = torch.clamp((window_inc + w//16) * rand_overlap, min=0, max=w)
        window_dec2 = torch.abs(torch.arange(1,t+1).cuda().view(1,-1) - 0.5*(torch.max(rand_st,dim=0)[0]+torch.min(rand_st+rand_len,dim=0)[0]).view(-1,1)) * 2 * window_step * rand_overlap * w
        window_inc2 = torch.clamp(((w - window_dec2) + w//16) * rand_overlap, min=0, max=w)
        window_dec2 = torch.clamp((window_dec2 + w//16) * rand_overlap, min=0, max=w)

        rel_st = rand_st[0] - rand_st[1]
        rel_end = (rand_st+rand_len)[0] - (rand_st+rand_len)[1]
        rand_windows = rand_mask.clone() * w # M B T

        condition = ((rel_st>=0) * (rel_end>=0)).view(-1,1) * rand_overlap # B T
        rand_windows[0][condition.bool()] = window_inc[condition.bool()]
        rand_windows[1][condition.bool()] = window_dec[condition.bool()]

        condition = ((rel_st<0) * (rel_end<0)).view(-1,1) * rand_overlap # B T
        rand_windows[0][condition.bool()] = window_dec[condition.bool()]
        rand_windows[1][condition.bool()] = window_inc[condition.bool()]

        condition = ((rel_st>=0) * (rel_end<0)).view(-1,1) * rand_overlap # B T
        rand_windows[0][condition.bool()] = window_inc2[condition.bool()]
        rand_windows[1][condition.bool()] = window_dec2[condition.bool()]

        condition = ((rel_st<0) * (rel_end>=0)).view(-1,1) * rand_overlap # B T
        rand_windows[0][condition.bool()] = window_dec2[condition.bool()]
        rand_windows[1][condition.bool()] = window_inc2[condition.bool()]


        rand_windows = rand_windows.unsqueeze(-1).unsqueeze(-1) # M B T 1 1
        window_mask = torch.arange(w).view(1,1,1,-1).repeat(b,t,1,1).cuda() # B T 1 W
        window_mask = torch.stack([(window_mask < rand_windows[0]) * 1., (window_mask >= (w-rand_windows[1])) * 1.], dim=0) # M B T 1 W

        window_overlap = window_mask[0] * window_mask[1] # B T 1 W
        gamma_step = 1./(torch.sum(window_overlap, dim=-1, keepdim=True)+1) # B T 1 1
        gamma_inc = (torch.arange(1,w+1).cuda().view(1,1,1,-1) - (w-rand_windows[1])) * gamma_step * window_overlap # B T 1 W
        gamma_inc[(window_mask[1]-window_overlap)==1.] = 1.
        gamma_dec = (1. - gamma_inc) * window_mask[0]

        rand_gammas = torch.stack([gamma_dec, gamma_inc], dim=0) # M B T 1 W

        mix_in, mix_l, mix_gammas, mix_mask = [inputs], [labels], [prev_gammas], [prev_mask]
        for mix_ind in range(MIX-1):
            rand_index = torch.randperm(b).cuda() if self.training else ((torch.arange(b) + ((mix_ind+1)*b)//MIX + 2) % (b)).cuda()
            mix_in.append(inputs[rand_index])
            mix_l.append(labels[:,rand_index,:])
            mix_gammas.append(prev_gammas[:,rand_index,...])
            mix_mask.append(prev_mask[:,rand_index,...])
        mix_in = torch.stack(mix_in, dim=0) # M B C T H W
        mix_l = torch.cat(mix_l, dim=0) # M B T
        mix_gammas = torch.cat(mix_gammas, dim=0) # M B T 1 W
        mix_mask = torch.cat(mix_mask, dim=0) # M B T 1 W

        inputs = torch.sum(mix_in * rand_gammas.view(MIX,b,1,t,1,w), dim=0) # B C T H W
        labels = mix_l # M B T

        mix_gammas = mix_gammas * rand_gammas.unsqueeze(1).repeat(1,prev_gammas.shape[0],1,1,1,1).view(-1,b,t,1,w)
        mix_mask = mix_mask * window_mask.unsqueeze(1).repeat(1,prev_mask.shape[0],1,1,1,1).view(-1,b,t,1,w)

        mix_gammas = torch.sum(mix_gammas, dim=(3,4)) / (mix_gammas.shape[3]*mix_gammas.shape[4])
        mix_mask = (torch.sum(mix_mask, dim=(3,4))>0) * 1.

        return inputs, labels, mix_gammas, mix_mask'''


    def cutmix_v2(self, inputs, labels, prev_gammas, prev_mask):

        b,c,t,h,w = inputs.shape
        MIX = 2

        if self.training:
            rand_len = (torch.ones(1).long().cuda() * (9*w//16)).view(1,1,1,1,1).repeat(MIX,b,t,1,1)
            rand_st = (torch.arange(t).float() * (7*w//16) / t).long().cuda().view(1,1,-1,1,1).repeat(MIX,b,1,1,1)
            rand_mask = torch.arange(w).view(1,1,1,-1).repeat(b,t,1,1).cuda() # B T 1 W
            rand_mask = torch.stack([(rand_mask < rand_len[0]) * 1., (rand_mask >= (w-rand_len[1])) * 1.], dim=0) # M B T 1 W
            rand_st[1] = -rand_st[1]
            shift_ind = ((torch.arange(w).cuda().view(1,1,1,1,-1).repeat(MIX,b,t,1,1) + rand_st) % w).view(-1,w) + w*torch.arange(MIX*b*t).view(-1,1).cuda()

        else:
            rand_len = (torch.ones(1).long().cuda() * (9*w//16)).view(1,1,1,1,1).repeat(MIX,b,t,1,1)
            rand_st = (torch.arange(t).float() * (7*w//16) / t).long().cuda().view(1,1,-1,1,1).repeat(MIX,b,1,1,1)
            rand_mask = torch.arange(w).view(1,1,1,-1).repeat(b,t,1,1).cuda() # B T 1 W
            rand_mask = torch.stack([(rand_mask < rand_len[0]) * 1., (rand_mask >= (w-rand_len[1])) * 1.], dim=0) # M B T 1 W
            rand_st[1] = -rand_st[1]
            shift_ind = ((torch.arange(w).cuda().view(1,1,1,1,-1).repeat(MIX,b,t,1,1) + rand_st) % w).view(-1,w) + w*torch.arange(MIX*b*t).view(-1,1).cuda()


        rand_overlap = rand_mask[0] * rand_mask[1] # B T 1 W
        gamma_step = 1./(torch.sum(rand_overlap, dim=-1, keepdim=True)+1) # B T 1 1
        gamma_inc = (torch.arange(1,w+1).cuda().view(1,1,1,-1) - (w-rand_len[1])) * gamma_step * rand_overlap # B T 1 W
        gamma_inc[(rand_mask[1]-rand_overlap)==1.] = 1.
        gamma_dec = (1. - gamma_inc)

        rand_gammas = torch.stack([gamma_dec, gamma_inc], dim=0) # M B T 1 W

        mix_in, mix_l, mix_gammas, mix_mask = [inputs], [labels], [prev_gammas], [prev_mask]
        for mix_ind in range(MIX-1):
            rand_index = torch.randperm(b).cuda() if self.training else ((torch.arange(b) + ((mix_ind+1)*b)//MIX + 2) % (b)).cuda()
            mix_in.append(inputs[rand_index])
            mix_l.append(labels[:,rand_index,:])
            mix_gammas.append(prev_gammas[:,rand_index,...])
            mix_mask.append(prev_mask[:,rand_index,...])
        mix_in = torch.stack(mix_in, dim=0) # M B C T H W
        mix_l = torch.cat(mix_l, dim=0) # M B T
        mix_gammas = torch.cat(mix_gammas, dim=0) # M B T 1 W
        mix_mask = torch.cat(mix_mask, dim=0) # M B T 1 W

        mix_in = mix_in.permute(2,4,0,1,3,5).reshape(c,h,-1)[:,:,shift_ind.view(-1)].reshape(c,h,MIX,b,t,w).permute(2,3,0,4,1,5).contiguous()

        inputs = torch.sum(mix_in * rand_gammas.view(MIX,b,1,t,1,w), dim=0) # B C T H W
        labels = mix_l # M B T

        mix_gammas = mix_gammas * rand_gammas.unsqueeze(1).repeat(1,prev_gammas.shape[0],1,1,1,1).view(-1,b,t,1,w)
        mix_mask = mix_mask * rand_mask.unsqueeze(1).repeat(1,prev_mask.shape[0],1,1,1,1).view(-1,b,t,1,w)

        mix_gammas = torch.sum(mix_gammas, dim=(3,4)) / (mix_gammas.shape[3]*mix_gammas.shape[4])
        mix_mask = (torch.sum(mix_mask, dim=(3,4))>0) * 1.

        return inputs, labels, mix_gammas, mix_mask


    def mixup(self, inputs, labels, prev_gammas, prev_mask):

        b,c,t,h,w = inputs.shape
        MIX = 2

        if self.training:
            rand_len = torch.randint(t//2,t+1,(MIX,b)).cuda() # t//4,3*t//4+1 # M B
            rand_st = torch.tensor([torch.randint(ind_st+1,(1,)) for ind_st in t-rand_len.view(-1)]).view(rand_len.shape).cuda() # M B

            rand_st[torch.min(rand_st, dim=0)[1], [bi for bi in range(b)]] = 0
            rand_len[torch.max(rand_st+rand_len, dim=0)[1], [bi for bi in range(b)]] = t - rand_st[torch.max(rand_st+rand_len, dim=0)[1], [bi for bi in range(b)]]

            rand_mask = torch.arange(t).view(1,1,-1).repeat(MIX,b,1).cuda() # M B T
            rand_mask = ((rand_mask >= rand_st.unsqueeze(-1)) * (rand_mask < (rand_st+rand_len).unsqueeze(-1)) * 1.)# M B T

        else:
            rand_len = torch.randint(3*t//4,3*t//4+1,(MIX,)).view(-1,1).repeat(1,b).cuda() # 3*t//4,3*t//4+1 # M BN
            rand_st = torch.tensor([0, t//4]).view(-1,1).repeat(1,b).cuda() # t//12, t//6, t//4 # M BN
            rand_mask = torch.arange(t).view(1,1,-1).repeat(MIX,b,1).cuda() # M BN T
            rand_mask = ((rand_mask >= rand_st.unsqueeze(-1)) * (rand_mask < (rand_st+rand_len).unsqueeze(-1)) * 1.) # M BN T


        rand_overlap = rand_mask[0]*rand_mask[1] # B T
        gamma_step = 1./(torch.sum(rand_overlap, dim=-1, keepdim=True)+1) # B 1
        gamma_inc = (torch.arange(1,t+1).cuda().view(1,-1) - torch.max(rand_st,dim=0)[0].view(-1,1)) * gamma_step * rand_overlap
        gamma_dec = (1. - gamma_inc) * rand_overlap
        gamma_dec2 = torch.abs(torch.arange(1,t+1).cuda().view(1,-1) - 0.5*(torch.max(rand_st,dim=0)[0]+torch.min(rand_st+rand_len,dim=0)[0]).view(-1,1)) * 2 * gamma_step * rand_overlap
        gamma_inc2 = (1. - gamma_dec2) * rand_overlap

        rel_st = rand_st[0] - rand_st[1]
        rel_end = (rand_st+rand_len)[0] - (rand_st+rand_len)[1]
        rand_gammas = rand_mask.clone() # M B T

        condition = ((rel_st>=0) * (rel_end>=0)).view(-1,1) * rand_overlap # B T
        rand_gammas[0][condition.bool()] = gamma_inc[condition.bool()]
        rand_gammas[1][condition.bool()] = gamma_dec[condition.bool()]

        condition = ((rel_st<0) * (rel_end<0)).view(-1,1) * rand_overlap # B T
        rand_gammas[0][condition.bool()] = gamma_dec[condition.bool()]
        rand_gammas[1][condition.bool()] = gamma_inc[condition.bool()]

        condition = ((rel_st>=0) * (rel_end<0)).view(-1,1) * rand_overlap # B T
        rand_gammas[0][condition.bool()] = gamma_inc2[condition.bool()]
        rand_gammas[1][condition.bool()] = gamma_dec2[condition.bool()]

        condition = ((rel_st<0) * (rel_end>=0)).view(-1,1) * rand_overlap # B T
        rand_gammas[0][condition.bool()] = gamma_dec2[condition.bool()]
        rand_gammas[1][condition.bool()] = gamma_inc2[condition.bool()]

        mix_in, mix_l, mix_gammas, mix_mask = [inputs], [labels], [prev_gammas], [prev_mask]
        for mix_ind in range(MIX-1):
            rand_index = torch.randperm(b).cuda() if self.training else ((torch.arange(b) + ((mix_ind+1)*b)//MIX + 1) % (b)).cuda()
            mix_in.append(inputs[rand_index])
            mix_l.append(labels[:,rand_index,:])
            mix_gammas.append(prev_gammas[:,rand_index,:])
            mix_mask.append(prev_mask[:,rand_index,:])
        mix_in = torch.stack(mix_in, dim=0) # M B C T H W
        mix_l = torch.cat(mix_l, dim=0) # M B T
        mix_gammas = torch.cat(mix_gammas, dim=0) # M B T
        mix_mask = torch.cat(mix_mask, dim=0) # M B T

        inputs = torch.sum(mix_in * rand_gammas.view(MIX,b,1,t,1,1), dim=0) # B C T H W
        labels = mix_l # M B T

        mix_gammas = mix_gammas * rand_gammas.unsqueeze(1).repeat(1,prev_gammas.shape[0],1,1).view(-1,b,t)
        mix_mask = mix_mask * rand_mask.unsqueeze(1).repeat(1,prev_mask.shape[0],1,1).view(-1,b,t)

        return inputs, labels, mix_gammas, mix_mask

    ################################################# END #############################################################################
    ###################################################################################################################################




def replace_logits(self, n_classes):
        self.fc2 = nn.Linear(2048, n_classes)


def get_inplanes(version):
    planes = {'S':[(54,24), (108,48), (216,96), (432,192)],
              'M':[(54,24), (108,48), (216,96), (432,192)],
              'XL':[(72,32), (162,72), (306,136), (630,280)]}
    return planes[version]


def get_blocks(version):
    blocks = {'S':[3,5,11,7],
              'M':[3,5,11,7],
              'XL':[5,10,25,15]}
    return blocks[version]


def generate_model(x3d_version, **kwargs):
    model = ResNet(Bottleneck, get_blocks(x3d_version), get_inplanes(x3d_version), **kwargs)
    return model
