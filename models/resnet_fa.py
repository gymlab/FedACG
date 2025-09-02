'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from utils import *
import numpy as np
from models.resnet import BasicBlock, Bottleneck, ResNet
from models.build import ENCODER_REGISTRY
from typing import Callable, Dict, Tuple, Union, List, Optional
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)

class ResNet_base(ResNet):

    def forward_layer(self, layer, x, no_relu=True):

        if isinstance(layer, nn.Linear):
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            out = layer(x)
        else:
            if no_relu:
                out = x
                for sublayer in layer[:-1]:
                    out = sublayer(out)
                out = layer[-1](out, no_relu=no_relu)
            else:
                out = layer(x)

        return out
    
    def forward_layer_by_name(self, layer_name, x, no_relu=True):
        layer = getattr(self, layer_name)
        return self.forward_layer(layer, x, no_relu)

    def forward_layer0(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out0 = self.bn1(self.conv1(x))
        if not no_relu:
            out0 = F.relu(out0)
        return out0

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'fc' not in n:
            # if True:
                p.requires_grad = False
        logger.warning('Freeze backbone parameters (except fc)')
        return
    
    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer1'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer2'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer3'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer4'] = out
            out = F.relu(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.logit_detach:
            logit = self.fc(out.detach())
        else:
            logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results['layer5'] = logit

        return results


class ResNetFA(ResNet_base):
    def __init__(self, *args, prob=0.5, momentum1=0.99, momentum2=0.99, **kwargs):
        super().__init__(*args, **kwargs)

        # ResNet-18/34 기준 채널 구성
        self.ffa0 = FFALayer(prob=prob, momentum1=momentum1, momentum2=momentum2, nfeat=64)
        self.ffa1 = FFALayer(prob=prob, momentum1=momentum1, momentum2=momentum2, nfeat=64)
        self.ffa2 = FFALayer(prob=prob, momentum1=momentum1, momentum2=momentum2, nfeat=128)
        self.ffa3 = FFALayer(prob=prob, momentum1=momentum1, momentum2=momentum2, nfeat=256)
        self.ffa4 = FFALayer(prob=prob, momentum1=momentum1, momentum2=momentum2, nfeat=512)

    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            # ---- stem ----
            out0 = self.bn1(self.conv1(x))      # [B,64,H/2,W/2] (구현에 따라 maxpool 유무 다름)
            results['layer0'] = out0
            # FFA: stem 출력 직후, ReLU 전에 통과
            out0 = self.ffa0(out0)
            out0 = F.relu(out0)

            # ---- layer1 ----
            out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer1'] = out
            # FFA 후 ReLU
            out = self.ffa1(out)
            out = F.relu(out)

            # ---- layer2 ----
            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer2'] = out
            out = self.ffa2(out)
            out = F.relu(out)

            # ---- layer3 ----
            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer3'] = out
            out = self.ffa3(out)
            out = F.relu(out)

            # ---- layer4 ----
            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer4'] = out
            out = self.ffa4(out)
            out = F.relu(out)

        else:
            # no_relu=False 경로에서도 동일하게 FFA를 stage 출력 직후에 삽입
            out0 = self.bn1(self.conv1(x))
            out0 = self.ffa0(out0)
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = self.layer1(out0)
            results['layer1'] = out
            out = self.ffa1(out); out = F.relu(out)

            out = self.layer2(out)
            results['layer2'] = out
            out = self.ffa2(out); out = F.relu(out)

            out = self.layer3(out)
            results['layer3'] = out
            out = self.ffa3(out); out = F.relu(out)

            out = self.layer4(out)
            results['layer4'] = out
            out = self.ffa4(out); out = F.relu(out)

        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        logit = self.fc(out.detach() if self.logit_detach else out)
        results['feature'] = out
        results['logit'] = logit
        results['layer5'] = logit
        return results



class FFALayer(nn.Module):
    def __init__(self, prob=0.5, eps=1e-6, momentum1=0.99, momentum2=0.99, nfeat=None):
        super(FFALayer, self).__init__()
        self.prob = prob
        self.eps = eps
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.nfeat = nfeat

        self.register_buffer('running_var_mean_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_var_std_bmic', torch.ones(self.nfeat))
        self.register_buffer('running_mean_bmic', torch.zeros(self.nfeat))
        self.register_buffer('running_std_bmic', torch.ones(self.nfeat))

    def forward(self, x):
        if not self.training: return x
        if np.random.random() > self.prob: return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps)
        std = std.sqrt()

        self.momentum_updating_running_mean_and_std(mean, std)

        var_mu = self.var(mean)
        var_std = self.var(std)

        running_var_mean_bmic = 1 / (1 + 1 / (self.running_var_mean_bmic + self.eps))
        gamma_mu = x.shape[1] * running_var_mean_bmic / sum(running_var_mean_bmic)

        running_var_std_bmic = 1 / (1 + 1 / (self.running_var_std_bmic + self.eps))
        gamma_std = x.shape[1] * running_var_std_bmic / sum(running_var_std_bmic)

        var_mu = (gamma_mu + 1) * var_mu
        var_std = (gamma_std + 1) * var_std

        var_mu = var_mu.sqrt().repeat(x.shape[0], 1)
        var_std = var_std.sqrt().repeat(x.shape[0], 1)

        beta = self.gaussian_sampling(mean, var_mu)
        gamma = self.gaussian_sampling(std, var_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    def gaussian_sampling(self, mu, std):
        e = torch.randn_like(std)
        z = e.mul(std).add_(mu)
        return z

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def var(self, x):
        t = x.var(dim=0, keepdim=False) + self.eps
        return t

    def momentum_updating_running_mean_and_std(self, mean, std):
        with torch.no_grad():
            self.running_mean_bmic = self.running_mean_bmic * self.momentum1 + \
                                     mean.mean(dim=0, keepdim=False) * (1 - self.momentum1)
            self.running_std_bmic = self.running_std_bmic * self.momentum1 + \
                                    std.mean(dim=0, keepdim=False) * (1 - self.momentum1)

    def momentum_updating_running_var(self, var_mean, var_std):
        with torch.no_grad():
            self.running_var_mean_bmic = self.running_var_mean_bmic * self.momentum2 + var_mean * (1 - self.momentum2)
            self.running_var_std_bmic = self.running_var_std_bmic * self.momentum2 + var_std * (1 - self.momentum2)
            
@ENCODER_REGISTRY.register()
class ResNet18_FA(ResNetFA):
    def __init__(self, args: DictConfig, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

            
@ENCODER_REGISTRY.register()
class ResNet34_FA(ResNetFA):
    def __init__(self, args: DictConfig, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    
