'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import *
from models.build import ENCODER_REGISTRY
from typing import Dict
from omegaconf import DictConfig
import torch.nn.init as init

import logging
logger = logging.getLogger(__name__)

class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rho=1e-3, init_mode="kaiming_uniform"):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.rho = rho
        self.init_mode = init_mode
        self._reset_parameters()
        self.global_std = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.local_std = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        
    # TODO: Check!
    def _reset_parameters(self):
        if self.init_mode == "kaiming_uniform":
            init.kaiming_uniform_(self.weight)
        elif self.init_mode == "kaiming_normal":
            init.kaiming_normal_(self.weight)
        else:
            raise ValueError(f"{self.init_mode} is not supported.")
        
    def set_std(self, std):
        self.local_std.data = torch.full_like(self.local_std.data, std)
        
    def update_global_std(self, momentum=0.1):
        self.global_std.data = self.global_std.data * (1. - momentum) + self.local_std.data * momentum

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight) * self.rho
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def set_rho(self, rho):
        self.rho = rho

class Conv_Net(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32), dataset='cifar10', quant=None, **kwargs):
        super(Conv_Net, self).__init__()
        channel = 3
        self.quant = quant
        self.base, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.quant_layer = self.quant
        self.num_layers = net_depth

    def forward(self, x):
        results = {}

        out = self.base(x)
        results['feature'] = out  

        out = self.classifier(out)
        results['logit_before_quant'] = out  

        out = self.quant_layer(out)
        results['logit'] = out  

        results['layer3'] = out 
        
        return results

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
            # return nn.GroupNorm(1, shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            layers += [self.quant]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            layers += [self.quant]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
                layers += [self.quant]
        layers += [nn.Flatten()]

        return nn.Sequential(*layers), shape_feat

@ENCODER_REGISTRY.register()
class ConvNet(Conv_Net):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, quant = None, **kwargs):
        super().__init__(num_classes=num_classes, quant= quant, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
