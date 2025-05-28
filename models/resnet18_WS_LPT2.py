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
import matplotlib.pyplot as plt
import numpy as np
import time, os

import logging
logger = logging.getLogger(__name__)

def plot_tensor_distribution(tensor: torch.Tensor, 
                              title: str = "Tensor Distribution", 
                              bins: int = 100, 
                              save_path: str = "./tensor_distribution.png",
                              add_timestamp: bool = True):

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        tensor = tensor.copy()
    else:
        raise ValueError("Input must be a torch.Tensor or numpy.ndarray.")

    tensor_flat = tensor.flatten()

    # 저장 경로 수정: 타임스탬프 추가
    if add_timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(save_path)
        save_path = f"./weight_figure/{base}_{timestamp}{ext}"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.hist(tensor_flat, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rho=1e-3, init_mode="kaiming_normal"):
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


class BasicBlockWS_LPT(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, rho=1e-3, init_mode="kaiming_normal", quant = None, quant2 = None, quant3 = None, quant4 = None, quant5 = None, quant6 = None,):
        super(BasicBlockWS_LPT, self).__init__()
        self.quant = quant
        self.quant2 = quant2
        self.quant3 = quant3 
        self.quant4 = quant4
        self.quant5 = quant5
        self.quant6 = quant6 
        
        self.conv1 = WSConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
        self.conv2 = WSConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                WSConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, rho=rho, init_mode=init_mode),
                nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(self.expansion*planes)
            ) # quant
            
    def set_rho(self, rho):
        self.conv1.set_rho(rho)
        self.conv2.set_rho(rho)
        if len(self.downsample) > 0:
            self.downsample[0].set_rho(rho)

    def forward_intermediate(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out_i = self.bn1(self.conv1(x))
        out = F.relu(out_i)
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        return out, out_i

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        
        # backward unq
        c_out = self.quant6(x)
        
        out = self.bn1(self.conv1(c_out))
        out_mean = out.mean()
        out_std = out.std(unbiased=False)
        # relu이전의 통계값 계산
        
        out = F.relu(out)
        # out = F.relu(self.bn1(self.conv1(x)))
        
        # plot_tensor_distribution(out)
        # # relu 후 양자화
        # if self.quant is not None:
        #     out = self.quant(out, out_mean, out_std)
            
        if self.quant4 is not None:
            out = self.quant4(out)
            
        out = self.bn2(self.conv2(out))
        
        # plot_tensor_distribution(out)
        # if self.quant2 is not None:
        #     out = self.quant2(out)
            
        if self.quant5 is not None:
            out = self.quant5(out)
            
        dx = self.downsample(c_out)
        
        if len(self.downsample) != 0:
            if self.quant5 is not None:
                dx = self.quant5(dx)
                # plot_tensor_distribution(dx)
                out = out + dx
            else:
                out = out + dx
        else:
            out = out + dx
                
        out_mean = out.mean()
        out_std = out.std(unbiased=False)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        
        # 양자화
        
        
        if not no_relu:
            if self.quant is not None:
            #     # plot_tensor_distribution(out)
                out = self.quant(out, out_mean, out_std)
            # if self.quant4 is not None:
            #     out = self.quant4(out)
                
        return out


class BottleneckWS_LPT(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, rho=1e-3, init_mode="kaiming_normal", quant = None, quant2 = None, quant3 = None, quant4 = None, quant5 = None, quant6 = None):
        super(BottleneckWS_LPT, self).__init__()
        
        self.quant = quant
        self.conv1 = WSConv2d(in_planes, planes, kernel_size=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv2 = WSConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv3 = WSConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn3 = nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                WSConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, rho=rho, init_mode=init_mode),
                nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)
            )
            
    def set_rho(self, rho):
        self.conv1.set_rho(rho)
        self.conv2.set_rho(rho)
        self.conv3.set_rho(rho)
        if len(self.downsample) > 0:
            self.downsample[0].set_rho(rho)

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        if self.quant is not None:
            out = self.quant(out)
            
        out = F.relu(self.bn2(self.conv2(out)))
        if self.quant is not None:
            out = self.quant(out)
            
        out = self.bn3(self.conv3(out))
        if self.quant is not None:
            out = self.quant(out)
        
        dx = self.downsample(x)
        if self.quant is not None:
            dx = self.quant(dx)
            
        out = out + dx
        
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        
        if self.quant is not None:
            out = self.quant(out)
        
        return out


class ResNet_WSConv_LPT(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
                 last_feature_dim=512, rho=1e-3, init_mode="kaiming_normal", quant = None, quant2 = None, quant3 = None, quant4 = None, quant5 = None, quant6 = None, **kwargs):
        
        #use_pretrained means whether to use torch torchvision.models pretrained model, and use conv1 kernel size as 7
        
        super(ResNet_WSConv_LPT, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        conv1_kernel_size = 3
        self.quant = quant
        self.quant2 = quant2
        self.quant3 = quant3
        self.quant4 = quant4
        self.quant5 = quant5
        self.quant6 = quant6
        
        if use_pretrained:
            conv1_kernel_size = 7

        Linear = self.get_linear()   
        self.conv1 = WSConv2d(3, 64, kernel_size=conv1_kernel_size,
                               stride=1, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64) 
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode, quant = self.quant, quant2 = self.quant2, quant3 = self.quant3, quant4 = self.quant4, quant5 = self.quant5, quant6 = self.quant6)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode, quant= self.quant, quant2 = self.quant2, quant3 = self.quant3, quant4 = self.quant4, quant5 = self.quant5, quant6 = self.quant6)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode, quant= self.quant, quant2 = self.quant2, quant3 = self.quant3, quant4 = self.quant4, quant5 = self.quant5, quant6 = self.quant6)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode, quant= self.quant, quant2 = self.quant2, quant3 = self.quant3, quant4 = self.quant4, quant5 = self.quant5, quant6 = self.quant6)

        self.logit_detach = False        

        if use_pretrained:
            resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
            self.layer1.load_state_dict(resnet.layer1.state_dict(), strict=False)
            self.layer2.load_state_dict(resnet.layer2.state_dict(), strict=False)
            self.layer3.load_state_dict(resnet.layer3.state_dict(), strict=False)
            self.layer4.load_state_dict(resnet.layer4.state_dict(), strict=False)

        self.num_layers = 6 # layer0 to layer5 (fc)

        if l2_norm:
            self.fc = Linear(last_feature_dim * block.expansion, num_classes, bias=False)
        else:
            self.fc = Linear(last_feature_dim * block.expansion, num_classes)
            
    def update_all_global_std(self, momentum=0.1):
        """모델 내부의 모든 WSConv2d 레이어에서 update_global_std() 호출"""
        for module in self.modules():  # self.modules()를 사용하면 모델의 모든 서브모듈을 가져옴
            if isinstance(module, WSConv2d):  # WSConv2d인 경우
                module.update_global_std(momentum)
    
    def set_rho(self, rho):
        self.conv1.rho
        self.layer1.set_rho(rho)
        self.layer2.set_rho(rho)
        self.layer3.set_rho(rho)
        self.layer4.set_rho(rho)        

    def get_linear(self):
        return nn.Linear

    def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False, rho=1e-3, init_mode="kaiming_normal", quant = None, quant2 = None, quant3 = None, quant4 = None, quant5 = None, quant6 = None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode, quant= self.quant, quant2 = self.quant2, quant3 = self.quant3, quant4 = self.quant4, quant5 = self.quant5, quant6 = self.quant6))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.l2_norm:
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
            logit = self.fc(out)
        else:
            logit = self.fc(out)
            
        if return_feature==True:
            return out, logit
        else:
            return logit
        
    
    def forward_classifier(self,x):
        logit = self.fc(x)
        return logit        
    
    
    def sync_online_and_global(self):
        state_dict=self.state_dict()
        for key in state_dict:
            if 'global' in key:
                x=(key.split("_global"))
                online=(x[0]+x[1])
                state_dict[key]=state_dict[online]
        self.load_state_dict(state_dict)


class ResNet_WS_LPT(ResNet_WSConv_LPT):

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
            
            out_mean = out0.mean()
            out_std = out0.std(unbiased= False)
            
            out0 = F.relu(out0)
            # plot_tensor_distribution(out0)

            if self.quant is not None:
                out0 = self.quant(out0, out_mean, out_std)
                
            # if self.quant4 is not None:
            #     out0 = self.quant4(out0)

            out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer1'] = out
            
            out_mean = out.mean()
            out_std = out.std(unbiased= False)
            
            out = F.relu(out)
            
            if self.quant is not None:
                out = self.quant(out, out_mean, out_std)

            # if self.quant4 is not None:
            #     out = self.quant4(out)

            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer2'] = out
            out_mean = out.mean()
            out_std = out.std(unbiased= False)
            
            out = F.relu(out)
            
            if self.quant is not None:
                out = self.quant(out, out_mean, out_std)

            # if self.quant4 is not None:
            #     out = self.quant4(out)

            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer3'] = out
            out_mean = out.mean()
            out_std = out.std(unbiased= False)
            
            out = F.relu(out)
            
            if self.quant is not None:
                out = self.quant(out, out_mean, out_std)

            # if self.quant4 is not None:
            #     out = self.quant4(out)
          
            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer4'] = out
            out_mean = out.mean()
            out_std = out.std(unbiased= False)
            
            out = F.relu(out)
            
            if self.quant is not None:
                out = self.quant(out, out_mean, out_std)

            # if self.quant4 is not None:
            #     out = self.quant4(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out_mean = out.mean()
            out_std = out.std(unbiased=False)
            out0 = F.relu(out0)
            
            if self.quant3 is not None:
                out0 = self.quant3(out0)
            
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

        # plot_tensor_distribution(logit)
        
        if self.quant4 is not None:
            logit = self.quant4(logit)
        
        results['feature'] = out
        results['logit'] = logit
        results['layer5'] = logit

        return results

@ENCODER_REGISTRY.register()
class ResNet18_WS_LPT2(ResNet_WS_LPT):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, quant = None, quant2 = None, quant3 = None, quant4= None, quant5= None, quant6= None, **kwargs):
        super().__init__(BasicBlockWS_LPT, [2, 2, 2, 2], num_classes=num_classes, quant= quant, quant2 = quant2, quant3 = quant3, quant4 = quant4, quant5 = quant5, quant6 = quant6, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )

@ENCODER_REGISTRY.register()
class ResNet34_WS_LPT2(ResNet_WS_LPT):

    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlockWS_LPT, [3, 4, 6, 3], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
     
@ENCODER_REGISTRY.register()
class ResNet8_WS_LPT2(ResNet_WS_LPT):

    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlockWS_LPT, [1, 1, 1, 1], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )