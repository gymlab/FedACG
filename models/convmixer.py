# https://github.com/berniwal/swin-transformer-pytorch

from einops import rearrange, repeat
import numpy as np
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY
from typing import Dict
from omegaconf import DictConfig
from timm.models.layers import DropPath
# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    
    
class ConvMixerBackbone(nn.Module):
    def __init__(self, dim=256, depth=8, kernel_size=9, patch_size=4, in_chans=3):
        super().__init__()
        # patch embedding
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        # convmixer blocks
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)     # (B, dim, H', W')
        x = self.blocks(x)   # (B, dim, H', W')

        return x
    
class ConvMixerTiny(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=10):
        super().__init__()
        self.num_layers = dim
        self.backbone = ConvMixerBackbone(
            dim=dim, depth=depth, kernel_size=kernel_size, patch_size=patch_size, in_chans=3
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes),
        )
        
    def forward(self, x):
        x = self.backbone(x)   # (B, dim, H', W')
        
        results = {}
        results['feature'] = x
        results['logit'] = self.head(x)
        
        return results    
    
    
@ENCODER_REGISTRY.register()
class ConvMixer(ConvMixerTiny):    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(dim=args.model.dim, depth=args.model.depth, kernel_size=args.model.kernel_size, patch_size=args.model.patch_size, n_classes=num_classes)

