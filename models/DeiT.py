# https://github.com/berniwal/swin-transformer-pytorch

import numpy as np
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY
from typing import Dict
from omegaconf import DictConfig
import timm

class DeiT_base(nn.Module):
    def __init__(self, args: DictConfig, model_name: str, num_classes: int = 10, **kwargs):
        super().__init__()
        self.num_layers = 4
        use_pretrained = args.model.pretrained
        model_name = args.model.model_name
        
        self.model = timm.create_model(model_name, pretrained=use_pretrained, num_classes=num_classes)
              
              
    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:

        features = self.model.forward_features(x)
        feature_vec = self.model.forward_head(features, pre_logits=True)
        logit = self.model.head(feature_vec)

        results = {}
        results['feature'] = feature_vec
        results['logit'] = logit
        
        return results

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True
        
        
@ENCODER_REGISTRY.register()
class DeiT(DeiT_base):    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(args=args, model_name=args.model.model_name, num_classes=num_classes)

