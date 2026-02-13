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
import timm
from timm.models import load_checkpoint
# from transformers import AutoImageProcessor, AutoModelForImageClassification

class DeiT_base(nn.Module):
    def __init__(self, args: DictConfig, model_name: str, num_classes: int = 10, **kwargs):
        super().__init__()
        self.num_layers = 4
        use_pretrained = True
        if hasattr(args, 'model') and hasattr(args.model, 'pretrained'):
            use_pretrained = args.model.pretrained
            model_name = args.model.model_name

        # print(f"Loading model from Hugging Face: {model_name}...")
        
        self.model = timm.create_model(model_name, pretrained=use_pretrained)
        # load_checkpoint(self.model, 'hf_hub:hiendang7613/vit-l-tiny-imagenet')
            # Load model directly

        # processor = AutoImageProcessor.from_pretrained("hiendang7613/vit-l-tiny-imagenet")
        # model = AutoModelForImageClassification.from_pretrained("hiendang7613/vit-l-tiny-imagenet")
        
        if hasattr(self.model, 'head'):
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
        
    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:

        features = self.model.forward_features(x)
        feature_vec = self.model.forward_head(features, pre_logits=True)
        logit = self.model.head(feature_vec)

        results = {}
        results['feature'] = feature_vec
        results['logit'] = logit
        
        return results

    def freeze_backbone(self):
        for name, p in self.model.named_parameters():
            if 'head' not in name:
                p.requires_grad = False
        print('Freeze backbone parameters (except head)')
        
        
@ENCODER_REGISTRY.register()
class DeiT(DeiT_base):    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(args=args, model_name=args.model.model_name, num_classes=num_classes)

