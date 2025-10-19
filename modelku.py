import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from van_utils import load_model_weights
from van_models import VAN


def get_model(args):
    """
    Returns model instance according to args.model_name
    """
    num_classes = getattr(args, "num_classes", 1000)

    if args.model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif args.model_name == 'van':
        embed_dims = [32, 64, 160, 256]
        depths = [3, 3, 5, 2]
        if args.van_arch == 'van_b1':
            embed_dims = [64, 128, 320, 512]
            depths = [2, 2, 4, 2]    
        elif args.van_arch == 'van_b2':
            embed_dims = [64, 128, 320, 512]
            depths = [3, 3, 12, 3]

        elif args.van_arch == 'van_b3':
            embed_dims = [64, 128, 320, 512]
            depths = [3, 5, 27, 3]            
        model = VAN(
            embed_dims=embed_dims, mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths,
            num_classes=args.num_classes)
        model.default_cfg = _cfg()
        model = load_model_weights(model, args.van_arch, args.num_classes)
        return model

    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    return model
