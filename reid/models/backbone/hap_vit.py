from functools import partial
from itertools import repeat
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.utils.checkpoint import checkpoint as checkpoint_train
import timm.models.vision_transformer


__all__ = [
    'vit_base_patch16',
    'vit_large_patch16',
    'vit_huge_patch14'
]

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        img_size = to_2tuple(kwargs['img_size'])
        patch_size = to_2tuple(kwargs['patch_size'])
        stride_size_tuple = to_2tuple(kwargs['patch_size'])
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            local_outcome = x[:, 1:]

        return outcome, local_outcome

    def forward(self, x):
        x, local = self.forward_features(x)
        # logit = self.head(x)
        
        return x, local
    

def vit_base_patch16(num_classes, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = VisionTransformer(img_size=img_size, num_classes=num_classes,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate)
    return model


def vit_large_patch16(num_classes, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = VisionTransformer(img_size=img_size, num_classes=num_classes,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate)
    return model


def vit_huge_patch14(num_classes, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = VisionTransformer(img_size=img_size, num_classes=num_classes,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate)
    return model