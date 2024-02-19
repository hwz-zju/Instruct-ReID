from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.layers import create_classifier
from timm.models.resnet import Bottleneck, downsample_conv

__all__ = [
    'lem_base_patch16',
    'lem_large_patch16',
    'lem_huge_patch14'
]

class LocalityEnhancedModule(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(LocalityEnhancedModule, self).__init__(**kwargs)

        in_plane = kwargs['embed_dim']
        self.conv_head = nn.Sequential(
            Bottleneck(in_plane, in_plane // 2, stride=2, downsample=downsample_conv(in_plane, in_plane * 2, kernel_size=1, stride=2)),
            Bottleneck(in_plane * 2, in_plane // 2, stride=1),
            Bottleneck(in_plane * 2, in_plane // 2, stride=1)
        )
        self.conv_global_pool, self.conv_fc = create_classifier(in_plane * 2, self.num_classes)
        self.conv_fc_norm = kwargs['norm_layer'](in_plane * 2)
        self.conv_bn = nn.BatchNorm1d(in_plane * 2)
        self.conv_bn.bias.requires_grad_(False)

        self.bn = nn.BatchNorm1d(kwargs['embed_dim'])
        self.bn.bias.requires_grad_(False)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        C = self.embed_dim
        H, W = self.patch_embed.grid_size
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)    

        for blk in self.blocks:
            x = blk(x)

        _x = x[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)
        _x = self.conv_head(_x)
        _x = self.conv_global_pool(_x)
        _x = self.conv_fc_norm(_x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = (self.fc_norm(x), _x)
        else:
            x = self.norm(x)
            outcome = (x[:, 0], _x)

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            logit = (self.head(self.bn(x[0])), self.conv_fc(self.conv_bn(x[1])))
        return x, logit


def lem_base_patch16(num_classes, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = LocalityEnhancedModule(img_size=img_size, num_classes=num_classes,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate)
    return model


def lem_large_patch16(num_classes, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = LocalityEnhancedModule(img_size=img_size, num_classes=num_classes,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate)
    return model


def lem_huge_patch14(num_classes, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = LocalityEnhancedModule(img_size=img_size, num_classes=num_classes,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate)
    return model
