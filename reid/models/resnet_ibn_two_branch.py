from __future__ import absolute_import

import copy

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch

from reid.models.backbone.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from reid.models.layers.metric import build_metric
from torchvision.models import resnet50

__all__ = ['ResNetIBN', 'resnet_ibn50a_two_branch', 'resnet_ibn101a_two_branch']


class MaskModule(nn.Module):
    def __init__(self, in_channels, num_masks=20):
        super(MaskModule, self).__init__()
        self.in_channels = in_channels
        self.num_masks = num_masks

        self.mask_module = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=self.num_masks, kernel_size=1)
        )

        # self.fc = nn.Linear(self.num_masks*128, 128, bias=False)
        # init.normal(self.fc.weight, std=0.001)

        for sub_module in self.mask_module.modules():
            if isinstance(sub_module, nn.Conv2d):
                nn.init.xavier_normal(sub_module.weight.data)
                sub_module.bias.data.fill_(0)

    def softmax_mask(self, x):
        w = x.shape[2]
        h = x.shape[3]
        x = torch.exp(x)
        sum = torch.sum(x, dim=(2, 3), keepdim=True)
        sum = sum.repeat([1,1,w,h])
        x = x / sum
        return x

    def forward(self, x):
        # mask_feat: BxKxHxW
        mask_feat = self.mask_module(x)
        b, c, h, w = mask_feat.shape
        mask_feat = mask_feat.view(b, c, h*w)
        mask = torch.softmax(mask_feat, dim=-1).view(b, c, h, w)
        # mask = self.softmax_mask(mask_feat)
        # feat
        mask_extend = mask.unsqueeze(1)
        mask = F.max_pool3d(mask_extend, [self.num_masks, 1, 1])
        mask = mask.squeeze(1)
        feat = x
        feat = feat.mul(mask)
        return feat, mask


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, num_classes=0, net_config=None):
        super(ResNetIBN, self).__init__()
        self.depth = depth
        self.net_config = net_config

        resnet = ResNetIBN.__factory[depth](pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.clothes_branch = copy.deepcopy(self.base)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fusion = nn.Linear(4096, 256, bias=False)
        init.normal(self.fusion.weight, std=0.001)

        self.num_classes = num_classes

        out_planes = resnet.fc.in_features

        self.num_features = out_planes

        self.mask_module = MaskModule(2048)

        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.fusion_feat_bn = nn.BatchNorm1d(256)
        self.fusion_feat_bn.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn.weight, 1)
        init.constant_(self.fusion_feat_bn.bias, 0)

        self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        self.classifier_f = nn.Linear(256, self.num_classes, bias=False)
        init.normal_(self.classifier_f.weight, std=0.001)

    def forward(self, x, clot):
        x = self.base(x)
        #x, _ = self.mask_module(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        clot = self.clothes_branch(clot)
        clot = self.gap(clot).view(x.size(0), -1)

        bn_x = self.feat_bn(x)

        fusion_x = torch.cat([x, clot], dim=1)
        fusion_x = self.fusion(fusion_x)
        bn_fusion_x = self.fusion_feat_bn(fusion_x)

        if not self.training:
            bn_fusion_x = F.normalize(bn_fusion_x)
            return bn_x, clot,  bn_fusion_x

        logits = self.classifier(bn_x)
        logits2 = self.classifier_f(bn_fusion_x)
        return x, fusion_x, logits, logits2


def resnet_ibn50a_two_branch(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a_two_branch(**kwargs):
    return ResNetIBN('101a', **kwargs)
