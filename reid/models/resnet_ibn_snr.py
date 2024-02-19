from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init

from reid.models.backbone.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from reid.models.layers.metric import build_metric

from .backbone.modules.SNR import SNR
from .backbone.modules.spatial_snr import SpatialSNR

__all__ = ['ResNetIBNSNR', 'resnet_ibn50a_snr', 'resnet_ibn101a_snr',
           'resnet_ibn50a_snr_spatial', 'resnet_ibn101a_snr_spatial']


class ResNetIBNSNR(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, spatial=False, num_classes=0, net_config=None):
        super(ResNetIBNSNR, self).__init__()
        self.depth = depth
        self.net_config = net_config

        resnet = ResNetIBNSNR.__factory[depth](pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            )
        self.layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
        if spatial:
            self.decouple = nn.ModuleList([SpatialSNR(256), SpatialSNR(512), SpatialSNR(1024), SpatialSNR(2048)])
        else:
            self.decouple = nn.ModuleList([SNR(256), SNR(512), SNR(1024), SNR(2048)])
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_features = self.net_config.num_features
        self.dropout = self.net_config.dropout
        self.has_embedding = self.net_config.num_features > 0
        self.num_classes = num_classes

        out_planes = resnet.fc.in_features

        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            self.num_features = out_planes

        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if self.num_classes > 0:
            if self.net_config.metric == 'linear':
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
            else:
                self.classifier = build_metric(self.net_config.metric, self.num_features,
                                               self.num_classes, self.net_config.scale, self.net_config.metric_margin)

    def forward(self, x, y=None):
        x = self.base(x)

        dual_feats = []
        for item, layer in enumerate(self.layers):
            x = layer(x)
            x, f = self.decouple[item](x)
            dual_feats.append(f)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if not self.training:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            if isinstance(self.classifier, nn.Linear):
                logits = self.classifier(bn_x)
            else:
                logits = self.classifier(bn_x, y)
        else:
            return bn_x

        return x, bn_x, logits, dual_feats


def resnet_ibn50a_snr(**kwargs):
    return ResNetIBNSNR('50a', **kwargs)


def resnet_ibn101a_snr(**kwargs):
    return ResNetIBNSNR('101a', **kwargs)


def resnet_ibn50a_snr_spatial(**kwargs):
    return ResNetIBNSNR('50a', spatial=True, **kwargs)


def resnet_ibn101a_snr_spatial(**kwargs):
    return ResNetIBNSNR('101a', spatial=True, **kwargs)