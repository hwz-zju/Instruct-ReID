from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init

#from efficientnet_pytorch import EfficientNet
from reid.models.layers.metric import build_metric

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
           'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5']


class Efficient(nn.Module):
    __factory = {
        '0': 'efficientnet-b0',
        '1': 'efficientnet-b1',
        '2': 'efficientnet-b2',
        '3': 'efficientnet-b3',
        '4': 'efficientnet-b4',
        '5': 'efficientnet-b5',
    }

    def __init__(self, depth, num_classes=0, net_config=None):
        super(Efficient, self).__init__()

        self.depth = depth
        self.net_config = net_config

        self.base = EfficientNet.from_pretrained(self.__factory[depth])

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_features = self.net_config.num_features
        self.dropout = self.net_config.dropout
        self.has_embedding = self.net_config.num_features > 0
        self.num_classes = num_classes

        out_planes = self.base._fc.in_features

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

        # if not pretrained:
        #     self.reset_params()

    def forward(self, x, y=None):
        x = self.base.extract_features(x)
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

        return x, bn_x, logits

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def efficientnet_b0(**kwargs):
    return Efficient('0', **kwargs)


def efficientnet_b1(**kwargs):
    return Efficient('1', **kwargs)


def efficientnet_b2(**kwargs):
    return Efficient('2', **kwargs)


def efficientnet_b3(**kwargs):
    return Efficient('3', **kwargs)


def efficientnet_b4(**kwargs):
    return Efficient('4', **kwargs)


def efficientnet_b5(**kwargs):
    return Efficient('5', **kwargs)
