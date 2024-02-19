import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, int(num_channels / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(num_channels / reduction), num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SpatialSNR(nn.Module):
    def __init__(self, num_channels):
        super(SpatialSNR, self).__init__()
        self.num_channels = num_channels
        self.channel_mask = SELayer(self.num_channels)
        self.spatial_mask = nn.Sequential(
            nn.Conv2d(self.num_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid())

        self.IN = nn.InstanceNorm2d(self.num_channels, affine=False)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        in_x = self.IN(x)
        r = x - in_x
        channel_mask = self.channel_mask(r)
        spatial_mask = self.spatial_mask(r)

        p_1 = channel_mask * r
        p_2 = spatial_mask * r

        d_1 = (1 - channel_mask) * r
        d_2 = (1 - spatial_mask) * r

        x = p_1 - 0.1*d_2
        y = p_2 - 0.1*d_1

        m = (x + y) / 2

        f = self.avg_pooling(in_x).view(-1, self.num_channels)
        f_p = self.avg_pooling(in_x + m).view(-1, self.num_channels)
        f_n = self.avg_pooling(in_x + (d_1 + d_2) / 2).view(-1, self.num_channels)

        return m + in_x, (f, f_p, f_n)

