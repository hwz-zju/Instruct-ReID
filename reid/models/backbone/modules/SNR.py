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


class SNR(nn.Module):
    def __init__(self, num_channels):
        super(SNR, self).__init__()
        self.num_features = num_channels
        self.IN = nn.InstanceNorm2d(num_channels, affine=True)
        self.SE = SELayer(num_channels)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        in_x = self.IN(x)
        r = x - in_x
        mask = self.SE(r)
        r_plus = mask * r
        r_minus = (1 - mask) * r

        x_plus = r_plus + in_x
        x_minus = r_minus + in_x

        f = self.avg_pooling(in_x).view(-1, self.num_features)
        fp = self.avg_pooling(x_plus).view(-1, self.num_features)
        fm = self.avg_pooling(x_minus).view(-1, self.num_features)
        return x_plus, (f, fp, fm)
