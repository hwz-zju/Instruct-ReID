import torch
import torch.nn as nn
from torch.nn import Parameter

def l2_dist(x, y):
    m, n = x.size(0), y.size(0)
    # x = x.view(m, -1)
    # y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_m.addmm_(1, -2, x, y.t())
    dist_m = dist_m - 2 * torch.mm(x,y.t())
    return dist_m


class RBFLogits(nn.Module):
    # 8~35, 2/4~16
    def __init__(self, feature_dim, class_num, scale = 16.0, gamma = 8.0):
        super(RBFLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weight = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        # self.bias = nn.Parameter(torch.FloatTensor(class_num))
        self.scale = scale
        self.gamma = gamma
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, label):
        # weight: L*2048 -> 1*L*2048
        # feat: B*2048 -> B*1*2048
        # diff: B*L*2048

        # diff = torch.unsqueeze(self.weight, dim=0) - torch.unsqueeze(feat, dim=1)
        # diff = torch.mul(diff, diff)
        # metric = torch.sum(diff, dim=-1)
        kernal_metric = l2_dist(feat, self.weight)
        kernal_metric = torch.exp(-1.0 * kernal_metric / self.gamma)
        logits = self.scale * kernal_metric
        return logits


class MarginRBFLogits(nn.Module):
    def __init__(self, feature_dim, class_num, scale = 35.0, gamma = 16.0, margin=0.1):
        super(MarginRBFLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weight = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        # self.bias = nn.Parameter(torch.FloatTensor(class_num))
        self.scale = scale
        self.gamma = gamma
        self.margin = margin
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, label):
        # diff = torch.unsqueeze(self.weight, dim=0) - torch.unsqueeze(feat, dim=1)
        # diff = torch.mul(diff, diff)
        # metric = torch.sum(diff, dim=-1)
        metric = l2_dist(feat, self.weight)
        kernal_metric = torch.exp(-1.0 * metric / self.gamma)

        if self.training:
            phi = kernal_metric - self.margin
            one_hot = torch.zeros(kernal_metric.size()).cuda()
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            train_logits = self.scale * ((one_hot * phi) + ((1.0 - one_hot) * kernal_metric))
            return train_logits
        else:
            test_logits = self.scale * kernal_metric
            return test_logits
