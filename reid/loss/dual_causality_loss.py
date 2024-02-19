import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from reid.utils import to_numpy


class DualCausalityLoss(nn.Module):
    def __init__(self):
        super(DualCausalityLoss, self).__init__()

    def forward(self, s_dual, label):
        f, fp, fm = s_dual
        pos, negs = self._sample_triplet(label)
        f_ap, f_an = self._forward(f, pos, negs)
        fp_ap, fp_an = self._forward(fp, pos, negs)
        fm_ap, fm_an = self._forward(fm, pos, negs)

        l1 = torch.mean(self.soft_plus(fp_ap - f_ap)) + torch.mean(self.soft_plus(f_an - fp_an))
        l2 = torch.mean(self.soft_plus(f_ap - fm_ap)) + torch.mean(self.soft_plus(fm_an - f_an))
        return l1 + l2

    def _forward(self, f: torch.Tensor, pos, negs):
        n = f.shape[0]
        dist = self.pairwise_distance(f)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][pos[i]].unsqueeze(dim=0))
            dist_an.append(dist[i][negs[i]].unsqueeze(dim=0))
        dist_ap = torch.cat(dist_ap, dim=0)
        dist_an = torch.cat(dist_an, dim=0)
        return dist_ap, dist_an

    @staticmethod
    def _sample_triplet(label):
        label = label.view(-1, 1)
        n = label.shape[0]
        mask = label.expand(n, n).eq(label.expand(n, n).t())
        mask = to_numpy(mask)

        pos, negs = [], []
        for i in range(n):
            pos_indices = np.where(mask[i, :] == 1)
            idx = random.sample(list(pos_indices[0]), 1)[0]
            while idx == i:
                idx = random.sample(list(pos_indices[0]), 1)[0]
            pos.append(idx)
            neg_indices = np.where(mask[i, :] == 0)
            negs.append(random.sample(list(neg_indices[0]), 1)[0])
        return pos, negs

    @staticmethod
    def pairwise_distance(x: torch.Tensor):
        x = F.normalize(x)
        cosine = torch.matmul(x, x.t())
        distmat = -cosine + 0.5
        return distmat

    @staticmethod
    def soft_plus(x):
        return torch.log(1 + torch.exp(x))
