from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine

def cosine_simalirity(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p1 = sorted_mat_distance[:, 0]
    hard_p_indice1 = positive_indices[:, 0]
    
    hard_p2 = sorted_mat_distance[:, 1]
    hard_p_indice2 = positive_indices[:, 1]
    
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    # import pdb;pdb.set_trace()
    if (indice):
        return hard_p1, hard_p2, hard_n, hard_p_indice1, hard_p_indice2, hard_n_indice
    return hard_p1, hard_p2, hard_n

def _batch_hard_(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

class TripletLoss(nn.Module):

    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, emb, label, clot_feats_s):
        if self.normalize_feature:
            # equal to cosine similarity
            emb = F.normalize(emb)
        mat_dist = euclidean_dist(emb, emb)
        
        mat_dist_clot_feats_s = cosine_simalirity(clot_feats_s, clot_feats_s)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        
        dist_ap1, dist_ap2, dist_an, dist_ap1_indice, dist_ap2_indice, dist_an_indice = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap1.size(0)
        
        alpha1 = torch.rand(dist_ap1_indice.shape).to(dist_ap1_indice.device)
        for b_index1, index1 in enumerate(dist_ap1_indice):
            alpha1[b_index1] = mat_dist_clot_feats_s[b_index1][index1].detach()
        
        alpha2 = torch.rand(dist_ap2_indice.shape).to(dist_ap2_indice.device)
        for b_index2, index2 in enumerate(dist_ap2_indice):
            alpha2[b_index2] = mat_dist_clot_feats_s[b_index2][index2].detach()
        
        alphan = torch.rand(dist_an_indice.shape).to(dist_an_indice.device)
        for b_indexn, indexn in enumerate(dist_an_indice):
            alphan[b_indexn] = mat_dist_clot_feats_s[b_indexn][indexn].detach()
        
        y11 = torch.ones_like(dist_ap1)
        y11_m = torch.ones_like(dist_ap1)
        y11[alpha1 < alpha2] = -1
        y11_m[alpha1 == alpha2] = 0
        
        loss11 = self.margin_loss(dist_ap2*y11_m, dist_ap1*y11_m + self.margin*(alpha1 - alpha2 - y11), y11)
        
        y13 = torch.ones_like(dist_ap1)
        
        dist_ap1 =  dist_ap1 + self.margin*(alpha1 - 1)
        
        loss13 = self.margin_loss(dist_an, dist_ap1, y13)
        
        y23 = torch.ones_like(dist_ap2)
        
        dist_ap2 =  dist_ap2 + self.margin*(alpha2 - 1)
        
        loss23 = self.margin_loss(dist_an, dist_ap2, y23)
        loss = 0.1 * loss11 + loss13
        prec = (dist_an.data > dist_ap1.data).sum() * 1. / y11.size(0)
        return loss, prec


class SoftTripletLoss(nn.Module):

    def __init__(self, margin=None, normalize_feature=False):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

    def forward(self, emb1, emb2, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = self.logsoftmax(triple_dist)
        if self.margin is not None:
            loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            return loss

        mat_dist_ref = euclidean_dist(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = self.softmax(triple_dist_ref).detach()

        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss
