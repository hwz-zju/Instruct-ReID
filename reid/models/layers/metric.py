import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math

from .rbf_layer import MarginRBFLogits, RBFLogits
from .cos_layer import MarginCosineProduct


class CircleLoss(nn.Module):
    def __init__(self, in_features, out_features, s=256, m=0.25):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self._s = s
        self._m = m
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __call__(self, bn_feat, targets):
        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        one_hot = torch.zeros(sim_mat.size(), device=targets.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        pred_class_logits = one_hot * s_p + (1.0 - one_hot) * s_n

        return pred_class_logits


class ArcFace(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type_as(cosine)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class CosFace(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.weight.t(), p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.weight.t(), w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        delt_costh = torch.zeros(costh.size(), device='cuda').scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        return costh_m_s


def build_metric(loss_type, in_dim, out_dim, s=64, m=0.35, **kwargs):
    if loss_type == 'circle':
        return CircleLoss(in_dim, out_dim, s, m)
    elif loss_type == 'arc':
        return ArcFace(in_dim, out_dim, s, m)
    elif loss_type == 'cos':
        return MarginCosineProduct(in_dim, out_dim, s, m)
    elif loss_type == 'am':
        return AMSoftmax(in_dim, out_dim, s, m)
    elif loss_type == 'rbf':
        return MarginRBFLogits(in_dim, out_dim, scale=s, margin=m, **kwargs)
    else:
        assert "Unknown metric {}".format(loss_type)
