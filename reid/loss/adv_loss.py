import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)

        # dist.reduce_scatter(grad_out, list(grads))
        # grad_out.div_(dist.get_world_size())

        grad_out[:] = grads[dist.get_rank()]

        return grad_out
    
class CosFaceLoss(nn.Module):
    """ CosFace Loss based on the predictions of classifier.

    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """
    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (inputs - one_hot * self.m)

        return F.cross_entropy(output, targets)
  
class ClothesBasedAdversarialLoss(nn.Module):
    """ Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
        """
        inputs = self.scale * inputs
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()

        return loss