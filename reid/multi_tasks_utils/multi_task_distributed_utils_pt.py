import os
import time
import numpy as np
import torch

import torch.distributed as dist


class Multitask_DistModule(torch.nn.Module):
    def __init__(self, module, sync=False, ignore=None, task_grp=None, task_root_rank=0):
        super(Multitask_DistModule, self).__init__()
        self.module = module
        self.ignore = ignore
        self.task_grp = task_grp
        self.task_root_rank = task_root_rank
        broadcast_params(self.module, self.ignore, self.task_grp, self.task_root_rank)

        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(Multitask_DistModule, self).train(mode)
        self.module.train(mode)

    def _register_hooks(self):
        for i,(name,p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                #if name.startswith(self.prefix):
                if not self.ignore in name:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(name, p))
                    self._grad_accs.append(grad_acc)
                else:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(name, p, self.task_grp))
                    self._grad_accs.append(grad_acc)


    def _make_hook(self, name, p, task_grp=None):
        def hook(*ignore):
            if task_grp:
                allreduce_async(name, p.grad.data, group_idx=task_grp)
            else:
                allreduce_async(name, p.grad.data)
        return hook


def multitask_reduce_gradients(model, sync=False, ignore_list=None, task_grp=None):
    """ average gradients """
    if sync:
        if ignore_list is not None:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    for ignore in ignore_list:
                        if not ignore in name:
                            allreduce(param.grad.data)
                        else:
                            allreduce(param.grad.data, group_idx=task_grp)
                elif param.grad is None:
                    param.grad = param.data * 0
                    for ignore in ignore_list:
                        if not ignore in name:
                            allreduce(param.grad.data)
                            # param.grad
                        else:
                            # print('reduce task-specific param {} from {} to {}'.format(name,link.get_rank(),task_grp))
                            allreduce(param.grad.data, group_idx=task_grp)
        else:
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    allreduce(param.grad.data)
    else:
        dist.synchronize()

def allreduce(x, group_idx=None, ):
    if group_idx == 0:
        group_idx = None
    return dist.all_reduce(x,  group=group_idx)

def allreduce_async(name, x, group_idx=None, ):
    if group_idx == 0:
        group_idx = None
    return dist.all_reduce(x,  group=group_idx)

def broadcast_params(model, ignore_list=None, task_grp=None, task_root_rank=0):
    """ broadcast model parameters """
    if ignore_list is not None:
        for name, p in model.state_dict().items():
            Flag = False
            for ignore in ignore_list:
                if ignore in name:
                    Flag = True
            if Flag:
                dist.broadcast(p, task_root_rank, group=task_grp)
            else:
                dist.broadcast(p, 0)
    else:
        for name,p in model.state_dict().items():
            dist.broadcast(p, 0)
