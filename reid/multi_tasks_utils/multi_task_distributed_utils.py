import os
import time
import numpy as np
import torch

# import linklink as link
# try:
#     import spring.linklink as link
# except:
#     import linklink as link

class Multitask_DistModule(torch.nn.Module):
    def __init__(self, module, sync=False, ignore=None, task_grp=None):
        super(Multitask_DistModule, self).__init__()
        self.module = module
        self.ignore = ignore
        self.task_grp = task_grp
        # import pdb;pdb.set_trace()
        broadcast_params(self.module, self.ignore, self.task_grp)

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
                    #print('reduce task-specific param {} from {} to {}'.format(name,link.get_rank(),self.task_grp))
                    grad_acc.register_hook(self._make_hook(name, p, self.task_grp))
                    self._grad_accs.append(grad_acc)


    def _make_hook(self, name, p, task_grp=None):
        def hook(*ignore):
            if task_grp:
                link.allreduce_async(name, p.grad.data, group_idx=task_grp)
            else:
                link.allreduce_async(name, p.grad.data)
        return hook


def multitask_reduce_gradients(model, sync=False, ignore_list=None, task_grp=None):
    """ average gradients """
    if sync:
        if ignore_list is not None:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Flag = False
                    # for ignore in ignore_list:
                    #     if ignore in name:
                    #         Flag = True
                    for ignore in ignore_list:
                        if not ignore in name:
                            link.allreduce(param.grad.data)
                            # param.grad
                        else:
                            # print('reduce task-specific param {} from {} to {}'.format(name,link.get_rank(),task_grp))
                            link.allreduce(param.grad.data, group_idx=task_grp)
                elif param.grad is None:
                    param.grad = param.data * 0
                    for ignore in ignore_list:
                        if not ignore in name:
                            link.allreduce(param.grad.data)
                            # param.grad
                        else:
                            # print('reduce task-specific param {} from {} to {}'.format(name,link.get_rank(),task_grp))
                            link.allreduce(param.grad.data, group_idx=task_grp)
        else:
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    link.allreduce(param.grad.data)
    else:
        # multi task 会不会有问题？
        link.synchronize()


def broadcast_params(model, ignore_list=None, task_grp=None):
    """ broadcast model parameters """
    if ignore_list is not None:
        # writer = open(f'rank_{link.get_rank()}.txt','w')
        for name, p in model.state_dict().items():
            Flag = False
            for ignore in ignore_list:
                if ignore in name:
                    Flag = True
            if Flag:
                link.broadcast(p, 0, group_idx=task_grp)
            else:
                link.broadcast(p, 0)
            # for ignore in ignore_list:
                
            #     # import pdb;pdb.set_trace()
            #     if ignore not in name:
            #         # import pdb;pdb.set_trace()
            #         # print('param {} in broadcast'.format(name), 'ignore not in name', ignore not in name)
            #         link.broadcast(p, 0)
            #         # import pdb;pdb.set_trace()
            #     else:
            #         # import pdb;pdb.set_trace()
            #         # print('param {} ignored in broadcast'.format(name), ignore_list)
            #         # continue
            #         # import pdb;pdb.set_trace()
            #         # if link.get_rank()==0:
            #         #     print('broadcasting task-specific param {} from {} to {}'.format(name,link.get_rank(),task_grp))
            #         link.broadcast(p, 0, group_idx=task_grp)
            #         # link.broadcast(p, 0)
            # writer.writelines(f'-Rank {link.get_rank()} {name} , {p.data.view(-1)[0] if p.data.numel() > 0 else -111}\n')
    else:
        for name,p in model.state_dict().items():
            link.broadcast(p, 0)
    # import pdb;pdb.set_trace()
