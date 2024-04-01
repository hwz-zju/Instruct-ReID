from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import random
import sys
import yaml
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import copy
import torch
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
import itertools
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from reid.utils.adamw import AdamW
import torch.nn as nn

from reid import models
from reid.datasets import dataset_entry
from reid.datasets.data_builder_cc import DataBuilder_cc
from reid.datasets.data_builder_sc_mnt import DataBuilder_sc
from reid.datasets.data_builder_t2i import DataBuilder_t2i
from reid.datasets.data_builder_attr import DataBuilder_attr
from reid.datasets.data_builder_cross import DataBuilder_cross
from reid.datasets.data_builder_ctcc import DataBuilder_ctcc
from reid.trainer import TrainerFactory
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineLR
from reid.utils.distributed_utils_pt import dist_init, dist_init_singletask

from reid.multi_tasks_utils.task_info_pt import get_taskinfo
from reid.multi_tasks_utils.multi_task_distributed_utils_pt import Multitask_DistModule
from reid.utils.serialization import load_checkpoint, copy_state_dict
import reid.utils

def configuration():
    parser = argparse.ArgumentParser(description="train simple person re-identification models")

    # distributed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=str, metavar='PATH', default='23446')

    # data
    parser.add_argument('--config', default='scripts/config.yaml')
    parser.add_argument('--data-config', type=str, default=None)
    parser.add_argument('--train-list', type=str, required=True)
    parser.add_argument('--validate', action='store_true', help='validation when training')
    parser.add_argument('--validate_feat', type=str, default='fusion', choices = ['person', 'clothes','fusion'])
    parser.add_argument('--query-list', type=str, default='')
    parser.add_argument('--gallery-list', type=str, default='')
    parser.add_argument('--gallery-list-add', type=str, default=None)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--root_additional', type=str, default=None)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--width_clo', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=1)

    # model
    parser.add_argument('--test_feat_type', type=str, choices=['f','f_c','f_b','b','c'])
    parser.add_argument('-t', '--test_task_type', type=str, default='sc', choices=['cc','sc','ctcc','attr', 't2i', 'cross'])
    parser.add_argument('--dropout_clo', type=float, default=0)
    parser.add_argument('--patch_size_clo', type=int, default=16)
    parser.add_argument('--stride_size_clo', type=int, default=16)
    parser.add_argument('--patch_size_bio', type=int, default=16)
    parser.add_argument('--stride_size_bio', type=int, default=16)
    parser.add_argument('--attn_type', type=str, choices=['mix','dual_attn', 'fc'])
    parser.add_argument('--fusion_loss',type=str)
    parser.add_argument('--fusion_branch', type=str)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--vit_type',type=str)
    parser.add_argument('--vit_fusion_layer',type=int)
    parser.add_argument('--pool_clo', action='store_true')
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--aug', action='store_true',help='whether to add auto augmentor')
    parser.add_argument('--colorjitter', type=str, default='all',help='whether to use colorjitter')
    parser.add_argument('--num_features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--metric', type=str, default='linear')
    parser.add_argument('--scale', type=float, default=30.0)
    parser.add_argument('--metric_margin', type=float, default=0.30)

    #parser.add_argument('--auto_aug', action='store_true', help='wheather to use auto augmentation')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='step_lr', choices=['step_lr', 'cosine_lr'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--aug_lr', type=float, default=0.001, help="learning rate of augmentor parameters")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=1000)
    parser.add_argument('--milestones', nargs='+', type=int, default=[7000, 14000],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--aug_start_iter',type=int,default=0)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--iters', type=int, default=24000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=1000)
    parser.add_argument('--refresh-freq', type=int, default=1000)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    parser.add_argument('--fp16', action='store_true', help="training only")
    parser.add_argument('--loss', type=str, default='ce+tri+bio', help='loss function')
    parser.add_argument('--transe_loss', action='store_true', help="whether to use loss of translation relationship")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
    if 'common' in config:
        for k, v in config['common'].items():
            print(k, v)
            setattr(args, k, v)
    args.config = config
    return args


class Runner(object):
    def __init__(self, args):
        super(Runner, self).__init__()
        if args.data_config is not None:
            with open(args.data_config) as f:
                data_config = yaml.load(f, Loader=yaml.FullLoader)
                # data_config = yaml.load(f)
            args.data_config = data_config
        else:
            args.data_config = None
        self.args = args

    @staticmethod
    def build_optimizer(model, args, lr=None):
        params = []
        if lr is None:
            lr = args.lr
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            opt_args = dict(lr=lr, weight_decay=0.02)
            optimizer = torch.optim.AdamW(params, **opt_args)
            # optimizer = AdamW({lr=1e-05, weight_decay=0.02, lr_custm=0.0001})
        else:
            raise AttributeError('Not support such optimizer `{}`'.format(args.optimizer))

        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        return model, optimizer

    @staticmethod
    def distributed(model, is_distribuited):
        if is_distribuited:
            model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                            find_unused_parameters=True,broadcast_buffers=False)
        return model

    @staticmethod
    def build_scheduler(optimizer, args):
        if args.scheduler == 'step_lr':
            lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01,
                                             warmup_iters=args.warmup_step+args.aug_start_iter)
        elif args.scheduler == 'cosine_lr':
            lr_scheduler = WarmupCosineLR(optimizer, max_iters=args.iters, warmup_factor=0.01,
                                          warmup_iters=args.warmup_step+args.aug_start_iter)
        else:
            raise AttributeError('Not support such scheduler `{}`'.format(args.scheduler))

        return lr_scheduler

    @staticmethod
    def build_trainer(model_dicts, args, this_task_info=None):
        trainer_factory = TrainerFactory()
        if len(model_dicts.keys()) ==1:
            model = model_dicts['extractor']
            trainer = trainer_factory.create(args.arch, model, args, this_task_info=this_task_info)
        else:
            model = model_dicts['extractor']
            model_aug_bio = model_dicts['aug_bio']
            model_aug_clo = model_dicts['aug_clo']
            trainer = trainer_factory.create(args.arch, model, model_aug_bio, model_aug_clo, args, this_task_info=this_task_info)
        return trainer

    @staticmethod
    def build_validator(args, this_task_info=None):
        if not args.validate:
            return None, None, None
        if args.test_task_type == 'cc':
            data_builder = DataBuilder_cc(args)
        elif args.test_task_type == 'sc':
            data_builder = DataBuilder_sc(args)
        elif args.test_task_type == 'attr':
            data_builder = DataBuilder_attr(args, this_task_info)
        elif args.test_task_type == 'ctcc':
            data_builder = DataBuilder_ctcc(args)
        elif args.test_task_type == 't2i':
            data_builder = DataBuilder_t2i(args, this_task_info)
        elif args.test_task_type == 'cross':
            data_builder = DataBuilder_cross(args)
        else:
            AttributeError('Not support such test type `{}`'.format(args.test_task_type))
        test_loader, query_dataset, gallery_dataset = data_builder.build_data(is_train=False)
        return test_loader, query_dataset.data, gallery_dataset.data

    def run(self):
        args = self.args
        if args.data_config is not None:
            rank, world_size = dist_init()
            tasks = args.data_config['tasks']
            this_task_info, all_task_info = get_taskinfo(tasks, world_size, rank)
        else:
            is_distributed = dist_init_singletask(args)
            this_task_info = None
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))
        # Create data loaders
        data_builder = dataset_entry(this_task_info)(args, this_task_info)
        train_loader, train_set = data_builder.build_data(is_train=True)
        if args.aug:
            data_builder_aug = DataBuilder_multi_clo(args)
            train_loader_aug, train_set_aug = data_builder.build_data(is_train=True)
        # Create model
        model = models.create('PASS_Transformer_DualAttn_joint', num_classes=train_set.num_classes, net_config=args, this_task_info=this_task_info)

        checkpoint_p = '<your project root> + instructReID/logs/pretrained/ALBEF.pth'
        if checkpoint_p:
            checkpoint = torch.load(checkpoint_p, map_location='cpu')
            state_dict = checkpoint['model']
            state_dict_new = {}
            for key in state_dict.keys():
                if "visual_encoder" not in key:
                    state_dict_new[key] = state_dict[key]
                elif "visual_encoder_m" in key:
                    state_dict_new[key] = state_dict[key]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
            state_dict_new['visual_encoder_m.pos_embed'] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict_new, strict=False)
        model_path = '<your project root> + instructReID/logs/pretrained/pass_vit_base_full.pth'
        if model_path != '':
            model.visual_encoder.load_param(model_path,hw_ratio=2)
        print(msg)
        
        model.cuda()
        if args.aug:
            # Create aug model
            model_aug_bio = models.create('augmentor', num_classes=3, net_config=args)
            model_aug_bio.cuda()
            model_aug_clo = models.create('augmentor', num_classes=3, net_config=args)
            model_aug_clo.cuda()

        model_dicts = {}
        model, optimizer = self.build_optimizer(model, args)
        if args.data_config:
            # import pdb;pdb.set_trace()
            model = Multitask_DistModule(model, sync=True, ignore=['classifier', 'bank'], task_grp=this_task_info.task_handle, task_root_rank=this_task_info.task_root_rank)
        else:
            model = self.distributed(model, is_distributed)
        lr_scheduler = self.build_scheduler(optimizer, args)
        
        
        model_dicts['extractor'] = model
        if args.aug:
            model_aug_bio, optimizer_bio = self.build_optimizer(model_aug_bio, args, args.aug_lr)
            if args.data_config:
                model_aug_bio = Multitask_DistModule(model_aug_bio, sync=True, ignore=None, task_grp=this_task_info.task_handle)
            else:
                model_aug_bio = self.distributed(model_aug_bio, is_distributed)
            lr_scheduler_bio = self.build_scheduler(optimizer_bio, args)
            model_aug_clo, optimizer_clo = self.build_optimizer(model_aug_clo, args, args.aug_lr)
            if args.data_config:
                model_aug_clo = Multitask_DistModule(model_aug_clo, sync=True, ignore=None, task_grp=this_task_info.task_handle)
            else:
                model_aug_clo = self.distributed(model_aug_clo, is_distributed)
            lr_scheduler_clo = self.build_scheduler(optimizer_clo, args)
            model_dicts['aug_bio'] = model_aug_bio
            model_dicts['aug_clo'] = model_aug_clo

        trainer = self.build_trainer(model_dicts, args, this_task_info=this_task_info)

        test_loader, query, gallery = self.build_validator(args)
        if args.aug:
            trainer.train(train_loader, train_loader_aug, optimizer, optimizer_bio, optimizer_clo, lr_scheduler, lr_scheduler_bio, lr_scheduler_clo, test_loader, query, gallery, aug_start_iter=args.aug_start_iter)   
        else:
            trainer.train(train_loader, optimizer, lr_scheduler, test_loader, query, gallery)

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        return new_pos_embed    
    else:
        return pos_embed_checkpoint

def get_num_layer_for_vit(var_name, config):
    if (var_name == "base" or var_name.endswith("prompt_embed_kv")) and config.get('lpe_lr', False):
        return config['num_layers'] - 1
    if var_name in ("base", "base.cls_token", "base.mask_token"):
        return 0
    elif var_name.startswith("base.patch_embed"):
        return 0
    elif var_name.startswith("base") and not (var_name.startswith("base.norm") or
                                                                var_name.startswith("base.ln_pre")):
        if len(var_name.split('.')) < 3:
            import pdb;pdb.set_trace()
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return config['num_layers']  - 1

def count_parameters_num(model):
    count = 0
    count_fc = 0
    param_dict = {name:param for name,param in model.named_parameters()}
    param_keys = param_dict.keys()
    for m_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count += temp_params.data.nelement()
        elif isinstance(m, nn.Linear):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count_fc += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count_fc += temp_params.data.nelement()
    print('Number of conv/bn params: %.2fM' % (count / 1e6))
    print('Number of linear params: %.2fM' % (count_fc / 1e6))
    return count / 1e6, count_fc / 1e6

class AdamWWithClipDev(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_norm=None, norm_type=2):
        import pdb;pdb.set_trace()
        for param in params:
            import pdb;pdb.set_trace()
            if not isinstance(param['params'], torch.Tensor):
                import pdb;pdb.set_trace()
        super(AdamWWithClipDev, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        import pdb;pdb.set_trace()
        self.clip_norm = clip_norm
        self.norm_type = norm_type

        self._split_param_groups = None
        self.reset_split_param_groups()

    def reset_split_param_groups(self):
        if self.clip_norm is not None:
            backbone_param = []
            for x in self.param_groups:
                if x["params"][0].backbone_specific:
                    backbone_param.append(x["params"])
            self._split_param_groups = [_g for _g in [backbone_param] if len(_g) > 0]
            print(f">>> reset_split_param_groups, backbone_param: {len(backbone_param)}")

    def step(self, closure=None):
        if self.clip_norm is not None:
            for _g in self._split_param_groups:
                all_params = itertools.chain(*_g)
                clip_grad_norm_(all_params, self.clip_norm, self.norm_type)

        super(AdamWWithClipDev, self).step(closure)

if __name__ == '__main__':
    cfg = configuration()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    runner = Runner(cfg)
    runner.run()
