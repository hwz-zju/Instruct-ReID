from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import sys
import yaml
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from torch import nn

from reid import models
from reid.datasets.data_builder_cc import DataBuilder_cc
from reid.datasets.data_builder_sc_mnt import DataBuilder_sc
from reid.datasets.data_builder_attr import DataBuilder_attr
from reid.datasets.data_builder_t2i import DataBuilder_t2i
from reid.datasets.data_builder_ctcc import DataBuilder_ctcc
from reid.datasets.data_builder_cross import DataBuilder_cross
from reid.evaluation.evaluators_t import Evaluator as Evaluator_t2i
from reid.evaluation.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, copy_state_dict


def main_worker(args):
    log_dir = osp.dirname(args.resume)
    checkpoint_name = args.resume.split('/')[-1].split('.')[0]
    sys.stdout = Logger(osp.join(log_dir, checkpoint_name + 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    if args.test_task_type == 'cc':
        data_builder = DataBuilder_cc(args)
    elif args.test_task_type == 'sc':
        data_builder = DataBuilder_sc(args)
    elif args.test_task_type == 'attr':
        data_builder = DataBuilder_attr(args)
    elif args.test_task_type == 'ctcc':
        data_builder = DataBuilder_ctcc(args)
    elif args.test_task_type == 't2i':
        data_builder = DataBuilder_t2i(args)
    elif args.test_task_type == 'cross':
        data_builder = DataBuilder_cross(args)
    else:
        AttributeError('Not support such test type `{}`'.format(args.test_task_type))
    test_loader, query_dataset, gallery_dataset = data_builder.build_data(is_train=False)

    # Create model
    model = models.create('PASS_Transformer_DualAttn_joint', num_classes=query_dataset.num_classes, net_config=args)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')

    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    if args.test_task_type == 't2i':
        evaluator = Evaluator_t2i(model, args.validate_feat)
    else:
        evaluator = Evaluator(model, args.validate_feat)
    print("Test:")
    evaluator.evaluate(test_loader, query_dataset.data, gallery_dataset.data)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('--config', default='scripts/config.yaml')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--width_clo', type=int, default=128, help="input width")
    parser.add_argument('--query-list', type=str, required=True)
    parser.add_argument('--gallery-list', type=str, required=True)
    parser.add_argument('--gallery-list-add', type=str, default=None)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--vis_root', type=str, default='/mnt/lustre/tangshixiang/hwz/instructReID/data/vis/t2i')
    parser.add_argument('--root_additional', type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--validate_feat', type=str, default='fusion', choices = ['person', 'clothes','fusion'])
    # model
    parser.add_argument('--dropout_clo', type=float, default=0)
    parser.add_argument('--patch_size_clo', type=int, default=16)
    parser.add_argument('--stride_size_clo', type=int, default=16)
    parser.add_argument('--patch_size_bio', type=int, default=16)
    parser.add_argument('--stride_size_bio', type=int, default=16)
    parser.add_argument('--attn_type', type=str, )
    parser.add_argument('--fusion_loss',type=str)
    parser.add_argument('--fusion_branch', type=str)
    parser.add_argument('--vit_type',type=str)
    parser.add_argument('--vit_fusion_layer',type=int)
    parser.add_argument('--test_feat_type', type=str, choices=['f','f_c','f_b','b','c'])
    parser.add_argument('-t', '--test_task_type', type=str, default='cc', choices=['cc','sc','ctcc','attr', 't2i', 'cross'])
    parser.add_argument('--pool_clo', action='store_true')

    parser.add_argument('-a', '--arch', type=str, required=True, choices=models.names())
    parser.add_argument('--num_features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    # testing configs
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--lambda-value', type=float, default=0.3)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    args = parser.parse_args()
    with open(args.config) as f:
        # config = yaml.load(f,Loader=yaml.FullLoader)
        config = yaml.load(f)
    if 'common' in config:
        for k, v in config['common'].items():
            print(k, v)
            setattr(args, k, v)
    args.config = config

    main_worker(args)
