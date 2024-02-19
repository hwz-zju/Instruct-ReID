from __future__ import absolute_import

from .resnet import *
from .resnet_ibn import *
from .resnext_ibn import *
from .se_resnet_ibn import *
from .efficientnet import *
from .mgn import MGN
from .resnet_ibn_snr import *
from .resnet_ibn_two_branch import *
from .transformer import *
from .augmentor import *
from .transformer import Transformer_local, Transformer_DualAttn, Transformer_DualAttn_multi
from .pass_transformer_joint import PASS_Transformer_DualAttn_joint

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'resnext_ibn101a': resnext101_ibn_a,
    'se_resnet_ibn101a': se_resnet101_ibn_a,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'mgn': MGN,
    'resnet_ibn50a_snr': resnet_ibn50a_snr,
    'resnet_ibn101a_snr': resnet_ibn101a_snr,
    'resnet_ibn50a_snr_spatial': resnet_ibn50a_snr_spatial,
    'resnet_ibn101a_snr_spatial': resnet_ibn101a_snr_spatial,
    'resnet_ibn50a_two_branch': resnet_ibn50a_two_branch,
    'resnet_ibn101a_two_branch': resnet_ibn101a_two_branch,
    # 'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    # 'deit_small_patch16_224_TransReID_mask': deit_small_patch16_224_TransReID_mask,
    # 'deit_small_patch16_224_TransReID_aug': deit_small_patch16_224_TransReID,
    # 'deit_small_patch16_224_TransReID_mask_aug': deit_small_patch16_224_TransReID_mask,
    'augmentor': Augmentor,
    'transformer': Transformer_local,
    'transformer_dualattn': Transformer_DualAttn,
    'transformer_dualattn_multi': Transformer_DualAttn_multi,
    'PASS_Transformer_DualAttn_joint': PASS_Transformer_DualAttn_joint,
    'transformer_dualattn_joint': PASS_Transformer_DualAttn_joint,

}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    net_config  : ArgumentParser
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](**kwargs)
