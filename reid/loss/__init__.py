from __future__ import absolute_import

from .adaptive_triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .transloss import TransLoss
from .adv_loss import ClothesBasedAdversarialLoss, CosFaceLoss
__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'TransLoss',
    'ClothesBasedAdversarialLoss',
    'CosFaceLoss'
]
