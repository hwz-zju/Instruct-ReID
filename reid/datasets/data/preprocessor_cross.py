from __future__ import absolute_import

import os

from PIL import Image, ImageFilter
import cv2 
import numpy as np
from torch.utils.data import Dataset
import json
import random
import re

class PreProcessor(Dataset):
    def __init__(self, dataset, json_list=None, root=None, root_additional=None, transform=None, clothes_transform=None, blur_clo=False):
        super(PreProcessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.root_additional = root_additional
        self.transform = transform
        self.initialized = False
        self.clothes_transform = clothes_transform
        self.blur_clo = blur_clo
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)


    def _get_single_item(self, index):
        fname, attr_fname, pid, cid, cam = self.dataset[index]
        fpath = fname
        attr_item = 'cross modiality'
        if int(pid)==-1:
            if self.root_additional is not None:
                fpath = os.path.join(self.root_additional, fname)
        else:
            if self.root is not None:
                fpath = os.path.join(self.root, fname)
                
        img = Image.open(fpath).convert('RGB')
        attribute = pre_caption(attr_item, 50)
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, attribute, fname, attr_fname, pid, cid, cam, index
    
def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')
    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption