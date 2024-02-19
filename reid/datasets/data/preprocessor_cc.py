from __future__ import absolute_import

import os

from PIL import Image, ImageFilter
import cv2 
import numpy as np
from torch.utils.data import Dataset

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
        fname, clothes_fname, pid, cid, cam = self.dataset[index]
        fpath = fname
        clothes_path = clothes_fname

        if int(pid)==-1:
            if self.root_additional is not None:
                fpath = os.path.join(self.root_additional, fname)
                clothes_path = os.path.join(self.root_additional, clothes_fname)
        else:
            if self.root is not None:
                fpath = os.path.join(self.root, fname)
                clothes_path = os.path.join(self.root, clothes_fname)

        img = Image.open(fpath).convert('RGB')
        clothes_img = Image.open(clothes_path).convert('RGB')
        
        if self.blur_clo:
            clothes_img = cv2.cvtColor(np.asarray(clothes_img),cv2.COLOR_RGB2BGR)
            kernel_size = (3, 3)
            sigma = 1.5
            clothes_img = cv2.GaussianBlur(clothes_img, kernel_size, sigma)
            clothes_img = Image.fromarray(cv2.cvtColor(clothes_img, cv2.COLOR_BGR2RGB))
            #print(clothes_img.size)
        if self.transform is not None:
            img = self.transform(img)
            clothes_img = self.clothes_transform(clothes_img)

        return img, clothes_img, fname, clothes_fname, pid, cid, cam, index
