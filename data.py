#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:30:51 2024

@author: xingw
"""

import torch.utils.data as data
import glob
import os
import torch as th
import cv2
#     Add by wenzhu
import random
import numpy as np
from utils.utils_test import modcrop

class ImageSet(data.Dataset):
    def __init__(self, img_path, sigma):
        super(ImageSet, self).__init__()
        self.data_files = sorted(glob.glob(os.path.join(img_path+'/','*.png')))
        self.sigma = sigma

    def __getitem__(self, index):
            img_path = self.data_files[index]
#            print(img_path)
            data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = modcrop(data, 32)
            # ---------------------------------
            # augmentation - flip, rotate
            #     Add by wenzhu
            # ---------------------------------
            mode = random.randint(0, 7)
            data = np.ascontiguousarray(self.augment_img(data, mode=mode))
            noisy = np.float32(data/255.) + + np.random.normal(0, self.sigma/255.0, data.shape)
            
            data = th.from_numpy(np.float32(data/255.))
            noisy = th.from_numpy(np.float32(noisy))
            
            
            return data.permute(2,0,1), noisy.permute(2,0,1)

    def __len__(self):
        return len(self.data_files)

    # ---------------------------------
    # augmentation - flip, rotate
    #     Add by wenzhu
    # ---------------------------------
    def augment_img(self, img, mode=0):
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))
