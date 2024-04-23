#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:00:35 2024

@author: xingw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:35:39 2024

@author: xingw
"""

import torch.utils.data as data
import glob
import os
from matplotlib import pyplot as plt
from PIL import Image
import torch as th
import torch.nn.functional as Fn
from scipy import ndimage
import cv2
import torch
#from function_stdEst import *
#     Add by wenzhu
import random
import numpy as np
from utils.utils_test import modcrop, dwt_LL, dwt_HH, dwt_LH, dwt_HL

def concatenate_images(img1, img2, img3, img4):
    """
    Concatenate four images (256, 256, 3) into one (512, 512, 3).
    Args:
    - img1, img2, img3, img4: Arrays of shape (256, 256, 3)

    Returns:
    - concatenated_img: Array of shape (512, 512, 3)
    """
    # Create the first row by concatenating img1 and img2 horizontally
    top_half = torch.concat((img1, img2), dim=1)

    # Create the second row by concatenating img3 and img4 horizontally
    bottom_half = torch.concat((img3, img4), dim=1)

    # Concatenate the two halves vertically to form the final image
    concatenated_img = torch.concat((top_half, bottom_half), dim=0)

    return concatenated_img


# Concatenate the images

class DWTImageSet(data.Dataset):
    def __init__(self, img_path, sigma, dim):
        super(DWTImageSet, self).__init__()
        self.data_files = sorted(glob.glob(os.path.join(img_path,'*.png'))) 
        self.sigma = sigma
        self.dim = dim
        print('Number of acquired images = ',len(self.data_files))
        

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
        if self.dim == 'concat':
            data_LL = torch.from_numpy(dwt_LL(data / 255.).astype(np.float32))
            noisy_LL = torch.from_numpy(dwt_LL(noisy).astype(np.float32))
            
            data_LH = torch.from_numpy(dwt_LH(data / 255.).astype(np.float32))
            noisy_LH = torch.from_numpy(dwt_LH(noisy).astype(np.float32))
            
            data_HL = torch.from_numpy(dwt_HL(data / 255.).astype(np.float32))
            noisy_HL = torch.from_numpy(dwt_HL(noisy).astype(np.float32))
            
            data_HH = torch.from_numpy(dwt_HH(data / 255.).astype(np.float32))
            noisy_HH = torch.from_numpy(dwt_HH(noisy).astype(np.float32))
            
            # Concatenate the DWT outputs along a new dimension (channel dimension assumed)
            
            data = concatenate_images(data_LL, data_LH, data_HL, data_HH)
            noisy = concatenate_images(noisy_LL, noisy_LH, noisy_HL, noisy_HH)
            
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
