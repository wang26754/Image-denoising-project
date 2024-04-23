#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:10:33 2023

@author: xingw
"""
import numpy as np
import pywt

def augment_img(img, mode=0):
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

"""
Add on Thu Jan 04
By Wenzhu Xing
"""

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def dwt_LL(data):
    (H,W,C) = data.shape
    LL = np.zeros((H//2, W//2, C))
    for c in range(3):
        coeffs2 = pywt.dwt2(data[:,:,c], 'haar')
        LL[:,:,c], _ = coeffs2
    return LL

def dwt_LH(data):
    (H,W,C) = data.shape
    LH = np.zeros((H//2, W//2, C))
    for c in range(3):
        coeffs2 = pywt.dwt2(data[:,:,c], 'haar')
        _, (LH[:,:,c], _, _) = coeffs2
    return LH

def dwt_HL(data):
    (H,W,C) = data.shape
    HL = np.zeros((H//2, W//2, C))
    for c in range(3):
        coeffs2 = pywt.dwt2(data[:,:,c], 'haar')
        _, (_, HL[:,:,c], _) = coeffs2
    return HL

def dwt_HH(data):
    (H,W,C) = data.shape
    HH = np.zeros((H//2, W//2, C))
    for c in range(3):
        coeffs2 = pywt.dwt2(data[:,:,c], 'haar')
        _, (_, _, HH[:,:,c]) = coeffs2
    return HH


    
    