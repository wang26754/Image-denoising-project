#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:53:55 2024

@author: JUndong
"""


import math
import torch
import torch.nn as nn
import numpy as np
#from thop import profile
from einops import rearrange 
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
from unet.basicblock import PixelUnShuffle, PatchUnEmbedding
import torch.nn.functional as F
import pywt
import numpy
from unet.network_dncnn import DnCNN
from unet.network_scunet import SCUNet

# Add by Jundong 

def concatenate_images(img1, img2, img3, img4):
    """
    Concatenate four images (256, 256, 3) into one (512, 512, 3).
    Args:
    - img1, img2, img3, img4: Arrays of shape (256, 256, 3)

    Returns:
    - concatenated_img: Array of shape (512, 512, 3)
    """
    # Create the first row by concatenating img1 and img2 horizontally
    top_half = torch.concat((img1, img2), dim=3)

    # Create the second row by concatenating img3 and img4 horizontally
    bottom_half = torch.concat((img3, img4), dim=3)

    # Concatenate the two halves vertically to form the final image
    concatenated_img = torch.concat((top_half, bottom_half), dim=2)
    return concatenated_img
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_LH, x_HL, x_HH

def iwt_init(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel // (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h            
class LargeDwtnet(nn.Module):
    def __init__(self):
        super(LargeDwtnet, self).__init__()
        self.scunet_ll = SCUNet(in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128)
        self.scunet_lh = SCUNet(in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128)
        self.scunet_hl = SCUNet(in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128)
        self.scunet_hh = SCUNet(in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128)
        self.scunet_m = SCUNet(in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128)
        self.dncnn = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')


        #self.weight_ll = nn.Parameter(torch.rand(1))
        #self.weight_lh = nn.Parameter(torch.rand(1))
        #self.weight_hl = nn.Parameter(torch.rand(1))
        #self.weight_hh = nn.Parameter(torch.rand(1))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        ll, lh, hl, hh = [], [], [], []
        #noisy_img = noisy_img.float()
    
       
       # Assuming DWT is done correctly and returns tensors
        #for i in range(batch_size):
           #print(noisy_img[i].shape)
         #  recon1, recon2 = torch.split(noisy_img[i], 256, dim=1)
           #LL,  LH = torch.split(recon1, 256, dim=2)
           #HL,  HH = torch.split(recon2, 256, dim=2)
           #ll.append(LL)  # Convert to tensors
          # lh.append(LH)
          # hl.append(HL)
           #hh.append(HH)


       # Convert lists to tensor
        ll, lh, hl, hh = dwt_init(x)


        #x_cat = torch.cat((ll, hl, lh, hh), 1)
        #x = concatenate_images(ll, hl, lh, hh)
        # Process each component
        ll_out = self.scunet_ll(ll)
        lh_out = self.scunet_lh(lh)
        hl_out = self.scunet_hl(hl)
        hh_out = self.scunet_hh(hh)
        ll_out  = ll_out + ll
        lh_out  = lh_out + lh
        hl_out  = hl_out + hl
        hh_out  = hh_out + hh
        ll_out = self.scunet_m(ll_out)
        dwt_cat = torch.cat((ll_out, hl_out , lh_out, hh_out), 1)
        #dwt_cat = self.dncnn(dwt_ca)
        #dwt_cat = dwt_cat + dwt_ca
        # Apply weights
        #ll_out = self.sigmoid(self.weight_ll) * ll_out
        #lh_out = self.sigmoid(self.weight_lh) * lh_out
        #hl_out = self.sigmoid(self.weight_hl) * hl_out
        #hh_out = self.sigmoid(self.weight_hh) * hh_out
        #print(ll_out.shape)
        #dwt_cat = torch.cat((ll_out, hl_out , lh_out, hh_out), 1)
        
        reconstructed_x = iwt_init(dwt_cat)
        
        residual = x - reconstructed_x
        reconstructed_x = x - residual
        
        return reconstructed_x



if __name__ == '__main__':

    # torch.cuda.empty_cache()
    net = LargeDwtnet()

    x = torch.randn((2, 3, 512, 512))
    x = net(x)
    print(x.shape)