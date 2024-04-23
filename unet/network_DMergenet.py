#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:06:41 2024

@author: Jundong
"""
# -*- coding: utf-8 -*-
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

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1) #(B,C*4,H/2,W/2)

def iwt_init(x):
    
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel // (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width])

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h
    
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False 

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class RB(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(RB, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        c0 = self.conv(x)
        x = self.block(x)
        return x + c0

class NRB(nn.Module):
    def __init__(self, n, in_size, out_size, relu_slope):
        super(NRB, self).__init__()
        nets = []
        for i in range(n):
            nets.append(RB(in_size, out_size, relu_slope))
        self.body = nn.Sequential(*nets)

    def forward(self, x):
        return self.body(x)
    
    
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


class DMergenet(nn.Module):
    def __init__(self, in_chn=3, wf=16, relu_slope=0.2):
        super(DMergenet, self).__init__()        

        self.l1 = RB(in_size=in_chn, out_size=wf, relu_slope=relu_slope)
        self.down1 = DWT()

        self.l2 = RB(in_size=wf*4, out_size=wf*4, relu_slope=relu_slope)
        self.down2 = DWT()

        self.l3 = RB(in_size=wf*16, out_size=wf*16, relu_slope=relu_slope)
        self.down3 = DWT()
        
        self.l4 = RB(in_size=wf*64, out_size=wf*64, relu_slope=relu_slope)
        
        self.up3 = IWT()
        self.u3 = RB(in_size=wf*32, out_size=wf*16, relu_slope=relu_slope)

        self.up2 = IWT()
        self.u2 = RB(in_size=wf*8, out_size=wf*4, relu_slope=relu_slope)

        self.up1 = IWT()
        self.u1 = RB(in_size=wf*2, out_size=wf, relu_slope=relu_slope)
        
                               
        self.st_1 = SCUNet(in_nc=wf, out_nc=wf, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128)
        
        #self.st_2 = SCUNet1(in_nc=4*wf, out_nc=4*wf, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=64)
        
        self.st_2 = DnCNN(in_nc=4*wf, out_nc=4*wf, nc=64, nb=20, act_mode='R')
        
        self.st_3 = SCUNet(in_nc=16*wf, out_nc=16*wf, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=32)
        #self.st_3 = DnCNN(in_nc=16*wf, out_nc=16*wf, nc=64, nb=20, act_mode='R')
        
        #self.st_4 = SCUNet1(in_nc=64, out_nc=64, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128)
        self.st_4 = DnCNN(in_nc=64, out_nc=64, nc=64, nb=20, act_mode='R')

        self.last = conv3x3(wf, in_chn, bias=True)
        
        
        #DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')

    def forward(self, x1):
        o1 = self.l1(x1) # 16,512,512
        #print(o1.shape)
        d1 = dwt_init(o1) #64,256,256
        #print(d1.shape)
        #d1 = self.st_4(d1)
        o2 = self.l2(d1) #64,256,256
        #print(o2.shape)
        d2 = self.down2(o2) #256,128,128
        #print(d2.shape)
        o3 = self.l3(d2) #256,128,128
        #print(o3.shape)
        d3 = self.down3(o3) #1024,64,64
        #print(d3.shape)
        o4 = self.l4(d3) #1024,64,64
        #print(o4.shape)
        #o4 = self.st_4(o4)
        #print(o4.shape)
        u3 = iwt_init(o4) #256,128,128
        u3 = u3.to('cuda')
        #u3 = torch.cat([u3, self.st_3(o3)], dim=1) #512,128,128
        
        u3 = torch.cat([u3, o3], dim=1) 
        u3 = self.u3(u3) #256,128,128

        u2 = self.up2(u3) #64,256,256
        u2 = u2.to('cuda')
        u2 = torch.cat([u2, self.st_2(o2)], dim=1) #128,256,256
        u2 = self.u2(u2) #64,256,256

        u1 = self.up1(u2) #16,512,512
        u1 = u1.to('cuda')
        u1 = torch.cat([u1, self.st_1(o1)], dim=1) 
        u1 = self.u1(u1) #16,512,512

        out = self.last(u1) #3,512,512
        return out

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':

    # torch.cuda.empty_cache()
    net = DMergenet()

    x = torch.randn((2, 3, 512, 512))
    x = net(x)
    print(x.shape)
