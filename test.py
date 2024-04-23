#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:51:27 2024

@author: xingw
"""

import cv2
import numpy as np
from unet.network_scunet import SCUNet
from unet.network_unet import UNetRes
from unet.network_swinir import SwinIR
from unet.network_dncnn import DnCNN
import torch as th
import glob
import os
from matplotlib import pyplot as plt
import pathlib
import argparse
from utils.utils_test import modcrop


parser = argparse.ArgumentParser(description='Test denoising')
parser.add_argument('--net', type=str, required= False, default='scunet', dest='net')
parser.add_argument('--mode', type=str, required= False, default='normal', dest='mode')
parser.add_argument('--sigma', type=float, required= False, default=10., dest='sigma')
parser.add_argument('--log_path', type=str, required=False, dest='log_path',
                    default='logs')
parser.add_argument('--data_dir', type=str, required=False, dest='data_dir',
                    default='images/test')
parser.add_argument('--result_dir', type=str, required=False, dest='result_dir',
                    default='results/')
config = parser.parse_args()

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
#device = th.device('cpu')
psize = 512
inc = 3
outc = 3
logs_root_path = config.log_path+config.net+'_'+config.mode+'_'+str(config.sigma)[:-1]+'/checkpoints/model_decoder_30'
checkpoint = th.load(logs_root_path, map_location="cuda:0")
if config.net == 'scunet':
    decoder = SCUNet(in_nc=inc, out_nc=outc, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=psize//2)
elif config.net == 'drunet':
    decoder = UNetRes(in_nc=inc, out_nc=outc, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                      downsample_mode="strideconv", upsample_mode="convtranspose")
elif config.net == 'dncnn':
    decoder = DnCNN(in_nc=inc, out_nc=outc, nc=64, nb=20, act_mode='R')
elif config.net == 'swin':
    decoder = SwinIR(upscale=1, patch_size=2, in_chans=inc, img_size=psize//2, window_size=8,
                     img_range=1., depths=[4,4], embed_dim=36, 
                     num_heads=[6,6],
                     mlp_ratio=2, upsampler='', resi_connection='1conv')
decoder.load_state_dict(checkpoint)
decoder = decoder.to(device)

result_dir = config.result_dir+config.net+'_'+config.mode+'_'+str(config.sigma)[:-1]+'/'
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
data_dir = config.data_dir
print(data_dir)

data_files = sorted(glob.glob(os.path.join(data_dir,'*.png')))
for i in range(len(data_files)):
    m = data_files[i].split("/")
    print(m[-1], ',')
    data = cv2.imread(data_files[i], cv2.IMREAD_UNCHANGED)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = modcrop(data, 32)
    noise = np.random.normal(0, config.sigma/255.0, data.shape)
    noisy = np.float32(data/255.) + noise.astype(np.float32)
    
    noisy = th.from_numpy(noisy).type(th.FloatTensor)
    noisy = noisy.to(device)
    
    recons = th.squeeze(decoder(th.unsqueeze(noisy.permute(2,0,1),dim=0))).permute(1,2,0).cpu().detach().numpy()
    plt.imsave(result_dir+ m[-1],recons.clip(0.0,1.0))
