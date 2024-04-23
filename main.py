#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:13:18 2024

@author: wenzhu.xing@tuni.fi
"""

''' packages '''
from optics import *
from data import *
from data_wt import DWTImageSet
#from utils.support import *
from utils.training import *
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from unet.network_scunet import SCUNet
from unet.network_unet import UNetRes #drunet
from unet.network_swinir import SwinIR
from unet.network_dncnn import DnCNN
from unet.network_DMergenet import DMergenet
import torch as th
from data import ImageSet
from unet.network_LargeDwtnet import LargeDwtnet
''' Experiment configuration '''
print('Setting learning framework...')
parser = argparse.ArgumentParser(description='image denoising network abalation')
parser.add_argument('--net', type=str, required= False, default='scunet', dest='net')
parser.add_argument('--mode', type=str, required= False, default='normal', dest='mode') #'dwtLL' 'dwtLH' 'dwtHL' 'dwtHH'
parser.add_argument('--sigma', type=float, required= False, default=10., dest='sigma')
parser.add_argument('--batch_size', type=int, required=False, default=2, dest='batch_size')
parser.add_argument('--lr', type=float,required=False, default=1e-4, dest='learning_rate')
parser.add_argument('--epochs', type=int, required=False, default=30, dest='num_epochs')
parser.add_argument('--logloss', type=int, required=False, default=1, dest='log_loss_every')
parser.add_argument('--logdisp', type=int, required=False, default=10, dest='log_display_every')
parser.add_argument('--dataset_path', type=str, required=False, dest='data_path',
                    default='images/train')
parser.add_argument('--valid_path', type=str, required=False, dest='valid_path',
                    default='images/valid')
parser.add_argument('--log_path', type=str, required=False, dest='log_path',
                    default='logs')
parser.add_argument('--gpu', required=False, action='store_true', default=True, dest='use_gpu')
parser.add_argument('--device', type=str, required=False, default='cuda:0')
config = parser.parse_args()
print('done...')


''' Variables '''
print('Creating variables...')
logs_root_path = config.log_path+config.net+'_'+config.mode+'_'+str(config.sigma)[:-1]
print('done...')
    
if config.mode[:3] == 'dwt':
    print('Creating database...')
    dataset = DWTImageSet(img_path=config.data_path,sigma=config.sigma,dim=config.mode[3:])
    dataloader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True, num_workers=8)
    print('done...')
    ######Add by wenzhu####
    ######Validation#####
    print('Creating valid data...')
    validset = DWTImageSet(img_path=config.valid_path,sigma=config.sigma,dim=config.mode[3:])
    validloader = DataLoader(validset,batch_size=1,shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    print('done...')
    psize = 512//2
else:
    print('Creating database...')
    dataset = ImageSet(img_path=config.data_path,sigma=config.sigma)
    dataloader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True, num_workers=8)
    print('done...')
    ######Add by wenzhu####
    ######Validation#####
    print('Creating valid data...')
    validset = ImageSet(img_path=config.valid_path,sigma=config.sigma)
    validloader = DataLoader(validset,batch_size=1,shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    print('done...')
    psize = 512
    
inc = 3
outc = 3
''' Decoder: UNet '''
print('Instantiating decoder...')
if config.net == 'scunet':
    decoder = SCUNet(in_nc=inc, out_nc=outc, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=psize)
if config.net == 'dmergenet':
    decoder = DMergenet()
if config.net == 'largedwtnet':
    decoder = LargeDwtnet()
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
#decoder.load_state_dict(checkpoint)
decoder.train()
decoder = decoder.to(config.device)
print('done...')

''' Loss function and optimizers'''
print('Initializing optimizers...')
optimizer_decoder = optim.Adam(decoder.parameters(),lr=1e-4)
print('done...')

''' launch training '''
print('Starting training...')
train(config, logs_root_path, dataloader=dataloader, validloader=validloader, decoder=decoder, optim_decoder=optimizer_decoder)
print('done...')




