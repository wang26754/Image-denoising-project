import torch
from torch.utils.tensorboard import SummaryWriter
import pathlib
from tqdm.autonotebook import tqdm
import time
import os
from matplotlib import pyplot as plt
from torch import autograd
#from utils.support import *
from scipy.io import savemat
import torch.nn as nn
from torch.autograd import Variable
####Add by wenzhu####
import logging
from utils import utils_logger

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def psnr(pred, target, max_pixel=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def SAM(pred, target):
    # Ensure input is (N,C,H,W)
    assert pred.shape == target.shape, "Predicted and target images must have the same shape"
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1)  # Reshape to (N,C,H*W)
    target = target.reshape(target.shape[0], target.shape[1], -1)
    
    # Normalize vectors to unit vectors
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    
    # Compute the dot product
    dot_product = (pred_norm * target_norm).sum(1)
    
    # Compute the angle in radians and then convert to degrees
    angles = torch.acos(dot_product.clamp(-1, 1))  # Clamp for numerical stability
    return torch.mean(angles) * (180.0 / np.pi)

def SSIM_local(pred, target):
    pred = pred.permute(0, 2, 3, 1).cpu().numpy()  # Convert to numpy array in NHWC format
    target = target.permute(0, 2, 3, 1).cpu().numpy()
    
    batch_size = pred.shape[0]
    ssim_total = 0
    
    for i in range(batch_size):
        ssim_total += ssim(pred[i], target[i], multichannel=True, data_range=pred[i].max() - pred[i].min())
    
    return ssim_total / batch_size


''' training setup '''
def train(config, logs_root_path, dataloader, validloader, decoder, optim_decoder):
    
    events_dir = logs_root_path+'/events/'
    image_dir = logs_root_path+'/images/'
    checkpoints_dir = logs_root_path+'/checkpoints/'
    valid_dir = logs_root_path+'/valid/'
    logloss = config.log_loss_every
    logdisp = config.log_display_every
    pathlib.Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(valid_dir).mkdir(parents=True, exist_ok=True)
    l1loss = nn.L1Loss()

    print("Create summaries")
    writer = SummaryWriter(events_dir)
    ####Add by wenzhu####
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(events_dir, logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(utils_logger.dict2str(vars(config)))

    print("Start training")
    loss_avg = 0.0
#    noisy_loss_avg = 0.0
     
    total_steps = 0
    sam = 0.0
    sim = 0.0
    ####Add by wenzhu####
    current_step = 0
    
    with tqdm(total=len(dataloader) * config.num_epochs) as pbar:
        for epoch in range(config.num_epochs):
                        
            for i, (data, noisy) in enumerate(dataloader,0):

                if config.use_gpu:
                    data = data.to(config.device)
                    noisy = noisy.to(config.device)
                    
                for param in decoder.parameters():
                    param.requires_grad = True
                
                ''' init optimizer '''
                optim_decoder.zero_grad()
                
                ''' forward model (encoder) and inverse model (decoder) '''
               
                x_pred = decoder(noisy) 
                
                #if i%500==0:
                   # plt.imsave(image_dir+str(int(i)+1)+'.png',torch.clamp(x_pred[0,:,:,:].permute(1,2,0),0,1).cpu().detach().numpy())

                ''' supervision '''
                loss = l1loss(x_pred, data)
                
                loss.backward()

                ''' updates of optimizer '''
                optim_decoder.step()
                #pbar.update(1)

                loss_avg += psnr(x_pred, data).cpu().detach()

                total_steps += 1
                sam += SAM(data.cpu().detach(),x_pred.cpu().detach())
                sim += SSIM_local(data.cpu().detach(),x_pred.cpu().detach())
                
                current_step += 1
                #if current_step%(len(dataloader)*logdisp)==0:
                if (epoch + 1) ==30:
                    torch.save(decoder.state_dict(),
                               checkpoints_dir+'model_decoder_'+str(epoch+1))
                    ####Add by wenzhu####
                    logger.info('Saving the model.')
                #if current_step%(len(dataloader)*logdisp)==0:
                if (epoch + 1) ==30:
                  ######Add by wenzhu####
                  ######Validation#####
                     valid_psnr = 0.0
                     valid_num = 0
                     for i, (data, noisy) in enumerate(validloader,0):
                         if config.use_gpu:
                             data = data.to(config.device)
                             noisy = noisy.to(config.device)
                          
                         x_pred = decoder(noisy)
                         plt.imsave(valid_dir+'epoch'+str(epoch+1)+'_'+str(int(i)+1)+'.png',torch.clamp(x_pred[0,:,:,:].permute(1,2,0),0,1).cpu().detach().numpy())
                         valid_psnr += psnr(x_pred, data).cpu().detach()
                         valid_num += 1
                     logger.info('<Validation: epoch:{:3d}, Average PSNR : {:<.2f}dB\n'.format((epoch+1), valid_psnr/valid_num))
                if current_step%(len(dataloader)//logloss)==0:
                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.4f}\n'.format((epoch+1), current_step, loss_avg/total_steps, sim/total_steps))
#                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average noisy PSNR : {:<.2f}dB, Average SSIM : {:<.4f}\n'.format((epoch+1), current_step, loss_avg/total_steps, noisy_loss_avg/total_steps, sim/total_steps))
                
                

            pbar.update(len(dataloader))
            print(f"[{epoch}]"," PSNR=", loss_avg/total_steps,"dB", "SAM=",sam/total_steps,"SSIM=",sim/total_steps)
            
            #sched_coff.step()
            
            writer.add_scalar("loss", loss_avg/total_steps, epoch)
            writer.add_scalar("SAM", sam/total_steps, epoch)
            writer.add_scalar("SSIM", sim/total_steps, epoch)

            loss_avg = 0.0
            total_steps = 0
            sam = 0.0
            sim = 0.0
    ####Add by wenzhu####                 
    logger.info('End of training.')
