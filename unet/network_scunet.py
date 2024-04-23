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
class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]
    def flops(self, H, W):
        flops = 0
        # norm1
        flops += self.input_dim * H * W *3*self.input_dim
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        N = self.window_size * self.window_size
        flops += nW * self.n_heads * N * self.head_dim * N + self.n_heads * N * N * self.head_dim + N * self.input_dim * self.input_dim
        
        # norm2
        flops += self.input_dim * H * W*self.output_dim
        return flops


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x
    def flops(self, H, W):
        flops = 0
        flops += H*W*self.input_dim*2
        flops += self.msa.flops(H, W)
        flops += H*W*self.input_dim*self.input_dim*4+H*W*self.input_dim*4*self.output_dim
        return flops


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
                )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x
    def flops(self, H, W):
        flops = 0
        flops += H*W*self.conv_dim*self.conv_dim*2*2*2
        flops += self.trans_block.flops(H, W)
        flops += H*W*self.conv_dim*self.conv_dim*3*3*2
        return flops
    
class HPF(nn.Module):

    def __init__(self, in_channels=3, kernel_size=3, stride=1, group=3):
        super(HPF, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, 
                              kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x0):
        sigma = self.conv(x0)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x0.shape
        x = F.unfold(x0, kernel_size=self.kernel_size,padding=1).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x0-x
    
class HFGN(nn.Module):
    def __init__(self):
        super(HFGN, self).__init__()
        self.hpf = HPF()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, sigma):
#        batch_size = x.size(0)
        hf = self.hpf(x)#.view(batch_size,3,-1)
#        hf_t = hf.permute(0,2,1)
        hfa = self.sigmoid(hf)
        return x+sigma*hfa #torch.cat((x,sigma*hfa),dim=1)


        
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
class SCUNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        # self.ROI1_head = [nn.Conv2d(in_nc, dim//8, 3, 1, 0, bias=False)] + \
        #                 [PixelUnShuffle(2)]
        # self.ROI2_head = [PixelUnShuffle(2)] + \
        #                 [nn.UpsamplingBilinear2d((512,512))] + \
        #                 [nn.Conv2d(in_nc*4, dim//2, 3, 1, 1, bias=False)]
        # self.ROI3_head = [nn.UpsamplingBilinear2d((512,512))] + \
        #                 [nn.Conv2d(in_nc, dim//2, 3, 1, 1, bias=False)]
        # self.merge = [nn.Conv2d(dim//2, in_nc, 3, 1, 1, bias=False)]
#        self.hfgn = HFGN()
        self.m_head = [nn.Conv2d(in_nc, dim//4, 3, 1, 1, bias=False)]#1024x1024
        ###Added by wenzhu
        ### For reduce memory occupy
        self.m_down0 = PixelUnShuffle(2)

        begin = 0
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[4])]
                      
        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[5])]
                      
        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[6])]

        self.m_up0 = PatchUnEmbedding(2)
        # self.m_tail0 = [nn.Conv2d(dim//4, dim//2, 3, 1, 1, bias=False)]
        self.m_tail = [nn.Conv2d(dim//4, out_nc, 3, 1, 1, bias=False)]
#        self.m_tail = [nn.Conv2d(out_nc, dim, 3, 1, 1, bias=False)] + \
#                        [nn.Conv2d(dim, dim, 3, 1, 1, bias=False)] + \
#                        [nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False)]

        # self.ROI1_head = nn.Sequential(*self.ROI1_head)
        # self.ROI2_head = nn.Sequential(*self.ROI2_head)
        # self.ROI3_head = nn.Sequential(*self.ROI3_head)
#        self.hfgn = nn.Sequential(*self.hfgn)
        self.m_head = nn.Sequential(*self.m_head)
        # self.m_down0 = nn.Sequential(*self.m_down0)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        # self.m_tail0 = nn.Sequential(*self.m_tail0)  
#        self.m_blur_tail = nn.Sequential(*self.m_blur_tail)  
        self.m_tail = nn.Sequential(*self.m_tail)  
        #self.apply(self._init_weights)

    def forward(self, x0):

        # x1 = self.pre_process(x0, pad)
#        paddingBottom = int(np.ceil(h/64)*64-h)
#        paddingRight = int(np.ceil(w/64)*64-w)
#        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        
        # x00 = self.merge(x1)
#        x0 = self.hfgn(x0[:,:3,:,:],x0[0,3:,:,:])
        #print(x0.shape)
        x1 = self.m_head(x0)
        x11 = self.m_down0(x1)
        x2 = self.m_down1(x11)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_up0(x)
        # x = self.m_tail0(x)
        x = self.m_tail(x+x1)
#        x = self.hfgn(x,x0[0,3:,:,:])
#        out = self.m_tail(x)+x

        # x = x[..., :h-pad, :w-pad]
        
        return x#, out




if __name__ == '__main__':

    # torch.cuda.empty_cache()
    net = SCUNet()

    x = torch.randn((2, 3, 512, 512))
    x = net(x)
    print(x.shape)
