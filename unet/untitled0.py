#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:59:24 2024

@author: anchor2015
"""

class SCUNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        # Initialize heads without PixelUnShuffle
        self.m_head = nn.Sequential(
            nn.Conv2d(in_nc, dim//4, 3, 1, 1, bias=False)  # Adjust the channel depth directly
        )

        # Define the model architecture using the same concept without downsampling via PixelUnShuffle
        self.init_layers(config, dpr, input_resolution)

    def init_layers(self, config, dpr, input_resolution):
        layers = {
            'down': [],
            'up': [],
            'body': []
        }
        resolution = input_resolution
        num_features = self.dim // 4

        # Construct downsampling, body, and upsampling layers
        for i, num_blocks in enumerate(config):
            layer_type = 'body' if i == 3 else ('up' if i > 3 else 'down')
            for _ in range(num_blocks):
                layers[layer_type].append(ConvTransBlock(num_features, num_features, self.head_dim, self.window_size, dpr.pop(0), 'W' if _ % 2 == 0 else 'SW', resolution))
                if layer_type != 'body':
                    num_features *= 2 if layer_type == 'down' else num_features // 2
                    layers[layer_type].append(nn.Conv2d(num_features if layer_type == 'down' else num_features * 2, num_features, 2, 2, 0, bias=False))
                    resolution //= 2

        # Convert lists to nn.Sequential
        self.m_down1 = nn.Sequential(*layers['down'][:len(config[0])])
        self.m_down2 = nn.Sequential(*layers['down'][len(config[0]):len(config[0])+len(config[1])])
        self.m_down3 = nn.Sequential(*layers['down'][len(config[0])+len(config[1]):])
        self.m_body = nn.Sequential(*layers['body'])
        self.m_up3 = nn.Sequential(*layers['up'][:len(config[4])])
        self.m_up2 = nn.Sequential(*layers['up'][len(config[4]):len(config[4])+len(config[5])])
        self.m_up1 = nn.Sequential(*layers['up'][len(config[4])+len(config[5]):])
        self.m_tail = nn.Sequential(
            nn.Conv2d(num_features, out_nc, 3, 1, 1, bias=False)
        )

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x
