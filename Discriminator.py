# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:36:37 2023

@author: cakir
"""
import torch.nn as nn
import torch

class leakyBlock(nn.Module):
    def __init__(self, features: list):
        super(leakyBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, features[2], 1),
            nn.BatchNorm2d(features[1]),
            nn.LeakyReLU()
            )
    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        planes: int = 64
        plane_list: list = [[planes, planes, 2],
                            [planes, planes*2, 1],
                            [planes*2, planes*2, 2],
                            [planes*2, planes*4, 1],
                            [planes*4, planes*4, 2],
                            [planes*4, planes*8, 1],
                            [planes*8, planes*8, 2]]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, 1, 1),
            nn.LeakyReLU())
        
        layer = []
        for i in range(len(plane_list)):
            layer.append(leakyBlock(plane_list[i]))
            
        self.layers = nn.Sequential(*layer)
           
        self.fc = nn.Sequential(    
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1, kernel_size=1)
            )
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv1(x)
        out = self.layers(out)
        out = self.fc(out).view(batch_size)
        
        return torch.sigmoid(out)