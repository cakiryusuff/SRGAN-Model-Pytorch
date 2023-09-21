# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:33:54 2023

@author: cakir
"""
import torch.nn as nn
import math

class residualBlock(nn.Module):
  def __init__(self, planes):
    super(residualBlock, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(planes, planes, 3, 1, 1, bias = False),
        nn.BatchNorm2d(planes),
        nn.PReLU(),
        nn.Conv2d(planes, planes, 3, 1, 1, bias = False),
        nn.BatchNorm2d(planes),
    )

  def forward(self, x):
    out = self.layers(x)
    sum = out + x
    return sum

class pixelShuf(nn.Module):
  def __init__(self, planes: int, up_scale: int = 2):
    super(pixelShuf, self).__init__()
    self.layer = nn.Sequential(
        nn.Conv2d(planes, planes * up_scale ** 2, 3, 1, 1),
        nn.PixelShuffle(2),
        nn.PReLU()
    )
  def forward(self, x):
    return self.layer(x)

class Generator(nn.Module):
  def __init__(self, scale_factor: int = 4, num_of_blocks:int = 5):
    super(Generator, self).__init__()
    planes: int = 64
    num_of_pixelShuf: int = int(math.log(scale_factor, 2))
    
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, planes, 9, 1, 4),
        nn.PReLU())

    layer = []
    for i in range(num_of_blocks):
      layer.append(residualBlock(planes))
      
    self.layers = nn.Sequential(*layer)
        
    self.conv2 = nn.Sequential(
        nn.Conv2d(planes, planes, 3, 1, 1, bias = False),
        nn.BatchNorm2d(planes))
    
    pixelshufList = []
    for i in range(num_of_pixelShuf):
        pixelshufList.append(pixelShuf(planes))
    self.pixelshufList = nn.Sequential(*pixelshufList)

    self.lastConv = nn.Conv2d(planes, 3, 9, 1, 4)

  def forward(self, x):
    first_out = self.conv1(x)
    out = self.layers(first_out)
    out = self.conv2(out)
    out = first_out + out
    out = self.pixelshufList(out)
    out = self.lastConv(out)

    return out