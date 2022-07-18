"""
Contains the multitask stem model as seen in Figures 10 and 11.
For reference, based on Inception-V4 (https://arxiv.org/pdf/1602.07261.pdf) as stated in the article.
"""

import torch
import torch.nn as nn
from general_models import ConvBlock
from general_models import SRBlock

class Inception1(nn.Module):
    """
    Inception module with a parallel convolution and max pool.
    """
    def __init__(self, input_dim, output_dim, ckernel_size, cstride, pkernel_size, pstride, cpadding, ppadding):
        super().__init__()
        self.conv = ConvBlock(input_dim, output_dim, kernel_size=ckernel_size, stride=cstride, padding=cpadding)
        self.pool = nn.MaxPool2d(kernel_size=pkernel_size, stride=pstride, padding=ppadding)
    
    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.pool(x)        
        return torch.cat((out1, out2), dim=1)

class Inception2(nn.Module):
    """
    Only one in model, so not including parameters.
    """
    def __init__(self):
        super().__init__()
        self.left = nn.Sequential(
            ConvBlock(160, 64, kernel_size=1), 
            ConvBlock(64, 96, kernel_size=3, padding='same'))
        self.right = nn.Sequential(
            ConvBlock(160, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(5, 1), padding='same'),
            ConvBlock(64, 64, kernel_size=(1, 5), padding='same'),
            ConvBlock(64, 96, kernel_size=(3, 3), padding='same')
        )
    
    def forward(self, x):
        return torch.cat((self.left(x), self.right(x)), dim=1)

class EntryFlow(nn.Module):
    """
    Combine according to network architecture.
    Input assumed to be 3x256x256.
    Note that in the Tensorflow Implementation, padding='same' in the conv layers, meaning the output_dim = ceil(input_dim/stride)
        See https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave for more.
        Since PyTorch doesn't accept 'same' for a stride > 1, I instead added padding s.t. the arithmatic worked out.
    """
    def __init__(self):
        super().__init__()
        self.initial_convs = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1), # 32x128x128
            ConvBlock(32, 32, kernel_size=3, padding='same'), # 32x128x128
            ConvBlock(32, 64, kernel_size=3, padding='same') # 64x128x128
        )
        self.inception1 = Inception1(64, 96, ckernel_size=3, cstride=2, pkernel_size=3, pstride=2, cpadding=1, ppadding=1) #160x64x64
        self.inception2 = Inception2() #(96x64x64, 96x64x64) = 192x64x64
        self.inception3 = Inception1(192, 192, ckernel_size=3, cstride=2, pkernel_size=2, pstride=2, cpadding=1, ppadding=0) #(192x32x32, 192x32x32) = 384x32x32  
        self.sr = SRBlock(384, 576, 3) #576x32x32
    
    def forward(self, x):
        out = self.initial_convs(x)        
        out = self.inception1(out)        
        out = self.inception2(out)
        out = self.inception3(out)
        out = self.sr(out)
        return out

    # To-do:
        # See if I need to do batch normalization and relu after depthwise separable convs