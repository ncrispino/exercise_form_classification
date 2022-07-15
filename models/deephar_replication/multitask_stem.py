# See Figure 10 and 11 for more.
# For reference, based on Inception-V4 (https://arxiv.org/pdf/1602.07261.pdf) as stated in the article.

import torch
import torch.nn as nn

class SRBlock(nn.Module):
    """
    Separable residual module.
    Uses depthwise separable convolutions, meaning it combines a depthwise convolution with a pointwise convolution.
    Not using Relu between depthwise and pointwise, as recommended in this paper: https://arxiv.org/abs/1610.02357.
    """
    def __init__(self, n_in, n_out, s, include_batch_relu=True):
        self.n_in = n_in
        self.n_out = n_out     
        self.include_batch_relu = include_batch_relu   
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=1)
        # use n_in separate kernels of dim 1 x kernel_size x kernel_size, each for one channel, and concatenate together using pointwise conv
        self.depthwise_conv = nn.Conv2d(n_in, n_in, kernel_size=s, groups=n_in, padding='same') # as input and output both have W x H, as seen in Figure 10 of paper
        # use a 1x1 conv to increase output dim
        self.pointwise_conv = nn.Conv2d(n_in, n_out, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(n_out)
        self.relu = nn.Relu()
    
    def forward(self, x):
        out1 = self.conv(x)
        if self.n_in == self.n_out:
            out = x + out1
            if self.include_batch_relu:
                out = self.batch_norm(out)
                out = self.relu(out1)
            return out
        out2 = self.depthwise_conv(x)
        out2 = self.pointwise_conv(out2)
        out = out1 + out2
        if self.include_batch_relu:
            out = self.batch_norm(out)
            out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    """
    Convolution with batch normalization and relu.
    """
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLu()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))

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
            ConvBlock(192, 64, kernel_size=1), 
            ConvBlock(64, 96, kernel_size=3), padding='same')
        self.right = nn.Sequential(
            ConvBlock(192, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(5, 1), padding='same'),
            ConvBlock(64, 64, kernel_size=(1, 5), padding='same'),
            ConvBlock(64, 96, kernel_size=(3, 3), padding='same')
        )
    
    def forward(self, x):
        return torch.cat((self.left(x), self.right(x)), dim=1)

class MultitaskStem(nn.Module):
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
        self.inception1 = Inception1(64, 96, ckernel=3, cstride=2, pkernel=3, pstride=2, cpadding=1, ppadding=1) #192x64x64
        self.inception2 = Inception2() #(96x64x64, 96x64x64) = 192x64x64
        self.inception3 = Inception1(192, 192, ckernel=3, cstride=2, pkernel=2, pstride=2, cpadding=1, ppadding=0) #(192x32x32, 192x32x32) = 384x32x32  
        self.sr = SRBlock(384, 576, 3) #576x32x32
    
    def forward(self, x):
        x = self.initial_convs(x)

    # To-do:
        # Make sure dim is correct for torch.cat
        # See if I need to do batch normalization and relu after depthwise separable convs