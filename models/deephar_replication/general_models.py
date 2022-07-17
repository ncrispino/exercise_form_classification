"""
Holds models shared by different parts of the network.
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Convolution with batch normalization and relu.
    """
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))

class SCBlock(nn.Module):
    """
    Depth-wise separable convolution.
    include_batch_relu is a boolean that represents if a batch norm and relu are used after the convolution.
    """
    def __init__(self, n_in, n_out, s, include_batch_relu=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out     
        self.include_batch_relu = include_batch_relu 
        # use n_in separate kernels of dim 1 x kernel_size x kernel_size, each for one channel, and concatenate together using pointwise conv
        self.depthwise_conv = nn.Conv2d(n_in, n_in, kernel_size=s, groups=n_in, padding='same') # as input and output both have W x H, as seen in Figure 10 of paper
        # use a 1x1 conv to increase output dim
        self.pointwise_conv = nn.Conv2d(n_in, n_out, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        if self.include_batch_relu:
            out = self.relu(self.batch_norm(out))
        return out
        

class SRBlock(nn.Module):
    """
    Separable residual module.
    Uses depthwise separable convolutions, meaning it combines a depthwise convolution with a pointwise convolution.
    Not using Relu between depthwise and pointwise, as recommended in this paper: https://arxiv.org/abs/1610.02357.
    include_batch_relu is a boolean that represents if a batch norm and relu are used before the end of the block.
    """
    def __init__(self, n_in, n_out, s, include_batch_relu=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out     
        self.include_batch_relu = include_batch_relu   
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=1)
        self.sc = SCBlock(n_in, n_out, s, include_batch_relu=True)
        self.batch_norm = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out1 = self.conv(x)
        if self.n_in == self.n_out:
            out = x + out1
            if self.include_batch_relu:
                out = self.batch_norm(out)
                out = self.relu(out1)
            return out
        out2 = self.sc(x)
        out = out1 + out2
        if self.include_batch_relu:
            out = self.batch_norm(out)
            out = self.relu(out)
        return out

class SoftArgMax():
    """
    Soft-argmax operation.
    """
    def __call__(self, x):
        pass