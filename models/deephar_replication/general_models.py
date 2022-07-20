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
        self.depthwise_conv = nn.Conv2d(n_in, n_in, kernel_size=s, groups=n_in, padding='same') # as input and output both have H x W, as seen in Figure 10 of paper
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

class SoftArgMax(nn.Module):
    """
    NOT FINISHED
    Soft-argmax operation. Returns C x 2 if 2D, C x 1 if 1D.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim == 2:
            return torch.randint(0, 1, size=(20, 17, 2), dtype=torch.float)
        if self.dim == 1:
            return torch.randint(0, 1, size=(20, 17, 1), dtype=torch.float)

class MaxPlusMinPooling(nn.Module):
    """
    MaxPlusMin pooling implemented according to max_min_pooling in deephar/layers.py from the paper's code.
    """
    def __init__(self, kernel_size, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding=padding)
        self.minpool = nn.MaxPool2d(kernel_size, stride, padding=padding)

    def __call__(self, x):
        return self.maxpool(x) - self.minpool(-x)

class GlobalMaxPlusMinPooling(nn.Module):
    """
    GlobalMaxPlusMin pooling implemented according to max_min_pooling in deephar/layers.py from the paper's code. This is 2D pooling.
    Global max pooling takes the max value for each channel. For more, see https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/global-max-pooling-2d.
    Input: C x H x W
    Output: C
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_pool = torch.amax(x, dim=(2, 3)) 
        min_pool = torch.amax(-x, dim=(2, 3))    
        return max_pool - min_pool

def kronecker_prod(a, b):
    """
    Multiplies a and b by channel. Returns B x C1 x C2 x H x W tensor
    a -- B * T x C1 x H x W tensor
    b -- B * T x C2 x H x W tensor
    """
    C1 = a.shape[-3]
    C2 = b.shape[-3]
    a = a.unsqueeze(-3)
    b = b.unsqueeze(-4)
    a = a.tile(1, 1, C2, 1, 1)
    b = b.tile(1, C1, 1, 1, 1)
    return a * b