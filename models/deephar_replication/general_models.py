"""Holds models shared by different parts of the network."""

import torch
from torch import nn

class ConvBlock(nn.Module):
    """Convolution with batch normalization followed by an relu."""

    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size=kernel_size, stride=stride, 
            padding=padding)
        self.batch_norm = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))

class SCBlock(nn.Module):
    """Depth-wise separable convolution.
    
    Attributes:
        n_in: number of input channels.
        n_out: number of output channels.
        s: size of convolutional kernel in depthwise convolution.
        include_batch_relu: whether to include batch normalization and 
            relu after the convolutions.

    """

    def __init__(self, n_in, n_out, s, include_batch_relu=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out     
        self.include_batch_relu = include_batch_relu 
        # Use n_in separate kernels of dim 1 x kernel_size x kernel_size, 
        # each for one channel.        
        self.depthwise_conv = nn.Conv2d(
            n_in, n_in, kernel_size=s, groups=n_in, padding='same')
        # Use a 1x1 conv (pointwise conv) to increase output dim
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
    """Separable residual module as in Figure 10 of paper.

    Uses depthwise separable convolutions, meaning it combines a 
    depthwise convolution with a pointwise convolution.
    Not using Relu between depthwise and pointwise, 
    as recommended in this paper: https://arxiv.org/abs/1610.02357.

    Attributes:
        n_in: number of input channels.
        n_out: number of output channels.
        s: size of convolutional kernel in depthwise convolution.
        include_batch_relu: whether to include batch normalization and 
            relu after the convolutions.
.
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

def spacial_softmax(x):
    """Apply softmax to each H x W.

    Args:
        x: B x C x H x W tensor..
    
    Returns:
        B x C x H x W tensor..

    """

    H = x.shape[2]
    W = x.shape[3]
    softmax = nn.Softmax(2)
    # Collapse height and width to apply softmax.
    x_collapsed = x.view(x.shape[0], x.shape[1], -1)
    x_prob = softmax(x_collapsed).view(-1, x.shape[1], H, W)  
    return x_prob    

class SoftArgMax(nn.Module):
    """Soft-argmax operation.
    
    Based off Tensorflow code from the paper.

    Args:
        x: B x C x H x W tensor.
        apply_softmax: whether to apply spacial softmax.  

    Returns:
        B x C x 2 tensor if 2D, B x C x 1 tensor if 1D.  

    """

    def forward(self, x, apply_softmax=True):
        super().__init__()
        dim1 = False
        if len(x.shape) == 3: # adds dimension to end if 1D
            dim1 = True
            x = x.unsqueeze(-1)            
        H = x.shape[2]
        W = x.shape[3]        
        if apply_softmax:
            x_prob = spacial_softmax(x)
        else:
            x_prob = x
        # Create tensor with weights to multiply.
        height_values = torch.arange(0, H).unsqueeze(1)
        width_values = torch.arange(0, W).unsqueeze(0)
        # Each row has row idx/num rows.
        height_tensor = torch.tile(height_values, (1, W))/(H - 1)
        # Each col has col idx/num cols.
        width_tensor = torch.tile(width_values, (H, 1))/(W - 1)

        # Multiply prob maps times weight tensors and sum over H x W.        
        height_out = (x_prob * height_tensor).sum((2, 3))
        width_out = (x_prob * width_tensor).sum((2, 3))
        # Returns (x, y), which corresponds to (W, H).
        out = torch.cat(
            (width_out.unsqueeze(-1), height_out.unsqueeze(-1)), 
            dim=-1)        
        if dim1:            
            # Return only height_out, as width was only dim 1.
            return out[:, :, 1].unsqueeze(-1)
        return out

class MaxPlusMinPooling(nn.Module):
    def __init__(self, kernel_size, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.maxpool1 = nn.MaxPool2d(kernel_size, stride, padding=padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size, stride, padding=padding)

    def forward(self, x):
        return self.maxpool1(x) - self.maxpool2(-x)

class GlobalMaxPlusMinPooling(nn.Module):
    """ Takes the max value for each channel.
    
    This is 2D pooling. For more, see https://shorturl.at/gnstX.
    """

    def forward(self, x):
        """
        Args:                
            B x C x H x W tensor.

        Returns:
            B x C tensor.    
        """
        max_pool = torch.amax(x, dim=(2, 3)) 
        min_pool = torch.amax(-x, dim=(2, 3))    
        return max_pool - min_pool

def kronecker_prod(a, b):
    """Multiplies a and b by channel. 

    Args:
        a: B x C1 x H x W tensor.
        b: B x C2 x H x W tensor.
    
    Returns:
        B x C1 x C2 x H x W tensor.

    """

    C1 = a.shape[-3]
    C2 = b.shape[-3]
    a = a.unsqueeze(-3)
    b = b.unsqueeze(-4)
    a = a.tile(1, 1, C2, 1, 1)
    b = b.tile(1, C1, 1, 1, 1)
    return a * b