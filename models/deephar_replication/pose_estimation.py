"""
Contains prediction block for pose estimation, as seen in Figure 12.
Will split into two blocks.
For upsampling, the authors' code use UpSampling2D with params (2, 2). 
    This should be identical to PyTorch's Upsample using a scale factor of 2 and mode nearest.
"""
import torch
import torch.nn as nn
from general_models import *
        
class PoseDownBlock(nn.Module):
    """
    Bottom part of pose block.
    Input assumed to be 576x32x32.
    """
    def __init__(self):
        super().__init__()
        self.left = SRBlock(576, 576, 5)
        self.middle_left = SRBlock(576, 288, 5)
        self.middle_right = [
            nn.MaxPool2d(kernel_size=2, stride=2), 
            SRBlock(576, 288, 5), 
            SRBlock(288, 288, 5),
            SRBlock(288, 288, 5),
            nn.Upsample(scale_factor=2, str='nearest')]
        self.middle = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(576, 288, kernel_size=1),
            SRBlock(288, 288, 5),
        )
        self.right = nn.Sequential(
            SRBlock(576, 576),
            nn.Upsample(scale_factor=2, str='nearest')
        )
    
    def forward(self, x):
        left_out = self.left(x)
        middle_out = self.middle(x)
        right_out = self.middle_left(middle_out) + self.middle_right(middle_out)
        right_out = self.right(right_out)
        out = left_out + right_out
        return out

class PoseUpBlock(nn.Module):
    """
    Top part of pose block.
    Input assumed to be 576x32x32.
    N_d -- number of depth heat maps per joint
    N_J -- number of body joints
    Returns volumetric heat maps with soft-argmax applied and output 576x32x32 to be fed into the next block.    
    """
    def __init__(self, N_J, N_d):
        super().__init__()
        self.N_J = N_J
        self.N_d = N_d
        self.sc = SCBlock(576, 576, 5)
        self.conv1 = ConvBlock(576, N_d * N_J, 1)
        self.conv2 = ConvBlock(N_d * N_J, 576)
        self.softargmax = SoftArgMax()
        self.batch_norm = nn.BatchNorm2d(576)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out1 = self.sc(x)
        out2 = self.conv1(out1)        
        out2 = self.conv2(out2)
        # reshape to get N_J x N_d x W x H
        heatmaps = out2.view(self.N_J, self.N_d, out2.shape[1], out2.shape[2])
        # average the N_d heatmaps for each N_J to get N_J x W x H
        heatmaps = torch.mean(heatmaps, dim=1)
        heatmaps = self.softargmax(heatmaps)
        return heatmaps, x + self.relu(self.batch_norm(out1 + out2))

class PoseBlock(nn.Module):
    """
    Full pose block    
    """
    def __init__(self, N_J, N_d=16):
        self.pose_down = PoseDownBlock()
        self.pose_up = PoseUpBlock(N_J, N_d)
    
    def forward(self, x):
        out = self.pose_down(x)
        out = self.pose_up(out)
        return out

class PoseEstimationBlock(nn.Module):
    """
    Combines K pose blocks together.
    Returns K different joint estimation vectors of dimension N_J x D.
    """
    def __init__(self, K):
        super().__init__()
        self.K = K
        