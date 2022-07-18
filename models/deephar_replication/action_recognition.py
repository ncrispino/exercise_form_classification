"""
Contains model for action recognition, which contains the pose-based recognition model and the appearance-based recognition model.
The same general architecture is used for both, as seen in Figure 13.
"""
from general_models import *

class ActionStart(nn.Module):
    """
    Takes in B x N_f x T x N_J, where N_f is the number of dimensions (the number of coordinates for each point, T is the number of frames (temporal), and N_J is the number of joints
    pose_rec -- true if doing pose recognition (each conv does half the features that appearance recognition does)
    dim -- number of dimensions that define the position of each joint
    """
    def __init__(self, pose_rec, dim):
        super().__init__()
        self.pose_rec = pose_rec
        self.dim = dim
        if self.pose_rec: # half of what's shown in Figure 13
            self.conv_left1 = ConvBlock(dim, 6, (3, 1))
            self.conv_middle1 = ConvBlock(dim, 12, 3)
            self.conv_right1 = ConvBlock(dim, 18, (3, 5))
            self.conv_left2 = ConvBlock(36, 56, 3)
            self.conv_right2 = nn.Sequential(
                ConvBlock(36, 32, 1),
                ConvBlock(32, 56, 3)
            )
        else:
            self.conv_left1 = ConvBlock(dim, 12, (3, 1))
            self.conv_middle1 = ConvBlock(dim, 24, 3)
            self.conv_right1 = ConvBlock(dim, 36, (3, 5))
            self.conv_left2 = ConvBlock(72, 112, 3)
            self.conv_right2 = nn.Sequential(
                ConvBlock(72, 64, 1),
                ConvBlock(64, 112, 3)
            )
        self.maxplusmin = MaxPlusMinPooling(2)
    
    def forward(self, x):
        out_left = self.conv_left1(x)
        out_middle = self.conv_middle1(x)
        out_right = self.conv_right1(x)
        out = torch.cat((out_left, out_middle, out_right), dim=1)
        out_left1 = self.left2(out)
        out_right1 = self.right2(out)
        out1 = torch.cat((out_left1, out_right1), dim=1)
        out1 = self.maxplusmin(out1)
        return out1

class ActionBlock(nn.Module):
    """
    Action prediction block.
    Takes in input of B x N_f x T x N_J
    Outputs softmax from action heat maps and a tensor of shape B x 224 x T/2 x N_J/2.
    pose_rec -- true if doing pose recognition (each conv does half the features that appearance recognition does)
    N_a -- number of actions
    """
    def __init__(self, pose_rec, N_a):
        self.N_a = N_a
        self.conv1 = ConvBlock(224, 112, 1)
        self.conv2 = ConvBlock(112, 224, 3)
        self.conv3 = ConvBlock(224, 224, 3)
        self.maxplusmin = MaxPlusMinPooling(2)
        self.conv4 = ConvBlock(224, N_a, 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = ConvBlock(224, 224, 3)
        self.global_maxplusmin = GlobalMaxPlusMinPooling(2)
    
    def forward(self, x):
