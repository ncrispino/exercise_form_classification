"""
Contains model for pose estimation, consisting of K blocks as seen in Figure 12.
Will split into two blocks.
For upsampling, the authors' code use UpSampling2D with params (2, 2). 
    This should be identical to PyTorch's Upsample using a scale factor of 2 and mode nearest.
"""
from general_models import *
        
class PoseDownBlock(nn.Module):
    """
    Bottom part of pose block.
    Input assumed to be 576 x 32 x 32.
    """
    def __init__(self):
        super().__init__()
        self.left = SRBlock(576, 576, 5)
        self.middle_left = SRBlock(288, 288, 5)
        self.middle_right = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), 
            SRBlock(288, 288, 5), 
            SRBlock(288, 288, 5),
            SRBlock(288, 288, 5),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.middle = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(576, 288, kernel_size=1),
            SRBlock(288, 288, 5),
        )
        self.right = nn.Sequential(
            SRBlock(288, 576, 5),
            nn.Upsample(scale_factor=2, mode='nearest')
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
    Input assumed to be 576 x 32 x 32.      
    N_d -- number of depth heat maps per joint
    N_J -- number of body joints
    """
    def __init__(self, N_J, N_d):
        super().__init__()
        self.N_J = N_J
        self.N_d = N_d
        self.sc = SCBlock(576, 576, 5)
        self.conv1 = ConvBlock(576, N_d * N_J, 1)
        self.conv2 = ConvBlock(N_d * N_J, 576, 1)
        self.softargmax_xy = SoftArgMax()
        self.softargmax_z = SoftArgMax()
        self.batch_norm = nn.BatchNorm2d(576)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Returns probability maps obtained from xy heatmaps, 
        location of all joints (N_J x 3) from volumetric heat maps with soft-argmax applied, 
        and output 576 x 32 x 32 to be fed into the next block.    
        """
        out1 = self.sc(x)
        out2 = self.conv1(out1)        
        # reshape to get B x N_J x N_d x H x W
        heatmaps = out2.view(-1, self.N_J, self.N_d, out2.shape[2], out2.shape[3])
        # average the N_d heatmaps for each N_J to get B x N_J x H x W
        heatmaps_xy = torch.mean(heatmaps, dim=2) # avg on z
        prob_xy = spacial_softmax(heatmaps_xy)    
        joints_xy = self.softargmax_xy(prob_xy, apply_softmax=False)        
        heatmaps_z = torch.mean(heatmaps, dim=(3, 4)) # avg on x & y    
        joints_z = self.softargmax_z(heatmaps_z)        
        joints = torch.cat((joints_xy, joints_z), dim=2)
        # after heatmaps
        out2 = self.conv2(out2)
        return prob_xy, joints, x + self.relu(self.batch_norm(out1 + out2))

class PoseBlock(nn.Module):
    """
    Full pose block.    
    """
    def __init__(self, N_J, N_d):
        super().__init__()
        self.pose_down = PoseDownBlock()
        self.pose_up = PoseUpBlock(N_J, N_d)
    
    def forward(self, x):
        out = self.pose_down(x)
        heatmaps, joints, out = self.pose_up(out)
        return heatmaps, joints, out

class PoseEstimation(nn.Module):
    """
    Combines K pose blocks together.
    Returns the xy heatmaps obtained from the Kth block,
    the joint estimation vector of dimension B * T x N_J x 3 (obtained from the heatmaps) reshaped to B x 3 x T x N_J from the Kth block,
    and a tensor of dimension B * T x 576 x 32 x 32.
    """
    def __init__(self, N_J, B, N_d=16, K=8):
        super().__init__()
        self.K = K
        self.N_J = N_J
        self.B = B
        self.N_d = N_d
        self.prediction_blocks = [PoseBlock(N_J, N_d) for i in range(K)]        
    
    def forward(self, x):        
        out = x
        for block in self.prediction_blocks:
            heatmaps, joints, out = block(out)            
        return heatmaps, joints.view(self.B, joints.shape[2], -1, joints.shape[1]), out