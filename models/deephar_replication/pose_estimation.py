"""Contains model for pose estimation, consisting of K blocks.

Seen in Figures 2 and 12. I will split the estimation into two blocks.
The bottom block is the one with three branches and the top block
begins with the 5x5 separable convolution with output dim 576.

Note for upsampling, the authors' code use UpSampling2D with params (2, 2). 
This should be identical to PyTorch's Upsample using a 
scale factor of 2 and mode nearest.

Also note that the time dimension is combined into the batch dimension.
This information is saved and fed into the model so the output can be reshaped.

"""

import torch
from torch import nn
from general_models import ConvBlock
from general_models import SRBlock
from general_models import SCBlock
from general_models import SoftArgMax
from general_models import spacial_softmax
        
class PoseDownBlock(nn.Module):
    """Bottom part of pose block."""

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
        """

        Args:
            B x 576 x 32 x 32 tensor.

        """

        left_out = self.left(x)
        middle_out = self.middle(x)        
        right_out = self.middle_left(middle_out) + self.middle_right(middle_out)
        right_out = self.right(right_out)
        out = left_out + right_out
        return out

class PoseUpBlock(nn.Module):
    """Top part of pose block.

    Attributes:
        N_J: number of body joints
        N_d: number of depth heat maps per joint     

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
        self.sigmoid = nn.Sigmoid()      
    
    def forward(self, x):
        """

        Args: 
            B x 576 x 32 x 32 tensor.            
        
        Returns:
            B x N_J x 1 probability of a certain joint being visible.

            B x N_J x H x W probability maps obtained from xy heatmaps.

            B x N_J x 3 location of all joints. 

            B x 576 x 32 x 32 output tensor to be fed into the next block.  

        """

        out1 = self.sc(x)
        out2 = self.conv1(out1)        
        # Reshape to get B x N_J x N_d x H x W.
        heatmaps = out2.view(-1, self.N_J, self.N_d, out2.shape[2], out2.shape[3])
        # Average the N_d heatmaps for each N_J to get B x N_J x H x W.
        heatmaps_xy = torch.mean(heatmaps, dim=2) # Avg on z.
        prob_xy = spacial_softmax(heatmaps_xy)    
        joints_xy = self.softargmax_xy(prob_xy, apply_softmax=False)        
        heatmaps_z = torch.mean(heatmaps, dim=(3, 4)) # Avg on x & y.
        joints_z = self.softargmax_z(heatmaps_z)        
        joints = torch.cat((joints_xy, joints_z), dim=2)

        # After heatmaps for block output.
        out2 = self.conv2(out2)

        # Visibility is sigmoid applied to the sum of global max pooling 
        # on each of the heatmaps. See deephar/models/reception.py.
        v_xy, _ = torch.max(heatmaps_xy, dim=(2, 3))
        v_z, _ = torch.max(heatmaps_xy, dim=2)
        visibility = self.sigmoid(v_xy + v_z).unsqueeze(-1)
        return visibility, prob_xy, joints, x + self.relu(self.batch_norm(out1 + out2))

class PoseBlock(nn.Module):
    """Full pose block. 

    Attributes:
        N_J: number of body joints
        N_d: number of depth heat maps per joint   

    """

    def __init__(self, N_J, N_d):
        super().__init__()
        self.pose_down = PoseDownBlock()
        self.pose_up = PoseUpBlock(N_J, N_d)
    
    def forward(self, x):
        out = self.pose_down(x)
        visibility, prob_maps, joints, out = self.pose_up(out)
        return visibility, prob_maps, joints, out

class PoseEstimation(nn.Module):
    """Combines K pose blocks together.

    Attributes:
        N_J: number of body joints
        B: number of batches
        N_d: number of depth heat maps per joint
        K: number of blocks in the pose estimation network 

    """

    def __init__(self, N_J, B, N_d=16, K=8):
        super().__init__()
        self.K = K
        self.N_J = N_J
        self.B = B
        self.N_d = N_d
        self.prediction_blocks = [PoseBlock(N_J, N_d) for i in range(K)]        
    
    def forward(self, x): 
        """

        Args:
            B x 576 x 32 x 32 tensor.

        Returns:
            B x N_J x 1 probability of a certain joint being visible.

            B x N_J x H x W probability maps obtained from the Kth block.

            B x 3 x T x N_J location of all joints from the Kth block.

            Note: Does not return B x 576 x 32 x 32 output tensor from the 
            Kth block, as it's not used in action recognition.

        """ 

        out = x
        for block in self.prediction_blocks:
            visibility, prob_maps, joints, out = block(out)            
        return visibility, prob_maps, joints.view(self.B, joints.shape[2], -1, joints.shape[1])