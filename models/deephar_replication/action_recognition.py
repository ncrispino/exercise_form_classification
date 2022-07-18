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
        super().__init__()
        self.N_a = N_a
        self.conv1 = ConvBlock(224, 112, 1)
        self.conv2 = ConvBlock(112, 224, 3)
        self.conv3 = ConvBlock(224, 224, 3)
        self.maxplusmin = MaxPlusMinPooling(2)
        self.conv4 = ConvBlock(224, N_a, 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = ConvBlock(224, 224, 3)
        self.global_maxplusmin = GlobalMaxPlusMinPooling() # input: N_a x H x W. output: N_a
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = x + out
        out1 = self.conv3(out)
        out2 = self.maxplusmin(out1)
        heatmaps = self.conv4(out2)
        out2 = self.upsample(heatmaps)
        out2 = self.conv5(out2)
        out2 = out2 + out1
        out = out + out2
        actions = self.global_maxplusmin(heatmaps)
        actions = self.softmax(actions)
        return actions, out

class ActionCombined(nn.Module):
    """
    Pose recognition using K action recognition blocks -- a 'stacked architecture with intermediate supervision'
    See Figure 5 for more.
    pose_rec -- True if doing pose recognition
    """    
    def __init__(self, pose_rec, N_a, K=4):
        super().__init__()
        self.N_a = N_a
        self.K = K
        self.action_blocks = [ActionBlock(pose_rec=pose_rec, N_a=self.N_a) for i in range(K)]        
    
    def forward(self, x):
        all_actions = torch.zeros((self.K, self.N_a)) # holds a scalar for each action from each prediction block
        # keep track of output of previous block and add to input of next block
        out = x             
        prev_out = 0   
        for k, block in enumerate(self.action_blocks):      
            actions, new_out = block(out + prev_out)
            prev_out = out
            out = new_out           
            all_actions[k] = actions            
        return all_actions, out   

def AppearanceExtract(entry_input, prob_maps):
    """
    Extracts localized appearance features to be fed into action blocks.
    entry_input -- 576 x H x W output from global entry flow (multitask stem based on Inception-V4)
    prob_maps -- N_J x H x W probability maps obtained at the end of pose estimation part (softmax applied to heatmaps)
    """

        