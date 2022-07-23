"""Contains model for action recognition, composed of two parts.

1. pose-based recognition model 
2. appearance-based recognition model.
The same general architecture is used for both, as seen in Figure 13.
They are then combined with softmax to get action outputs.

As for pose estimation, the time dimension is combined into the batch dimension.
This information is saved and fed into the model so the output can be reshaped.

"""

import torch
from torch import nn
import numpy as np
from general_models import ConvBlock
from general_models import MaxPlusMinPooling
from general_models import GlobalMaxPlusMinPooling
from general_models import kronecker_prod

class ActionStart(nn.Module):
    """Part of model below action prediction block in Figure 13.

    Attributes:
        pose_rec: whether to use pose-based recognition. If so,
            each conv has half the features as that of appearance recognition. 
      
    """

    def __init__(self, pose_rec):
        super().__init__()
        self.pose_rec = pose_rec
        self.dim = 3 if pose_rec else 576
        if self.pose_rec: # half of what's shown in Figure 13
            self.conv_left1 = ConvBlock(self.dim, 6, (3, 1), padding='same')
            self.conv_middle1 = ConvBlock(self.dim, 12, 3, padding='same')
            self.conv_right1 = ConvBlock(self.dim, 18, (3, 5), padding='same')
            self.conv_left2 = ConvBlock(36, 56, 3, padding='same')
            self.conv_right2 = nn.Sequential(
                ConvBlock(36, 32, 1, padding='same'),
                ConvBlock(32, 56, 3, padding='same')
            )
        else:
            self.conv_left1 = ConvBlock(self.dim, 12, (3, 1), padding='same')
            self.conv_middle1 = ConvBlock(self.dim, 24, 3, padding='same')
            self.conv_right1 = ConvBlock(self.dim, 36, (3, 5), padding='same')
            self.conv_left2 = ConvBlock(72, 112, 3, padding='same')
            self.conv_right2 = nn.Sequential(
                ConvBlock(72, 64, 1, padding='same'),
                ConvBlock(64, 112, 3, padding='same')
            )
        self.maxplusmin = MaxPlusMinPooling(2, padding=0)
    
    def forward(self, x):
        """

        Args:
            B x N_f x T x N_J tensor.

        """

        out_left = self.conv_left1(x)
        out_middle = self.conv_middle1(x)
        out_right = self.conv_right1(x)        
        out = torch.cat((out_left, out_middle, out_right), dim=1)
        out_left2 = self.conv_left2(out)
        out_right2 = self.conv_right2(out)
        out2 = torch.cat((out_left2, out_right2), dim=1)
        out2 = self.maxplusmin(out2)
        return out2

class ActionBlock(nn.Module):
    """Action prediction block in Figure 13.
    
    Attributes:
        pose_rec: pose_rec: whether to use pose-based recognition. If so,
            each conv has half the features as that of appearance recognition.
        N_a: number of actions.

    """

    def __init__(self, pose_rec, N_a):
        super().__init__()
        self.N_a = N_a
        if pose_rec:
            dim = 112
        else:
            dim = 224
        self.conv1 = ConvBlock(dim, dim // 2, 1, padding='same')
        self.conv2 = ConvBlock(dim // 2, dim, 3, padding='same')
        self.conv3 = ConvBlock(dim, dim, 3, padding='same')
        # Padding should be equivalent to 'same' in tf -- done in forward as it's based on output.
        self.maxplusmin = MaxPlusMinPooling(2)
        self.conv4 = ConvBlock(dim, N_a, 3, padding='same')        
        self.conv5 = ConvBlock(N_a, dim, 3, padding='same')
        # Input: N_a x H x W. output: N_a.
        self.global_maxplusmin = GlobalMaxPlusMinPooling()
        self.softmax = nn.Softmax(0)
    
    def forward(self, x, up_shape):    
        """
        
        Args:
            x: B x N_f x T x N_J tensor.
            up_shape: size of tensor such that the output from the 
                upsampling block can be added to the output before the pooling
                (out1).

        Returns:
            B x N_a tensor with probabilities for each action.

            B x 224 x T/2 x N_J/2 tensor.
    
        """  

        self.upsample = nn.Upsample(size=up_shape, mode='nearest')

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
        actions = self.softmax(actions.flatten())        
        return actions, out

class ActionCombined(nn.Module):
    """Pose or appearance recognition using K action recognition blocks
    
    Referred to as a 'stacked architecture with intermediate supervision.'
    See Figure 5 for more.

    Attributes:
        pose_rec: whether to use pose-based recognition. If so,
            each conv has half the features as that of appearance recognition.
        N_a: number of actions.
        K: number of action recognition blocks.
        B: batch size.

    """   

    def __init__(self, pose_rec, N_a, K, B):
        super().__init__()
        self.N_a = N_a
        self.K = K
        self.B = B
        self.action_start = ActionStart(pose_rec)
        self.action_blocks = [ActionBlock(pose_rec=pose_rec, N_a=self.N_a) for i in range(K)]        
    
    def forward(self, x):
        """

        Args:
            3 x T x N_J tensor if doing pose recognition.
            N_f x T x N_J tensor if doing appearance recognition.
        
        Returns:
            B x N_a tensor with probabilities for each action from Kth block.
            B x 224 x T/2 x N_J/2 tensor output from Kth block.        

        """

        # Dimensions to upsample to so I can add outputs in the action blocks.
        up_shape = list(np.array(x.shape[-2:]) // 2) 
        # Keep track of previous block output and add to input of next block.        
        out = self.action_start(x)
        prev_out = 0   
        for block in self.action_blocks:      
            actions, new_out = block(out + prev_out, up_shape=up_shape)
            prev_out = out
            out = new_out                                     
        actions = actions.view(self.B, -1)   
        return actions, out   

def appearance_extract(entry_input, prob_maps, B):
    """Extracts localized appearance features to be fed into the action blocks.

    Args:
        entry_input: B x 576 x H x W output from global entry flow, which is
            the multitask stem based on Inception-V4.        
        prob_maps: B x N_J x H x W probability maps obtained at the 
            end of pose estimation part (softmax applied to the xy heatmaps).
        B: batch size.
    
    Returns:
        B x N_f x T x N_J tensor.    

    """

    out = kronecker_prod(entry_input, prob_maps) # B * T x N_f x N_J x H x W        
    out = torch.sum(out, dim=(-2, -1)) # B * T x N_f x N_J        
    out = out.view(B, out.shape[1], -1, out.shape[2])
    return out

class ActionRecognition(nn.Module):
    """Combines pose-based recognition and appearance-based recognition 
    using a fully-connected layer with Softmax activation.

    Uses only the action predicted in the last block.

    Attributes:
        N_a: number of actions.
        B: batch size.
        K: number of action recognition blocks.

    """

    def __init__(self, N_a, B, K=4):
        super().__init__()
        self.N_a = N_a
        self.B = B
        self.K = K        
        self.pose_rec = ActionCombined(True, N_a, K, B)
        self.appear_rec = ActionCombined(False, N_a, K, B)
        self.fc = nn.Linear(2 * N_a, N_a)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, pose_input, entry_input, prob_maps):
        """

        Args:
            pose_input: B x N_J x 3 input from joints for each timestep.
            entry_input: B x 576 x H x W output from global entry flow, which is
                the multitask stem based on Inception-V4.        
            prob_maps: B x N_J x H x W probability maps obtained at the 
                end of pose estimation part (softmax applied to the xy heatmaps).   

        Returns:
            B x N_a tensor with log probabilities for each action. 
                Use exp to retrieve probabilities.
    
        """

        appearance_input = appearance_extract(entry_input, prob_maps, self.B)
        pose_actions, pose_out = self.pose_rec(pose_input)
        appearance_actions, appearance_out = self.appear_rec(appearance_input)
        # Isolate actions in last block B x 2 * N_a.        
        fc_input = torch.cat((pose_actions, appearance_actions), dim=1) 
        # Output of fc is B x N_a then log softmax N_a to get probabilities.
        out = self.log_softmax(self.fc(fc_input))
        return out
        
