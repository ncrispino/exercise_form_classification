"""Puts entire model together to run.
"""

from torch import nn
from multitask_stem import EntryFlow
from pose_estimation import PoseEstimation
from action_recognition import ActionRecognition

class Model(nn.Module):
    """Combines all parts of the model.

    Args:        
        N_J: number of joints
        N_a: number of actions

    """
    
    def __init__(self, N_J=17, N_a=2):
        super().__init__()
        self.N_J = N_J
        self.N_a = N_a

    def forward(self, x):
        """

        Args:
            input tensor of shape B x T x C x H x W.

        """

        B = x.shape[0]
        mstem = EntryFlow()
        entry_input = mstem(x)        
        pose = PoseEstimation(self.N_J, B=B)
        prob_maps, joints, pose_out = pose(entry_input)                        
        action_recognition = ActionRecognition(self.N_a, B=B)
        out = action_recognition(joints, entry_input, prob_maps)
        return out