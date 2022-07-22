"""
Puts entire model together to run.
"""
from multitask_stem import *
from pose_estimation import *
from action_recognition import *

class Model(nn.Module):

    def __init__(self, N_J=17, N_a=2):
        self.N_J = N_J
        self.N_a = N_a

    def forward(self, x):
        B = x.shape[0]
        mstem = EntryFlow()
        entry_input = mstem(x)        
        pose = PoseEstimation(self.N_J, B=B)
        prob_maps, joints, pose_out = pose(entry_input)                        
        action_recognition = ActionRecognition(self.N_a, B=B)
        out = action_recognition(joints, entry_input, prob_maps)
        return out