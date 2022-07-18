"""
Make sure input and output
"""
import torch
from multitask_stem import *
from pose_estimation import *

x = torch.randint(0, 1, size=(1, 3, 256, 256), dtype=torch.float) # B x C x W x H
mstem = MultitaskStem()
out = mstem(x)
print(out.shape)
pose = PoseEstimation(17)
joints, out = pose(out)
print(len(joints))
print(joints[0].shape)
print(out.shape)

global_mpm = GlobalMaxPlusMinPooling()
pool_test = torch.randint(0, 1, size=(1, 10, 100, 100), dtype=torch.float)
pool_out = global_mpm(pool_test)
print(pool_out.shape)