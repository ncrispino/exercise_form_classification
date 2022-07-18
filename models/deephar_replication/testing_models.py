"""
Make sure input and output are correct shapes.
"""
import torch
from multitask_stem import *
from pose_estimation import *
from action_recognition import *

# # entire model
# x = torch.randint(0, 1, size=(1, 3, 256, 256), dtype=torch.float) # B x C x H x W
# mstem = EntryFlow()
# out = mstem(x)
# print(out.shape)
# pose = PoseEstimation(17)
# joints, out = pose(out)
# print(len(joints))
# print(joints[0].shape)
# print(out.shape)

# # global pooling
# global_mpm = GlobalMaxPlusMinPooling()
# pool_test = torch.randint(0, 1, size=(1, 10, 100, 100), dtype=torch.float)
# pool_out = global_mpm(pool_test)
# print(pool_out.shape)

# channel multiplication
a = torch.randint(0, 1, size=(3, 32, 32))
b = torch.randint(0, 1, size=(5, 32, 32))
print(a * b)