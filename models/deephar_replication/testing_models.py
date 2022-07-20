"""
Make sure input and output are correct shapes.
"""
import torch
from multitask_stem import *
from pose_estimation import *
from action_recognition import *

# entire model
# x = torch.randint(0, 1, size=(1, 3, 256, 256), dtype=torch.float) # B x C x H x W
# mstem = EntryFlow()
# entry_out = mstem(x)
# print(entry_out.shape)
# pose = PoseEstimation(17)
# heatmaps, joints, pose_out = pose(entry_out)
# # print(len(joints))
# # print(joints[0].shape)
# print(heatmaps.shape)
# print(pose_out.shape)
# action = ActionRecognition(2)
# joints = joints.permute(2, 0, 1).unsqueeze(0)
# print(joints.shape)
# out = action(joints, entry_out, heatmaps)
# print(out.shape)

# global pooling
# global_mpm = GlobalMaxPlusMinPooling()
# pool_test = torch.randint(0, 1, size=(1, 10, 100, 100), dtype=torch.float)
# pool_out = global_mpm(pool_test)
# print(pool_out.shape)

# testing only action
# pose rec
x = torch.randint(0, 1, size=(1, 3, 10, 17), dtype=torch.float) # B x N_f x T x N_J
action_start = ActionStart(True)
out = action_start(x)
action_block = ActionBlock(True, 2)
actions, out = action_block(out)
