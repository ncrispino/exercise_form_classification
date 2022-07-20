"""
Make sure input and output are correct shapes.
"""
import torch
from multitask_stem import *
from pose_estimation import *
from action_recognition import *

# entire model
x = torch.randint(0, 1, size=(2, 10, 3, 256, 256), dtype=torch.float) # B x T x C x H x W
T = x.shape[1]
mstem = EntryFlow()
entry_input = mstem(x)
print(entry_input.shape)
pose = PoseEstimation(17, T=T)
prob_maps, joints, pose_out = pose(entry_input)
# print(len(joints))
# print(joints[0].shape)
print(prob_maps.shape)
print(joints.shape)
print(pose_out.shape)
# action = ActionRecognition(2)
# joints = joints.permute(2, 0, 1).unsqueeze(0)
# print(joints.shape)
# out = action(joints, entry_out, heatmaps)
# print(out.shape)
action_recognition = ActionRecognition(2, T=T)
out = action_recognition(joints, entry_input, prob_maps)
print(out)

# global pooling
# global_mpm = GlobalMaxPlusMinPooling()
# pool_test = torch.randint(0, 1, size=(1, 10, 100, 100), dtype=torch.float)
# pool_out = global_mpm(pool_test)
# print(pool_out.shape)

# testing only action recognition
# for joints
# action_start = ActionStart(True)
# out = action_start(x)
# action_block = ActionBlock(True, 2)
# actions, out = action_block(out)
# action_combined = ActionCombined(True, 2, 4)
# actions, out = action_combined(x)
# print(out.shape)

# x = torch.randint(0, 1, size=(1, 3, 10, 17), dtype=torch.float) # B x N_f x T x N_J
# entry_input = torch.randint(0, 1, size=(1, 576, 32, 32), dtype=torch.float) # B x N_f x H x W
# prob_maps = torch.randint(0, 1, size=(1, 17, 32, 32), dtype=torch.float) # B x N_J x H x W
# action_recognition = ActionRecognition(2)
# out = action_recognition(x, entry_input, prob_maps)
# print(out)