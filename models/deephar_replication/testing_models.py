"""
Make sure input and output are correct shapes.
"""
import torch
from multitask_stem import *
from pose_estimation import *
from action_recognition import *

# entire model
# x = torch.randint(0, 1, size=(2, 10, 3, 256, 256), dtype=torch.float) # B x T x C x H x W
# B = x.shape[0]
# mstem = EntryFlow()
# entry_input = mstem(x)
# print(entry_input.shape)
# pose = PoseEstimation(17, B=B)
# prob_maps, joints, pose_out = pose(entry_input)
# print(prob_maps.shape)
# print(joints.shape)
# print(pose_out.shape)
# action_recognition = ActionRecognition(2, B=B)
# out = action_recognition(joints, entry_input, prob_maps)
# print(out.shape, out[0], out[1])

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

# softargmax
x = torch.randint(0, 1, size=(20, 16, 32, 32), dtype=torch.float) # B x C x H x W
y = torch.randint(0, 1, size=(20, 16, 15), dtype=torch.float) # B x C x D
softargmax = SoftArgMax()
out_x = softargmax(x)
out_y = softargmax(y)
print(out_x)
print(out_y)