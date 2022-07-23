"""Make sure input and output are correct shapes.
"""

import torch
from full_model import Model

# Testing entire model
x = torch.randint(0, 1, size=(2, 10, 3, 256, 256), dtype=torch.float) # B x T x C x H x W
model = Model(N_J=17, N_a=2)
out = model(x)
print(torch.exp(out))


# Old testing
# import torch
# from multitask_stem import *
# from pose_estimation import *
# from action_recognition import *
# from loss import *

# entire model
# x = torch.randint(0, 1, size=(2, 10, 3, 256, 256), dtype=torch.float) # B x T x C x H x W
# B = x.shape[0]
# mstem = EntryFlow()
# entry_input = mstem(x)
# # print("entry: " + str(entry_input.shape))
# pose = PoseEstimation(17, B=B)
# prob_maps, joints, pose_out = pose(entry_input)
# # print("prob maps: " + str(prob_maps.shape))
# # print("joints: " + str(joints.shape))
# # print("pose_out: " + str(pose_out.shape))
# action_recognition = ActionRecognition(2, B=B)
# out = action_recognition(joints, entry_input, prob_maps)
# print(out)

# global pooling
# global_mpm = GlobalMaxPlusMinPooling()
# pool_test = torch.randint(0, 1, size=(20, 10, 100, 100), dtype=torch.float)
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
# x = torch.randint(0, 1, size=(20, 16, 32, 32), dtype=torch.float) # B x C x H x W
# y = torch.randint(0, 1, size=(20, 16, 15), dtype=torch.float) # B x C x D
# softargmax = SoftArgMax()
# out_x = softargmax(x)
# out_y = softargmax(y)
# print(out_x)
# print(out_y)

# loss
# x = torch.randint(0, 10, size=(2, 3, 10, 17), dtype=torch.float) # B x C x T x N_J
# y = torch.randint(0, 10, size=(2, 3, 10, 17), dtype=torch.float) # B x C x T x N_J
# # 1-norm
# print((torch.norm(x - y, dim=(1), p=1) == abs(x - y).sum(dim=(1))).sum())
# # 2-norm
# print((torch.norm(x - y, dim=(1), p=2) == ((x - y)**2).sum(dim=(1)).sqrt()).sum())
# loss
# enl = ElasticNetLoss()
# print(enl(x, y))