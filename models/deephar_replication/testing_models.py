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