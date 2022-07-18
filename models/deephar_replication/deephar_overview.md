## Overview
Replicating the paper [2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning](https://arxiv.org/pdf/1802.09232.pdf) using PyTorch.
Note that the authors provided their code, which I use as a reference.

I plan to train the pose estimation part on data they used (likely MPII), then train the action recognition part on data I created myself (look up transfer learning for more--maybe can train most of action recognition architecture on broader data and only learn weights for last few blocks).

There are 3 parts to this model: 
1. multitask stem
2. pose estimation model
3. action recognition model
    3. pose recognition model
    4. appearance recognition model.

To run, I need to set the number of joints, N_J. In the paper, they use N_J = number of joints in dataset with most joints.

### To-do
- finish softargmax (may need to return probability maps separately to feed into action recognition)
- investigate relus and batch norm (see below)

### Model Changes
- I will be applying the batch norm before the relu, though in [some data models performed better the other way around](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/).
- I'll assume whenever there's a skip connection, it will be followed by a batch norm and relu, except at the end.
- Figuring out how to extract joint locations from the heatmaps (in PoseUpBlock) was pretty difficult for me. I looked at the implementation in the paper and was still confused; my implementation may be fairly different, though I think I figured it out by looking at the paper's code.
- When there's skip connections, I may not apply batch norm and relu's correctly. **Worth it to look back into this--analyze paper's code again**
- Model treats all inputs as 3D (if they're 2D, will be treated as 3D with depth dimension of 1)

### Misc
I didn't find any PyTorch implementations on [paperswithcode.com](https://paperswithcode.com/paper/2d3d-pose-estimation-and-action-recognition), though it says there is one. So, this will be somewhat novel for that reason (though I'm sure an implementation in PyTorch does exist).

Notes:
- take in RGB image, output pose vector with N_j body joints each of dim D
- use soft arg-max to estimate pose
- combine pose-based with appearance-based recognition
- Can train and use with 2D or 3D data