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
- may need to return probability maps separately to feed into action recognition
- **make work with T dimensions -- incrementally test all action rec blocks**  see if there's an alternative to Keras' TimeDistributed for PyTorch
    - need to convert joints to 3 x T x N_J
    - can I train with a different number of timesteps, i.e., with a video of 5 timesteps and a video of 10 timesteps?
- investigate relus and batch norm (see below)
- check params of model and compare with the paper's (visualize)
- remove print statements
- use biases?
- torch functional vs nn? Can I create new instances of cnns in forward functions?

### Model Changes
- I will be applying the batch norm before the relu, though in [some data models performed better the other way around](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/).
- I'll assume whenever there's a skip connection, it will be followed by a batch norm and relu, except at the end.
- Figuring out how to extract joint locations from the heatmaps (in PoseUpBlock) was pretty difficult for me. I looked at the implementation in the paper and was still confused; my implementation may be fairly different, though I think I figured it out by looking at the paper's code.
- When there's skip connections, I may not apply batch norm and relu's correctly. **Worth it to look back into this--analyze paper's code again**
- Model treats all inputs as 3D (if they're 2D, will be treated as 3D with depth dimension of 1). To do this, I need to wrap everything in a PyTorch equivalent of the [TimeDistributed layer in Keras](https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4), which applies the same weights to all time steps.
    - Ex: For an input B x T x C x H x W, a TimeDistributed wrapped Conv2D would apply the same instance of Conv2D at each time step. Based on [this discussion forum](https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4), I think the equivalent to this in PyTorch would be reshaping the input as B * T x C x H x W, then separating time back out before the output.
    - Also, note that in the ActionBlock, the output may have dim 0, throwing an error, as they take the floor of T / 2.
- I was having trouble making sure action convs are right dim -- also make sure maxplusmin pooling is equivalent to tf 'same' -- it may only work for my given shape inputs.
    - Feeding an odd input into ActionBlock (like (1, 3, 10, 17) will get an error, as the pooling results in (1, 3, 5, 8) while the upsampling results in(1, 3, 6, 8)). I'm not sure how the authors deal with this--it only seems like a problem when the latter two dimensions of the N_f x T x N_J input are odd. It seems they use an even number of joints, meaning it's not a problem, and also batches of 2 video clips. So, they use all even numbers, which may be why there's no error. If I do want to use an odd number, either I could add zero padding or upsample directly to this size. I chose to do the latter. This is done in the *ActionBlock method*.
- For categorical cross-entropy loss, I'll change the output of the neural network to use log softmax instead of softmax, then use NLLLoss. This should be [equivalent to cross-entropy loss and allow the probabilities to be easily recovered](https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss).

### Misc
I didn't find any PyTorch implementations on [paperswithcode.com](https://paperswithcode.com/paper/2d3d-pose-estimation-and-action-recognition), though it says there is one. So, this will be somewhat novel for that reason (though I'm sure an implementation in PyTorch does exist).

Notes:
- take in RGB image, output pose vector with N_j body joints each of dim D
- use soft arg-max to estimate pose
- combine pose-based with appearance-based recognition
- Can train and use with 2D or 3D data