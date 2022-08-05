## Datasets
### MPII
I will use MPII to initially train the pose estimation model (2D pose estimation).
As the methods provided by the authors didn't work with the direct downloaded data from [MPII](http://human-pose.mpi-inf.mpg.de/), I used a method from TensorLayer that put the data into Python lists (```mpii_tensorlayer.py```), edited so as to include all the data the authors of the paper I'm replicating used. Then, I combined it with the authors' preprocessing methods and put it into a PyTorch Dataset in ```mpii_torch.py```. Note that the annotations did exist, but don't anymore (see the link under MPII on the [install page](https://github.com/dluvizon/deephar/blob/master/INSTALL.md)). As the authors' project cannot be directly used, my PyTorch replicated repo may be a bit more useful than I otherwise thought.

TODO: What are afmat and head size used for in mpii? Used for validation, so maybe model should output? I think I need to add functions to do this
as in pose regression methods here: https://github.com/dluvizon/deephar/blob/fbebb148a3b7153f911b86e1a7300aa32c336f31/deephar/models/reception.py#L1.
So, I'll add joint visibility (from heatmap to Sigmoid) in forward methods as in Section 3.1.3.


Training: https://github.com/dluvizon/deephar/blob/fbebb148a3b7153f911b86e1a7300aa32c336f31/exp/mpii/train_mpii_singleperson.py
Find out how to add augmented training data as in the paper.

Testing: Dataset set up wrong, as it will show a list of length 0 (no annotations for test set). Need to specifically account for this