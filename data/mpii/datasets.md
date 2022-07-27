## Datasets
### MPII
I will use MPII to initially train the pose estimation model (2D pose estimation).
As the methods provided by the authors didn't work with the direct downloaded data from [MPII](http://human-pose.mpi-inf.mpg.de/), I used a method from TensorLayer that put the data into Python lists (```mpii_tensorlayer.py```), edited so as to include all the data the authors of the paper I'm replicating used. Then, I combined it with the authors' preprocessing methods and put it into a PyTorch Dataset in ```mpii_torch.py```.