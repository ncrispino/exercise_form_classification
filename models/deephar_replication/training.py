"""Trains both parts of the model.

Note: Won't work on my CPU. 
Use https://cloud.google.com/free/ to train on GPUs.

Recall: I need to set training/eval flags.
Also may need to change loss and get to work with T channel.

Uses Weights & Biases to keep track of training/val runs.
Initially, try to overfit on one batch to ensure model works.
Add some visualizations to see if it's working.

"""

import sys
sys.path.insert(0, '../../')
from data.data_config import mpii_dataconf
from data.mpii.mpii_tensorlayer import load_mpii_pose_dataset
from data.mpii.mpii_torch import Mpii

from torch.utils.data import DataLoader

mpii_train = Mpii(mode=0, dataset_path='../../data/mpii/data/')
train_dataloader = DataLoader(mpii_train, batch_size=2, shuffle=True) # Set batch size to 24 on GPU, as in paper.

for img, annot in train_dataloader:
    print(img, annot)
    quit()
