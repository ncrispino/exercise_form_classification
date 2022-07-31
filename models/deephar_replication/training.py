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

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import ElasticNetLoss
from multitask_stem import EntryFlow
from pose_estimation import PoseEstimation

import wandb
wandb.init(project='deephar_replication') #, mode='disabled')

# From parser.py
TEST_MODE = 0 # TODO: Change dataloader to work with test (no annotations I think)
TRAIN_MODE = 1
VAL_MODE = 2

mpii_train = Mpii(mode=TRAIN_MODE, dataset_path='../../data/mpii/data/')

# Set batch size to 24 on GPU, as in paper.
batch_size = 2 #24
train_dataloader = DataLoader(mpii_train, batch_size=batch_size, shuffle=True)

N_J = 16
pose_dim = 2

pose_model = nn.Sequential(
    EntryFlow(),
    PoseEstimation(16, batch_size, pose_dim)
)

def joint_training(model, loss_fn, optimizer, num_epochs, train_loader):
    model.train()
    wandb.watch(model, log_freq=1, log='all')    
    num_images = 0
    for epoch in range(1, num_epochs + 1):
        loss_train = 0.0
        for output in train_loader:
            imgs = output['frame']            
            joint_vis_true = output['pose'].permute(0, 2, 1).unsqueeze(2) # B x T x N_J x pose_dim + 1 with visibility concat to end            
            imgs = imgs.unsqueeze(1) # add T=1 channel
            visibility, _, joints = model(imgs) # B x 3 x T x N_J -- third channel should be zero here as I'm doing 2D?
            joint_vis_pred = torch.concat([joints, visibility], dim=1) 
            loss = loss_fn(joint_vis_pred, joint_vis_true)            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            wandb.log({'images': [wandb.Image(im) for im in imgs], 'epoch': epoch, 'loss': loss.item()})

        wandb.log({'loss_train': loss_train/len(train_loader)})

# def joint_validate(model, val_loader):

# Overfit one batch (starting with batch_size=2 as 20 is too much to handle for my CPU)
one_batch = [next(iter(train_dataloader))] # Make list so it can be iterated over.
wandb.config.update({    
    'lr': 3e-4,
    'num_epochs': 100,
    'dataset': 'mpii',    
})

loss_fn = ElasticNetLoss()
optimizer = optim.Adam(pose_model.parameters(), lr=wandb.config.lr)

joint_training(pose_model, loss_fn, optimizer, num_epochs=wandb.config.num_epochs, train_loader=one_batch)