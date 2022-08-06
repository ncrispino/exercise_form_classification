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
from exp.common.mpii_tools import eval_singleperson_pckh

import wandb
run = wandb.init(project='deephar_replication', mode='disabled')
wandb.config.update({    
    'lr': 3e-4,
    'num_epochs': 100,
    'dataset': 'mpii', 
    'batch_size': 2, # Set batch size to 24 on GPU, as in paper.
    'pose_blocks': 4,
})

# GPUs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# From parser.py
TEST_MODE = 0 # TODO: Change dataloader to work with test (no annotations I think)
TRAIN_MODE = 1
VALID_MODE = 2

mpii_train = Mpii(mode=TRAIN_MODE, dataset_path='../../data/mpii/data/')
mpii_val = Mpii(mode=VALID_MODE, dataset_path='../../data/mpii/data/')

train_dataloader = DataLoader(mpii_train, batch_size=wandb.config.batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(mpii_val, batch_size=wandb.config.batch_size, shuffle=True, drop_last=True)
val_loader_one_batch = DataLoader(mpii_val, batch_size=len(mpii_val), shuffle=True, drop_last=True)

N_J = 16
pose_dim = 2

pose_model = nn.Sequential(
    EntryFlow(),
    PoseEstimation(16, wandb.config.batch_size, pose_dim, K=wandb.config.pose_blocks)
)
pose_model.to(device)

# For summary table of params:
# from torchinfo import summary
# summary(pose_model, input_size=(batch_size, 1, 3, 256, 256))

def joint_training(model, loss_fn, optimizer, num_epochs, train_loader, val_loader):
    # For validation
    top_score = 0.0
    top_epoch = 0
    best_model_state_dict = model.state_dict()
    # Training:
    model.train()
    # wandb.watch(model, log_freq=1, log='all')        
    for epoch in range(1, num_epochs + 1):
        loss_train = 0.0
        for output in train_loader:
            imgs = output['frame'].to(device)           
            joint_vis_true = output['pose'].permute(0, 2, 1).unsqueeze(2).to(device) # B x T x N_J x pose_dim + 1 with visibility concat to end -> B x pose_dim + 1 x T x N_J          
            imgs = imgs.unsqueeze(1) # add T=1 channel
            visibility, _, all_joints = model(imgs)
            joints = all_joints[-1] # Last block.
            joint_vis_pred = torch.concat([joints, visibility], dim=1) # B x 3 x T x N_J
            loss = loss_fn(joint_vis_pred, joint_vis_true) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            wandb.log({'epoch': epoch, 'train_loss_batch': loss.item()}) 
          #  'joint_vis_pred': joint_vis_pred, 'joint_vis_true': joint_vis_true, 'images': [wandb.Image(im) for im in imgs]})

        wandb.log({'loss_train': loss_train/len(train_loader)})

        # Validation
        curr_score = joint_validation(model, loss_fn, val_loader, epoch)
        print(f'curr_score: {curr_score}')
        print('curr_score one batch: ', joint_validation(model, loss_fn, val_loader_one_batch, epoch))
        # print(f'top score: {top_score}')
        if curr_score > top_score:
            top_score = curr_score
            top_epoch = epoch
            best_model_state_dict = model.state_dict()
    wandb.log({'top_pckh_overall': top_score, 'top_epoch': top_epoch})

    # Save model with best validation score.
    torch.save(best_model_state_dict, 'mpii_pose_only.pth')

def joint_validation(model, loss_fn, val_loader, epoch=-1):        
    """

    Args:
        epoch: Epoch number. If not applicable, will log -1.

    Returns:
        top_score: Top score on validation set. Takes average of pckh scores.        
    
    """
    model.eval()
    loss_val = 0.0  
    top_pckh = 0.0  
    with torch.no_grad():           
        for output in val_loader:
            imgs = output['frame'].to(device).unsqueeze(1)
            joint_vis_true  = output['pose'].permute(0, 2, 1).unsqueeze(2).to(device)                 
            joints_true = joint_vis_true[:, :-1, :, :]                
            afmat  = output['afmat']
            headsize  = output['headsize']                 
            visibility , _, all_joints  = model(imgs)
            joint_vis_pred  = torch.concat([all_joints[-1], visibility], dim=1)
            loss = loss_fn(joint_vis_pred, joint_vis_true)
            loss_val += loss.item()
            pckh_scores_batch = eval_singleperson_pckh(
                all_joints, joints_true, afmat_val=afmat, 
                headsize_val=headsize, batch_size=wandb.config.batch_size, 
                num_blocks=wandb.config.pose_blocks, verbose=1)         
            # Scores is a list; pick max. TODO: change for multiple batches?
            top_pckh_batch = max(pckh_scores_batch)
            # top_pckh = top_pckh_epoch if top_pckh_epoch > top_pckh else top_pckh
            top_pckh += top_pckh_batch
            # print(f'top_pckh: {top_pckh}')
            # Note I'm only logging the last list of scores.
            pckh_table = wandb.Table(columns=[k for k in range(len(pckh_scores_batch))])
            pckh_table.add_data(*pckh_scores_batch)
            wandb.log({'val_loss_batch': loss.item(), 'pckh_scores_per_batch': pckh_table, 'top_pckh_per_batch': top_pckh_batch})                        
        wandb.log({'loss_val': loss_val/len(val_loader), 'epoch': epoch})  
    return top_pckh/len(val_loader)

# Overfit one batch (starting with batch_size=2 as 20 is too much to handle for my CPU)
one_batch_train = [next(iter(train_dataloader))] # Make list so it can be iterated over.
one_batch_val = [next(iter(val_dataloader))]

loss_fn = ElasticNetLoss()
optimizer = optim.Adam(pose_model.parameters(), lr=wandb.config.lr)

joint_training(pose_model, loss_fn, optimizer, num_epochs=wandb.config.num_epochs, train_loader=one_batch_train, val_loader=one_batch_val)

# Save model with wandb
artifact = wandb.Artifact('mpii_only_model', type='model')
artifact.add_file('mpii_pose_only.pth')
run.log_artifact(artifact)
run.finish()

# Check model saved correctly
# run = wandb.init(project='deephar_replication') #, mode='disabled')
# wandb.config.update({    
#     'lr': 3e-4,
#     'num_epochs': 100,
#     'dataset': 'mpii', 
#     'batch_size': 2, # Set batch size to 24 on GPU, as in paper.
#     'pose_blocks': 4,
# })
# pose_model_saved = nn.Sequential(
#     EntryFlow(),
#     PoseEstimation(16, wandb.config.batch_size, pose_dim, K=wandb.config.pose_blocks)
# )
# pose_model_saved.to(device)
# model = pose_model_saved
# model.load_state_dict(torch.load('mpii_pose_only.pth'))
# model.eval()
# saved_val_score = joint_validation(pose_model_saved, loss_fn, one_batch_val)

# wandb.log({'saved model val score': saved_val_score})