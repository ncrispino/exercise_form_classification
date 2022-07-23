"""Holds loss functions for modeling."""

import torch

class ElasticNetLoss():
    """
    
    Args:
        pose_pred: B x 3 x T x N_J tensor; estimated positions of each joint
        pose_true: B x 3 x T x N_J tensor; ground truth positions of each joint
    
    Returns:
        loss: B x T tensor; loss for each frame

    """

    def __call__(self, pose_pred, pose_true):
        N_J = pose_pred.shape[-1]
        pose_diff = pose_pred - pose_true
        # Calculate norm across joint coordinates and avg over them: 
        # B x 3 x T x N_J -> B x T x N_J -> B x T.
        loss = (torch.norm(pose_diff, dim=(1), p=1) + torch.norm(pose_diff, dim=(1), p=2)**2).sum(dim=-1)        
        return loss