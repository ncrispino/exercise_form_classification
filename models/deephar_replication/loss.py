"""Holds loss functions for modeling.
"""

import torch

class ElasticNetLoss():
    """
    
    Args:
        pose_pred: B x 4 x T x N_J tensor; estimated positions of each joint and visibility probability (4th dim)
        pose_true: B x 4 x T x N_J tensor; ground truth positions of each joint and visibility flag (4th dim)
    
    Returns:
        loss: B x T tensor; loss for each frame

    """

    def __init__(self, include_bincross=True):
        self.include_bincross = include_bincross

    def __call__(self, pose_pred, pose_true):
        N_J = pose_pred.shape[-1]        
        # Calculate norm across joint coordinates and avg over them: 
        # B x 3 x T x N_J -> B x T x N_J -> B x T.
        if self.include_bincross:
            bce_loss = torch.nn.BCELoss()            
            bc = 0.01 * bce_loss(pose_pred[:, -1], pose_true[:, -1])
            pose_diff = pose_pred[:, :-1] - pose_true[:, :-1]
        else:
            bc = 0
            pose_diff = pose_pred - pose_true        
        loss = (torch.norm(pose_diff, dim=(1), p=1) + torch.norm(pose_diff, dim=(1), p=2)**2 + bc).mean()     
        return loss