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
        # Non-visible and outer joints shouldn't be used in the loss. 
        # In preprocessing, they are set as -1e9 if they are either.        
        # idx = torch.nonzero(pose_true[:, :-1, :, :] > 0)
        idx = pose_true > 0
        # poses = pose_pred[idx]
        poses = torch.where(idx, pose_pred, 0 * pose_pred)
        print(poses.shape)
        print(poses)        
        N_J = idx.sum(-1)[:, :, 0]
        print('Idx: ' + str(idx))       
        print('N_J: ' + str(N_J))
        # Calculate norm across joint coordinates and avg over them: 
        # B x 3 x T x N_J -> B x T x N_J -> B x T.
        if self.include_bincross:
            bce_loss = torch.nn.BCELoss()            
            bc = 0.01 * bce_loss(pose_pred[:, -1], pose_true[:, -1])
            pose_diff = pose_pred[:, :-1] - pose_true[:, :-1]
        else:
            bc = 0
            pose_diff = pose_pred - pose_true

        # print(pose_pred)
        # print(pose_true)
        # # print(pose_diff) 
        # print(pose_diff.abs().sum(axis=1))
        # print((pose_diff**2).sum(axis=1))
        loss_by_frame = torch.norm(pose_diff, dim=(1), p=1) + torch.norm(pose_diff, dim=(1), p=2)**2 + bc
        # Zero out loss for non-visible and outer joints if applicable.
        loss = (torch.where(idx, loss_by_frame, 0 * pose_pred)).sum()/N_J
        return loss