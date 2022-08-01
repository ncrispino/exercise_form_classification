"""Holds loss functions for modeling.
"""

import torch

class ElasticNetLoss():
    """

    Attributes:
        visibility_weight: weight for visibility loss.
    
    """

    def __init__(self, visibility_weight=0.01):
        self.visibility_weight = visibility_weight

    def __call__(self, pose_pred, pose_true):
        """
        Mostly copied from deephar/losses.py. Tf code easily converted to PyTorch.

        Args:
            pose_pred: B x dim + 1 x T x N_J tensor; estimated positions of each joint and visibility probability (last dim)
            pose_true: B x dim + 1 x T x N_J tensor; ground truth positions of each joint and visibility flag (last dim)
    
        Returns:
            loss: B x T tensor; loss for each frame

        """
        # Non-visible and outer joints shouldn't be used in the loss. 
        # In preprocessing, they are set as -1e9 if they are either.
        y_pred_vis = pose_pred[:, -1] 
        y_true_vis = pose_true[:, -1] 
        y_pred = pose_pred[:, :-1]
        y_true = pose_true[:, :-1]        
        idx = (y_true > 0).float()
        num_joints = torch.clip(torch.sum(idx, axis=(-1, -3)), 1, None) # By batch and frame.    

        l1 = torch.abs(y_pred - y_true)
        l2 = torch.square(y_pred - y_true)
        bce_loss = torch.nn.BCELoss()  
        # Change BCE only uses visibility dimension.
        bc = self.visibility_weight * bce_loss(y_pred_vis, y_true_vis)
        dummy = 0. * y_pred        
        
        loss = torch.sum(torch.where(idx.bool(), l1 + l2 + bc, dummy),
                axis=(-1, -3)) / num_joints
        return loss.mean() # Over batches and frames.