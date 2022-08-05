"""Holds tools for modeling with MPII.

Copied from official repo. Only includes what's necessary.

https://github.com/dluvizon/deephar/blob/fbebb148a3b7153f911b86e1a7300aa32c336f31/exp/common/mpii_tools.py

"""

import torch
from measures import pckh
from measures import pckh_per_joint
from deephar_utils import *

def eval_singleperson_pckh(pred, pval, afmat_val, headsize_val,
        win=None, batch_size=8, refp=0.5, map_to_pa16j=None, num_blocks=1,
        verbose=0):
    """Evaluation of single-person PCKh for MPII.

    Copied from repo.
    Finds
    Changes: 
    - Always assumed there's a time dimension.
    - My time dimension is in a different place.

    Args: (some may not be right as they're autofilled with copilot)        
        pred: Output joint position from model.
        pval: Ground truth joint position.
        afmat_val: Affine transformation matrix for validation data.
        headsize_val: Head size for validation data.
        win: NO IMPLEMENTATION.
        batch_size: Batch size for evaluation.
        refp: Reference point for PCKh.
        map_to_pa16j: Map MPII joints to PA16 joints.
        pred_per_block: Number of predictions per block.
        verbose: Verbosity.

    """

    A = afmat_val[:]
    y_true = pval[:]    

    # Change to correct shape for pckh input (B * T, N_J, dim).
    pred = pred.view(pred.shape[0], -1, pred.shape[4], pred.shape[2]) # K x B x dim x T x N_J -> K x B * T x N_J x dim    
    y_true = y_true.contiguous().view(-1, y_true.shape[3], y_true.shape[1]) # B x dim x T x N_J -> B * T x N_J x dim    
        
    y_true = transform_pose_sequence(A.numpy(), y_true.cpu(), inverse=True)    

    if map_to_pa16j is not None:
        y_true = y_true[:, map_to_pa16j, :]
    scores = []
    # if verbose:
    #     printc(WARNING, 'PCKh on validation:')

    for b in range(num_blocks):

        y_pred = pred[b]

        if map_to_pa16j is not None:
            y_pred = y_pred[:, map_to_pa16j, :]              
        y_pred = transform_pose_sequence(A.numpy(), y_pred.cpu(), inverse=True)        
        s = pckh(y_true, y_pred, headsize_val, refp=refp)
        if verbose:
            printc(WARNING, ' %.1f' % (100*s))
        scores.append(s)

        if b == num_blocks-1:
            if verbose:
                printcn('', '')
            pckh_per_joint(y_true, y_pred, headsize_val, pa16j2d,
                    verbose=verbose)

    return scores