"""Makes MPII data into a PyTorch Dataset.

Most is copied from https://github.com/dluvizon/deephar/blob/cvpr18/deephar/data/mpii.py.
Had to change the loading process as the Matlab file wasn't exactly the same.
Instead, I use a function from the tensorlayer library
with slight modifications so as to have all the necessary data.

Note that this file can't be run individually anymore; needs to be run through training.py.

"""

from audioop import reverse
import os
from random import seed

import numpy as np
import torch
from PIL import Image
from data.mpii.mpii_tensorlayer import load_mpii_pose_dataset # use customized function

import sys
sys.path.insert(0, '../../models')
sys.path.insert(1, '../')
from deephar_replication.deephar_utils import *

from data.data_config import mpii_dataconf

from torch.utils.data import Dataset

def calc_head_size(head_annot):
    head = np.array([float(head_annot[0]), float(head_annot[1]),
        float(head_annot[2]), float(head_annot[3])])
    return 0.6 * np.linalg.norm(head[0:2] - head[2:4])

class Mpii(Dataset):
    """Holds images and annotations from MPII human pose benchmark for single person.
    
    See http://human-pose.mpi-inf.mpg.de/#download for more.
    Code copied from/based on MpiiSinglePerson class in mpii.py.
    I am using tensorlayer, as they do all the preprocessing for me.
    Link: https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/files/dataset_loaders/mpii_dataset.html.
    I modified the code in mpii_tensorlayer to include scale, objpos, 
    and pose (N_J, 2) array holding all visible joints, with non-visible set to NaN.

    This splits into training and testing sets. According to the paper I'm replicating, 
    around 15k images are used for training, 3k for validation, and 7k for testing.
    Currently, there are 18k images in the training set and 7k in the testing set.
    So, I will manually split the training set, using 1/6th of the images for validation.
    This is done before I split each image's annotations into single person annotations,
    resulting in ~40k total data human poses, which may be more unbalanced
    as I split the data before this separation. (e.g., in training, goes from 15065 to 16675)
    This paper https://cse.buffalo.edu/~siweilyu/papers/eccv18.pdf is clear 
    about this individual person split, which is what I'll do.

    """
    
    def __init__(self, mode, dataconf=mpii_dataconf, dataset_path='data',
                    poselayout=pa16j2d, remove_outer_joints=True, 
                    transform=None, target_transform = None):                    
        self.mode = mode  
        self.dataset_path = dataset_path # Set to just data folder as tensorlayer finds the mpii_human_pose dir.
        self.dataconf = dataconf    
        self.poselayout = poselayout
        self.remove_outer_joints = remove_outer_joints   
        img_train_list, ann_train_list, img_test_list, ann_test_list = load_mpii_pose_dataset(path=dataset_path, is_16_pos_only=True)    
        # Convert to numpy arrays for math operations and easier slicing.
        img_train_list = np.array(img_train_list)
        ann_train_list = np.array(ann_train_list, dtype='object')
        img_test_list = np.array(img_test_list)
        ann_test_list = np.array(ann_test_list, dtype='object')                    
        shuffled_train_idxs = np.arange(len(img_train_list))
        np.random.seed(42)
        np.random.shuffle(shuffled_train_idxs)
        val_idxs = shuffled_train_idxs[:int(len(img_train_list) / 6)]
        train_idxs = shuffled_train_idxs[int(len(img_train_list) / 6):]      
        if mode == TEST_MODE:
            self.img_list = img_test_list
            self.ann_list = ann_test_list
        elif mode == TRAIN_MODE:
            self.img_list = img_train_list[train_idxs]
            self.ann_list = ann_train_list[train_idxs]
        elif mode == VALID_MODE:
            self.img_list = img_train_list[val_idxs]
            self.ann_list = ann_train_list[val_idxs] 
        # Split ann_list into single person annotations. 
        # Note that indexes with img list will no longer match so need to create
        # a new img list with repeated images.
        single_ann_list = []
        single_img_list = []
        for i, img_ann in enumerate(self.ann_list):
            for person in img_ann:
                single_ann_list.append(person)
                single_img_list.append(self.img_list[i])
        self.img_list = single_img_list
        self.ann_list = single_ann_list

    def load_image(self, idx):
        try:
            image_path = self.img_list[idx]
            imgt = T(Image.open(image_path))
        except:
            warning('Error loading sample key/mode: %d/%d' % (idx, self.mode))
            raise

        return imgt

    def __len__(self):
        return len(self.ann_list)
    
    def __getitem__(self, idx, fast_crop=False):
        """

        The provided code has annot['pose'], which according to the 
        transform functions should have shape [N_joints, dim].

        The images are augmented according to self.dataconf.

        I already edited the tensorlayer code so as to provide only the visible joints.
        If a joint is not visible according to the retrieved data, I set the respective coordinates to NaN.

        Note that this may differ from how the authors intended, 
        as I'm not entirely sure how their pose data was organized.
        In my version, I remove the non-visible joints from tensorlayer data
        and also remove the joints according to the authors' method.

        Returns:
            output dict containing:
                frame: 3 x 256 x 256 input tensor
                pose: N_joints x 3 tensor with last dimension a visibility flag

        """
        output = {}

        if self.mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator() # For data augmentation.
        else:
            dconf = self.dataconf.get_fixed_config() # Doesn't change data for validation/testing.

        imgt = self.load_image(idx)
        annot = self.ann_list[idx]

        scale = 1.25*annot['scale']
        objpos = np.array([annot['objpos'][0], annot['objpos'][1] + 12*scale])
        objpos += scale * np.array([dconf['transx'], dconf['transy']])
        winsize = 200 * dconf['scale'] * scale
        winsize = (winsize, winsize)
        output['bbox'] = objposwin_to_bbox(objpos, winsize)

        if fast_crop:
            """Slightly faster method, but gives lower precision."""
            imgt.crop_resize_rotate(objpos, winsize,
                    self.dataconf.crop_resolution, dconf['angle'])
        else:
            imgt.rotate_crop(dconf['angle'], objpos, winsize)
            imgt.resize(self.dataconf.crop_resolution)

        if dconf['hflip'] == 1:
            imgt.horizontal_flip()

        imgt.normalize_affinemap()
        output['frame'] = torch.tensor(normalize_channels(imgt.asarray(),
                channel_power=dconf['chpower']), dtype=torch.float32).permute(2, 0, 1) # C x H x W

        p = np.empty((self.poselayout.num_joints, self.poselayout.dim))
        p[:] = np.nan                   

        head = annot['head_rect']
        p[self.poselayout.map_to_mpii, 0:2] = \
                transform_2d_points(imgt.afmat, annot['pose'], transpose=True)
        if imgt.hflip:
            p = p[self.poselayout.map_hflip, :]
        
        # Set invalid joints and NaN values as an invalid value
        p[np.isnan(p)] = -1e9
        v = np.expand_dims(get_visible_joints(p[:,0:2]), axis=-1)
        if self.remove_outer_joints:
            p[(v==0)[:,0],:] = -1e9        

        output['pose'] = torch.tensor(np.concatenate((p, v), axis=-1), dtype=torch.float32)
        output['headsize'] = torch.tensor(calc_head_size(annot['head_rect']), dtype=torch.float32)
        output['afmat'] = torch.tensor(imgt.afmat.copy(), dtype=torch.float32) # Affine transformation saved.

        return output        

if __name__=='__main__':              
    mpii = Mpii(mode=TRAIN_MODE)
    # print(len(mpii))
    output = mpii[0]
    # print(frame.shape, pose.shape)
    # print(frame)
    # print(pose)
