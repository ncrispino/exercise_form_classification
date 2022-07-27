"""Makes MPII data into a PyTorch Dataset.

Most is copied from https://github.com/dluvizon/deephar/blob/cvpr18/deephar/data/mpii.py.
Had to change the loading process as the Matlab file wasn't exactly the same.
Instead, I use a function from the tensorlayer library
with slight modifications so as to have all the necessary data.

"""

import os
from random import seed

import numpy as np
import scipy.io as sio
import pandas as pd
from PIL import Image
# import tensorlayer as tl
from mpii_tensorlayer import load_mpii_pose_dataset # use customized function

import sys
sys.path.insert(0, '../../models')
sys.path.insert(1, '../')
from deephar_replication.deephar_utils import *

from data_config import DataConfig
from data_config import mpii_dataconf

from torch.utils.data import Dataset

def serialize_annorect(rectidxs, annot):
    # assert len(rectidxs) == len(annorect)
    annorect = annot['annorect']

    sample_list = []
    for i in range(len(rectidxs)):
        rec = rectidxs[i]
        for j in rec:
        # for j in range(rec.size):
            # idx = rec[j,0]-1 # Convert idx from Matlab
            ann = annorect.loc[j, :]
            annot = {}
            try:
                annot['head'] = [annot.loc[j, 'annorect'][k][0][0] for k in range(4)] #ann['head'][0,0][0] # Coordinates of head rectangle (there are 4).
                annot['objpos'] = ann['objpos'][0,0][0]
                annot['scale'] = ann['scale'][0,0][0,0]
                annot['pose'] = ann['annopoints'][0,0]
                annot['imgidx'] = j
                sample_list.append(annot)
            except:
                print('Error:', j)
                continue

    return sample_list


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

    """
    
    def __init__(self, dataconf, annotation_path, mode, dataset_path='data\mpii_human_pose',
                    poselayout=pa16j2d, remove_outer_joints=True, 
                    transform=None, target_transform = None):                    
        self.mode = mode  
        self.dataset_path = dataset_path
        self.dataconf = dataconf    
        self.poselayout = poselayout
        self.remove_outer_joints = remove_outer_joints   
        img_train_list, ann_train_list, img_test_list, ann_test_list = load_mpii_pose_dataset(is_16_pos_only=True)    
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
        elif mode == VAL_MODE:
            self.img_list = img_train_list[val_idxs]
            self.ann_list = ann_train_list[val_idxs] 
    
    # def load_annotations(self, filename):
    #     try:
    #         # rectidxs, annot = load_mpii_mat_annotation(filename)
    #         # images = annot['image']

    #         self.samples = {}
    #         self.samples[TEST_MODE] = [] # No samples for test
    #         self.samples[TRAIN_MODE] = serialize_annorect(
    #                 rectidxs[TRAIN_MODE], annorect[TRAIN_MODE])
    #         self.samples[VALID_MODE] = serialize_annorect(
    #                 rectidxs[VALID_MODE], annorect[VALID_MODE])
    #         self.images = annot['img_paths']

    #     except:
    #         warning('Error loading the MPII dataset!')
    #         raise

    def load_image(self, idx):
        try:
            image = self.img_list[idx]
            imgt = T(Image.open(os.path.join(
                self.dataset_path, 'images', image)))
        except:
            warning('Error loading sample key/mode: %d/%d' % (idx, self.mode))
            raise

        return imgt

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx, fast_crop=False):
        """

        The provided code has annot['pose'], which according to the 
        transform functions should have shape [N_joints, dim].
        I already edited the tensorlayer code so as to provide only the visible joints.
        If a joint is not visible according to the retrieved data, I set the respective coordinates to NaN.

        Note that this may differ from how the authors intended, 
        as I'm not entirely sure how their pose data was organized.
        In my version, I remove the non-visible joints from tensorlayer data
        and also remove the joints according to the authors' method.

        TODO: How to deal with more than one pose in image? treat separately?
    
        """
        output = {}

        if self.mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
        else:
            dconf = self.dataconf.get_fixed_config()

        imgt = self.load_image(idx, self.mode)
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
        output['frame'] = normalize_channels(imgt.asarray(),
                channel_power=dconf['chpower'])

        p = np.empty((self.poselayout.num_joints, self.poselayout.dim))
        p[:] = np.nan

        head = annot['head_rect']
        p[self.poselayout.map_to_mpii, 0:2] = \
                transform_2d_points(imgt.afmat, annot['pose'].T, transpose=True)
        if imgt.hflip:
            p = p[self.poselayout.map_hflip, :]

        # Set invalid joints and NaN values as an invalid value
        p[np.isnan(p)] = -1e9
        v = np.expand_dims(get_visible_joints(p[:,0:2]), axis=-1)
        if self.remove_outer_joints:
            p[(v==0)[:,0],:] = -1e9

        output['pose'] = np.concatenate((p, v), axis=-1)
        output['headsize'] = calc_head_size(annot['head_rect'])
        output['afmat'] = imgt.afmat.copy()

        return output

if __name__=='__main__':           
    img_train_list, ann_train_list, img_test_list, ann_test_list = load_mpii_pose_dataset(is_16_pos_only=True)
    print(len(img_train_list), len(img_test_list)) 
    print(ann_train_list[0]) #, 0].keys())   
    # print(ann_train_list[2248])
    # print(ann_train_list[0][0].keys())
    # print(img_train_list[0], 'yo', ann_train_list[0])
    # image = tl.vis.read_image(img_train_list[0])
