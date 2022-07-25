"""Makes MPII data into a PyTorch Dataset.

Copied from https://github.com/dluvizon/deephar/blob/cvpr18/deephar/data/mpii.py.
Had to change the loading process as the Matlab file wasn't exactly the same.

"""

import os

import numpy as np
import scipy.io as sio
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, '../../models')
sys.path.insert(1, '../')
from deephar_replication.deephar_utils import *

from data_config import DataConfig
from data_config import mpii_dataconf

from torch.utils.data import Dataset

def load_mpii_mat_annotation(filename):
    """Takes mpii raw Matlab data and makes it easily accessible using Pandas.

    Args:
        filename: Path to the MPII raw Matlab data.
    
    Returns:
        rectidxs: A list of lists of indices of the indexes in the Pandas df 
        for testing [0], training [1], and validation data [2]. 

        annot: A Pandas df containing the annotations for the MPII dataset. 
        It has 4 columns, two of which, 'image' and 'annorect', we care about.
        'image' is the path to the image, and 'annorect' is a list of 
        attributes contained in the annotations.

    """
    mat = sio.loadmat(filename) # Read in as np record array
    mat = mat['RELEASE']        

    # Get indices for training and validation.
    img_train = mat['img_train'][0][0][0]
    train_idxs = np.where(img_train)
    val_idxs = np.where(img_train == 0)
    
    # ~25k total imgs; each holds 4 arrays: image, annorect, frame_sec, vididx.
    annolist = mat['annolist'][0][0][0]
    # annot_tr = annolist[train_idxs]
    # annot_val = annolist[val_idxs] 

    # Change to Pandas for ease of use.
    annot = pd.DataFrame(annolist)
    print(annot.head())
    print(annot.loc[4, 'annorect'][0][0])
    # print(annot['annorect'][0].dtype)
    # print(annot['annorect'][0][0][0].dtype)
    # annot_tr = pd.DataFrame(annot_tr)  
    # annot_val = pd.DataFrame(annot_val)  

    # print(annot_tr_df['image'])
    # print(annot_tr.head())    
    # annot_tr['image'] = annot_tr['image'].apply(lambda x: x[0][0][0][0])
    # annot_val['image'] = annot_val['image'].apply(lambda x: x[0][0][0][0])  
    annot['image'] = annot['image'].apply(lambda x: x[0][0][0][0])
    def isolate_annorect(x):
        try:
            return x[0][0]
        except:
            return np.nan    
    annot['annorect'] = annot['annorect'].apply(isolate_annorect)  
    print(annot.isna().sum())

    # Respect the order of TEST (0), TRAIN (1), and VALID (2)
    rectidxs = [None, train_idxs, val_idxs]
    print(annot.loc[4, 'annorect'][0])
    print([annot.loc[4, 'annorect'][k][0][0] for k in range(4)])
    # images = [None, annot_tr['image'], annot_val['image']]
    # print(annot_tr['annorect'][0][0].dtype)
    # quit()
    # annorect = [None, annot_tr[:, 2], annot_val[:, 2]]
    # rectidxs = [None, train_idxs, val_idxs]
    # images = [None, annot_tr[:, 1], annot_val[:, 1]]
    # annorect = [None, annot_tr[:, 2], annot_val[:, 2]]

    return rectidxs, annot


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
    """Holds images and annotations from MPII human pose benchmark.
    
    See http://human-pose.mpi-inf.mpg.de/#download for more.
    Code copied from/based on MpiiSinglePerson class in mpii.py.

    """
    
    def __init__(self, dataset_path, dataconf, annotation_path, mode, 
                    poselayout=pa16j2d, remove_outer_joints=True, 
                    transform=None, target_transform = None):
        self.mode = mode
        self.dataset_path = dataset_path  
        self.dataconf = dataconf    
        self.poselayout = poselayout
        self.remove_outer_joints = remove_outer_joints   
        self.load_annotations(annotation_path)     
        # self.load_annotations(os.path.join(annotation_path, 'annotations.mat'))
    
    def load_annotations(self, filename):
        try:
            rectidxs, annot = load_mpii_mat_annotation(filename)
            images = annot['image']

            self.samples = {}
            self.samples[TEST_MODE] = [] # No samples for test
            self.samples[TRAIN_MODE] = serialize_annorect(
                    rectidxs[TRAIN_MODE], annorect[TRAIN_MODE])
            self.samples[VALID_MODE] = serialize_annorect(
                    rectidxs[VALID_MODE], annorect[VALID_MODE])
            self.images = images

        except:
            warning('Error loading the MPII dataset!')
            raise

    def load_image(self, key):
        try:
            annot = self.samples[self.mode][key]
            image = self.images[self.mode][annot['imgidx']][0]
            imgt = T(Image.open(os.path.join(
                self.dataset_path, 'images', image)))
        except:
            warning('Error loading sample key/mode: %d/%d' % (key, self.mode))
            raise

        return imgt

    def __len__(self):
        return len(self.samples[self.mode])
    
    def __getitem__(self, key, fast_crop=False): # key is idx
        output = {}

        if self.mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
        else:
            dconf = self.dataconf.get_fixed_config()

        imgt = self.load_image(key, self.mode)
        annot = self.samples[self.mode][key]

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

        head = annot['head']
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
        output['headsize'] = calc_head_size(annot['head'])
        output['afmat'] = imgt.afmat.copy()

        return output

if __name__=='__main__':    
    # import h5py
    # f = h5py.File('annot.h5', 'r')
    # print(f.keys())

    # annotation_path = 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    # mpii = Mpii('mpii_human_pose_v1.tar.gz', mpii_dataconf, annotation_path,
    # mode=TRAIN_MODE)
    # print(type(mpii))
    annot = pd.read_json('mpii_annotations.json')
    print(annot.info())
    print(annot.head())
    print(annot['objpos'])
    print(annot['joint_self'])
    print(annot['scale_provided'])
    print(annot['objpos_other'])
    print(annot['joint_others'])
    print(annot['scale_provided_other'])
