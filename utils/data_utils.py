# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pickle
import numpy as np
import torch

from monai import data, transforms
from monai.data import *
import pandas as pd
import random
import h5py


def get_loader(args):
    '''Get the dataloader for the CCII dataset.'''
    # Transforms
    def __transforms__(augmentation=True, npy=None, args=None):
        RANDOM_BRIGHTNESS = 7
        RANDOM_CONTRAST = 5
        pre_size_x = 233
        final_size_x = 192
        pre_size_y = 197
        final_size_y = 192
        pre_size_z = 34
        final_size_z = 32
        spatial_limit_x = int((pre_size_x-final_size_x)/2.0)
        spatial_limit_y = int((pre_size_y-final_size_y)/2.0)
        spatial_limit_z = int((pre_size_z-final_size_z)/2.0)
        # pre_top_left = int((512-pre_size)/2.0)
        npy_normalized = npy.astype(np.float32) / 255.0 # cast to float
        if augmentation:
            # random flip
            if random.uniform(0, 1) < 0.5: #horizontal flip
                npy_normalized = np.flipud(npy_normalized)
            # color jitter
            br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
            npy_normalized = npy_normalized + br
            # Random contrast
            cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
            npy_normalized = npy_normalized * cr
            # clip values to 0-1 range
            npy_normalized = np.clip(npy_normalized, 0, 1.0)
            # random crop
            offset_x = random.randint(-spatial_limit_x, spatial_limit_x)
            offset_y = random.randint(-spatial_limit_y, spatial_limit_y)
            offset_z = random.randint(-spatial_limit_z, spatial_limit_z)
            npy_normalized = npy_normalized[
                spatial_limit_z+offset_z : spatial_limit_z+final_size_z+offset_z,
                spatial_limit_x+offset_x : spatial_limit_x+final_size_x+offset_x,
                spatial_limit_y+offset_y : spatial_limit_y+final_size_y+offset_y
                ]
        else:
            offset_x = 0
            offset_y = 0
            offset_z = 0
            npy_normalized = npy_normalized[
                spatial_limit_z+offset_z : spatial_limit_z+final_size_z+offset_z,
                spatial_limit_x+offset_x : spatial_limit_x+final_size_x+offset_x,
                spatial_limit_y+offset_y : spatial_limit_y+final_size_y+offset_y
                ]

        return npy_normalized

    train_files_name = args.train.img_file_path
    train_text_name = args.train.text_file_path
    val_files_name = args.test.img_file_path
    val_text_name = args.test.text_file_path


    train_ds = ICP(mode='train', data=train_files_name, text=train_text_name, transforms=__transforms__, augmentation=True, args=args)
    print(f'=>Train len {len(train_ds)}')
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    val_ds = ICP(mode='val', data=val_files_name, text=val_text_name, transforms=__transforms__, augmentation=False,args=args)
    print(f'=>Val len {len(val_ds)}')
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader

def get_test_loader(args):
    def __transforms__(augmentation=True, npy=None, args=None):
        RANDOM_BRIGHTNESS = 7
        RANDOM_CONTRAST = 5
        pre_size_x = 233
        final_size_x = 192
        pre_size_y = 197
        final_size_y = 192
        pre_size_z = 34
        final_size_z = 32
        spatial_limit_x = int((pre_size_x-final_size_x)/2.0)
        spatial_limit_y = int((pre_size_y-final_size_y)/2.0)
        spatial_limit_z = int((pre_size_z-final_size_z)/2.0)
        # pre_top_left = int((512-pre_size)/2.0)
        npy_normalized = npy.astype(np.float32) / 255.0 # cast to float
        if augmentation:
            # random flip
            if random.uniform(0, 1) < 0.5: #horizontal flip
                npy_normalized = np.flipud(npy_normalized)
            # color jitter
            br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
            npy_normalized = npy_normalized + br
            # Random contrast
            cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
            npy_normalized = npy_normalized * cr
            # clip values to 0-1 range
            npy_normalized = np.clip(npy_normalized, 0, 1.0)
            # random crop
            offset_x = random.randint(-spatial_limit_x, spatial_limit_x)
            offset_y = random.randint(-spatial_limit_y, spatial_limit_y)
            offset_z = random.randint(-spatial_limit_z, spatial_limit_z)
            npy_normalized = npy_normalized[
                spatial_limit_z+offset_z : spatial_limit_z+final_size_z+offset_z,
                spatial_limit_x+offset_x : spatial_limit_x+final_size_x+offset_x,
                spatial_limit_y+offset_y : spatial_limit_y+final_size_y+offset_y
                ]
        else:
            offset_x = 0
            offset_y = 0
            offset_z = 0
            npy_normalized = npy_normalized[
                spatial_limit_z+offset_z : spatial_limit_z+final_size_z+offset_z,
                spatial_limit_x+offset_x : spatial_limit_x+final_size_x+offset_x,
                spatial_limit_y+offset_y : spatial_limit_y+final_size_y+offset_y
                ]

        return npy_normalized
    
    val_files_name = os.path.join(args.dataset_path, args.split_list_test)
    val_ds = ICP(mode='val', data=val_files_name, transforms=__transforms__, augmentation=False,args=args)
    print(f'=>Val len {len(val_ds)}')
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)
    return val_loader

class ICP(torch.utils.data.Dataset):
    def __init__(self, mode, data=None, text=None, transforms=None, augmentation=True, args=None, use_h5=True):
        super().__init__()
   
        with open(data, 'r') as f:
            self.img_list = [line.strip() for line in f]
            
            if use_h5:
                self.img_list = [x.replace('registrated_filtered','registrated_filtered_h5').replace('.nii.gz','.h5') for x in self.img_list]
        if os.path.exists(text):
            with open(text, 'r') as f:
                self.text_list = [line.strip() for line in f]
        else:
            self.text_list = ['Abnormal brain CT scan' for _ in range(len(self.img_list))]
        
        if mode == 'val':
            datas = list(zip(self.img_list,self.text_list))
            random.shuffle(datas)
            self.img_list, self.text_list = zip(*datas)
        
        self.augmentation = augmentation

        self.transforms = transforms
        self.args = args
        

    def __getitem__(self, index):
        data_path = self.img_list[index]
        icp_value = int(self.img_list[index].split('_')[-1].split('.')[0])

        target = int(icp_value >= 20)
        target_onehot = np.eye(2)[target]

        if icp_value >= self.args.max_value:
            icp_value = 1
        else:
            icp_value = icp_value / self.args.max_value
        if icp_value < 0:
            icp_value = 0

        if data_path.endswith('.h5'):
            file = h5py.File(data_path, 'r')
            data = file['img'][()].transpose(2, 1, 0)

        npy_normalized = self.transforms(self.augmentation, data, self.args)
        npy_normalized = npy_normalized[np.newaxis].repeat(3,axis=0)
        
        text_all = self.text_list[index].split('###')[-4]
        text_basic = self.text_list[index].split('###')[-2]
        text_CT = self.text_list[index].split('###')[-3]
        
        return {
            'image': npy_normalized,
            'label': target,
            'icp_value_norm': icp_value,
            'label_onehot': target_onehot,
            'img_path': data_path,
            'text_ct_raw': text_CT,
            'text_basic_raw': text_basic,
            'text_all': text_all
        }

    def __len__(self):
        return len(self.img_list)