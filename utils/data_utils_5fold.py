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
import json
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append("..")
from data.augment import transforms as transforms_before
# from monai import data, transforms
# from monai.data import *
import pandas as pd
import random
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

LOGICSET_NAME_TRANSLATIONS = {
    "基底节": "basal ganglia hemorrhage",
    "丘脑": "thalamic hemorrhage",
    "脑室": "intraventricular hemorrhage",
    "小脑": "cerebellar hemorrhage",
    "脑干": "brainstem hemorrhage",
    "额颞顶叶/额顶叶/枕叶/额叶/颞叶": "lobar hemorrhage",
    "蛛网膜下腔": "subarachnoid hemorrhage",
    "脑实质内": "intraparenchymal hemorrhage",
    "硬膜下": "subdural hemorrhage",
    "硬膜外": "epidural hemorrhage",
    "高血压": "hypertensive etiology",
    "脑动脉瘤破裂": "ruptured cerebral aneurysm",
    "外伤": "traumatic etiology",
    "其他": "other etiology",
    "脑积水": "hydrocephalus",
    "脑疝": "brain herniation",
    "脑血肿": "hematoma",
    "脑动脉瘤": "cerebral aneurysm",
    "出血破入脑室": "hemorrhage extension into ventricles",
    "脑淀粉样变": "cerebral amyloid angiopathy",
    "脑血管畸形": "cerebrovascular malformation",
    "脑静脉窦血栓形成": "cerebral venous sinus thrombosis",
    "脑梗死": "cerebral infarction",
    "脑肿瘤": "brain tumor",
    "脑挫伤/颅骨骨折": "cerebral contusion or skull fracture",
    "脑室炎": "ventriculitis",
    "颅骨缺失": "skull defect",
    "硬脑膜动静脉瘘": "dural arteriovenous fistula",
}


def load_logic_set_schema_names():
    schema_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hemorrhage.json")
    if not os.path.exists(schema_path):
        return []

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    flattened_names = []
    for group_names in schema.get("脑出血分类（新）", {}).values():
        flattened_names.extend(group_names)
    return flattened_names

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def prepare_weights(labels, reweight='sqrt_inv', max_target=51, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(max_target)}
    # mbr
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
        
    if reweight == 'sqrt_inv':
        value_dict2 = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict2 = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict2[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict2.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label_new = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label_new]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

def categorize(value, thresholds):
    """
    Categorize a value based on thresholds.

    Parameters:
        value (float): The value to be categorized.
        thresholds (list of float): A sorted list of thresholds.

    Returns:
        int: The category index.
    """
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    return len(thresholds)

def get_loader(args,shuffle_val=True,using_LDS=False,debug=False,log_dir=None, distributed=False, rank=0, world_size=1):
    '''Get the dataloader for the CCII dataset.'''
    # Transforms
    def __transforms__(augmentation=True, npy=None, args=None):
        RANDOM_BRIGHTNESS = 7
        RANDOM_CONTRAST = 5
        pre_size_z, pre_size_x, pre_size_y = npy.shape
        # pre_size_x = 233
        final_size_x = 224
        # pre_size_y = 197
        final_size_y = 224
        # pre_size_z = 16#34
        final_size_z = pre_size_z#32
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
    train_label_name = args.train.label_file_path
    val_files_name = args.test.img_file_path
    val_text_name = args.test.text_file_path
    val_label_name = args.test.label_file_path

    train_ds = ICP(mode='train', data=train_files_name, fold_split=args.fold_split, fold_idx=args.fold, fold_num=args.fold_num, text=train_text_name, label=train_label_name, shuffle_val=shuffle_val, transforms=__transforms__, transformer=args.train.transformer, augmentation=True, args=args,using_LDS=using_LDS,debug=debug,log_dir=log_dir)
    print(f'=>Train len {len(train_ds)}')
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True, shuffle=(train_sampler is None), sampler=train_sampler
    )

    val_ds = ICP(mode='val', data=val_files_name, fold_split=args.fold_split, fold_idx=args.fold, fold_num=args.fold_num, text=val_text_name, label=val_label_name, shuffle_val=shuffle_val, transforms=__transforms__, transformer=args.test.transformer, augmentation=False,args=args,debug=debug,log_dir=log_dir)
    print(f'=>Val len {len(val_ds)}')
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=8, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, train_ds, val_ds

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
    val_ds = ICP(mode='val', data=val_files_name, fold_idx=args.fold, transforms=__transforms__, augmentation=False,args=args)
    print(f'=>Val len {len(val_ds)}')
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)
    return val_loader

class ICP(torch.utils.data.Dataset):
    def __init__(self, mode, fold_split=False, fold_idx=0, fold_num=10, data=None, text=None, label=None, transforms=None, transformer=None, augmentation=True, args=None, use_h5=True, shuffle_val=True,using_LDS=False,debug=False, log_dir=None):
        super().__init__()
        with open(data, 'r') as f:
            self.img_list_all = [line.strip().replace('/home/yizhenyu/projects/data', '/home/yizhenyu/data/ICP/data/huashan') for line in f]
            # if use_h5:
            #     self.img_list_all = [x.replace('registrated_filtered','registrated_filtered_h5').replace('.nii.gz','.h5') for x in self.img_list_all]
        # self.img_list_all = [x.replace(x.split('/')[-2],'registrated_filtered_h5') for x in self.img_list_all]
        with open(text, 'r') as f:
            self.text_list_all = [line.strip() for line in f]
        with open(label, 'r') as f:
            self.label_list_all = [eval(line.strip()) for line in f]
  
        self.label_list_all = np.array(self.label_list_all)
        class_count = np.sum(self.label_list_all, axis=0)
        # filter out classes with less than 15 samples
        valid_classes = np.where(class_count > 15)[0]
        self.valid_classes = valid_classes.tolist()
        self.label_list_all = self.label_list_all[:, valid_classes]
        class_count_after = np.sum(self.label_list_all, axis=0)
        schema_names_zh = load_logic_set_schema_names()
        if schema_names_zh and max(self.valid_classes, default=-1) < len(schema_names_zh):
            self.label_names_zh = [schema_names_zh[idx] for idx in self.valid_classes]
            self.label_names = [
                LOGICSET_NAME_TRANSLATIONS.get(name, name)
                for name in self.label_names_zh
            ]
        else:
            self.label_names_zh = [f"cls_{idx}" for idx in range(self.label_list_all.shape[1])]
            self.label_names = self.label_names_zh.copy()
        if mode == 'train':
            print(valid_classes)
            print("Class distribution before filtering:", class_count)
            print(f'Un-valid classes: {np.where(class_count <= 15)[0]}')
            print("Class distribution after filtering:", class_count_after)
            print("Filtered logic-set labels:", self.label_names)

        # 使用列表推导式过滤 img_list 和 text_list
        # remove_filenames = [
        #     'WANG XIAO MAO_202105141105_Batch 2_1_10',
        #     'WANG LI HONG_201906302030_Batch 2_1_13',
        #     'LU HUI DONG_201810121932_Batch 2_1_24',
        #     'HUANG ZU DE_202009281854_Batch 2_1_15',
        #     'WANG XIAO MAO_202105141105_Batch 1_1_10',
        #     'YAO WEN JUAN_201901061707_Batch 2_1_100']
        remove_filenames =[
            'JIANG HONG SEN_202301071529_S40560_Bone, iDose (2)_1_50', 

            'YAO WEN JUAN_201901061707_Batch 1_1_100', # diff modality
            ]
        with open('/home/yizhenyu/data/ICP/data/huashan/ICP_new/chest_list.txt', 'r') as f:
            chest_list = [line.strip() for line in f]
        with open('/home/yizhenyu/data/ICP/data/huashan/ICP_new/CTA_list.txt', 'r') as f:
            CTA_list = [line.strip() for line in f]
        remove_filenames = remove_filenames + chest_list + CTA_list
        
        # 去重：根据img文件名
        # filename = img.split('/')[-1]
        # num = ''.join(c for c in filename if c.isdigit())
        # idx = filename.split(num[0])[0].replace('_',' ').strip()
        # period_icp = filename.split('_')[-2] + filename.split('_')[-1]
        # patient_id = idx + period_icp
        # 逻辑如上述
        seen_patient_ids = set()
        unique_img_text_label = []
        for img, text, label in zip(self.img_list_all, self.text_list_all, self.label_list_all):
            filename = img.split('/')[-1]
            num = ''.join(c for c in filename if c.isdigit())
            idx = filename.split(num[0])[0].replace('_',' ').strip()
            period_icp = filename.split('_')[-2] + filename.split('_')[-1]
            patient_id = idx + period_icp
            if patient_id not in seen_patient_ids:
                seen_patient_ids.add(patient_id)
                unique_img_text_label.append((img, text, label))
        # self.img_list_all, self.text_list_all, self.label_list_all = zip(*unique_img_text_label) if unique_img_text_label else ([], [], [])

        filtered_img_text_label = [
            # (img, text, label) for img, text, label in zip(self.img_list_all, self.text_list_all, self.label_list_all)
            (img, text, label) for img, text, label in unique_img_text_label
            if not any(filename in img for filename in remove_filenames)  
            and int(img.split('_')[-1].split('.')[0]) >= args.min_value
            and int(img.split('_')[-1].split('.')[0]) <= args.max_value
            and (int(img.split('_')[-2]) == 1 or int(img.split('_')[-2]) == 2)
        ]
    
        # filtered_img_text = filtered_img_text
        
        print(f'### Filtered {len(self.img_list_all) - len(filtered_img_text_label)} samples')
        self.img_list_all, self.text_list_all, self.label_list_all = zip(*filtered_img_text_label) if filtered_img_text_label else ([], [], [])
        self.img_list_all = list(self.img_list_all)
        self.text_list_all = list(self.text_list_all)
        self.label_list_all = list(self.label_list_all)


        if debug:
            print('Debug mode: using only 100 samples')
            self.img_list_all = self.img_list_all[:100]
            self.text_list_all = self.text_list_all[:100]
            self.label_list_all = self.label_list_all[:100]

        if not fold_split:
            self.img_list = self.img_list_all
            self.text_list = self.text_list_all
            self.label_list = self.label_list_all
        else:
            np.random.seed(0)
            random.seed(0)
            # shuffle but keep the same patient together, patients_name: path.split('/')[-1].split('_')[0]
            patient_dict = {}
            for idx, path in enumerate(self.img_list_all):
                patient_name = path.split('/')[-1].split('_')[0]
                if patient_name not in patient_dict:
                    patient_dict[patient_name] = []
                patient_dict[patient_name].append((path, self.text_list_all[idx], self.label_list_all[idx]))
            patient_names = list(patient_dict.keys())
            random.shuffle(patient_names)
            shuffled_img_text = []
            for name in patient_names:
                shuffled_img_text.extend(patient_dict[name])
            self.img_list_all, self.text_list_all, self.label_list_all = zip(*shuffled_img_text)
            self.img_list_all = list(self.img_list_all)
            self.text_list_all = list(self.text_list_all)
            self.label_list_all = list(self.label_list_all)
            print(f'### Fold {fold_idx} / {fold_num}')
            ## K-fold cross val
            fold_len = len(self.img_list_all)//fold_num
            id_start = fold_idx*fold_len
            id_stop = (fold_idx+1)*fold_len
            name_id_start = self.img_list_all[id_start].split('/')[-1].split('_')[0]
            name_id_stop = self.img_list_all[id_stop].split('/')[-1].split('_')[0]
            for id in range(id_start-1,-1,-1):
                name_id = self.img_list_all[id].split('/')[-1].split('_')[0]
                if name_id == name_id_start:
                    continue
                else:
                    id_start = id + 1
                    break
            for id in range(id_stop-1,-1,-1):
                name_id = self.img_list_all[id].split('/')[-1].split('_')[0]
                if name_id == name_id_stop:
                    continue
                else:
                    id_stop = id
                    break
            if fold_idx == fold_num-1: 
                id_stop = len(self.img_list_all) + 1

            if mode == 'train':
                self.img_list = self.img_list_all[:id_start] + self.img_list_all[id_stop+1:]
                self.text_list = self.text_list_all[:id_start] + self.text_list_all[id_stop+1:]
                self.label_list = self.label_list_all[:id_start] + self.label_list_all[id_stop+1:]
            else:
                self.img_list = self.img_list_all[id_start:id_stop+1]
                self.text_list = self.text_list_all[id_start:id_stop+1]
                self.label_list = self.label_list_all[id_start:id_stop+1]
                

        assert len(self.img_list) == len(self.text_list) == len(self.label_list)
        if log_dir is not None:
            with open(os.path.join(log_dir,f'{mode}_list.txt'), 'w') as f:
                for item in self.img_list:
                    f.write(item+'\n')

        if mode == 'val' and shuffle_val:
            '''shuffle the val dataset once time at beginning'''
            datas = list(zip(self.img_list,self.text_list, self.label_list))
            random.shuffle(datas)
            self.img_list, self.text_list, self.label_list = zip(*datas)

        self.img_list = list(self.img_list)
        self.text_list = list(self.text_list)
        self.label_list = list(self.label_list)

        stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}
        Transformer = transforms_before.Transformer(transformer, stats)
        self.augmentation = augmentation
        self.raw_transform = Transformer.raw_transform()

        self.class_thresholds = args.cls_thresholds
        self.transforms = transforms
        self.args = args

        self.targets = [int(img_path.split('_')[-1].split('.')[0]) for img_path in self.img_list]
        self.targets = [np.clip(icp_value,self.args.min_value,self.args.max_value) for icp_value in self.targets]

        # # shuffle the targets
        # print('before shuffle:', self.targets[:5])
        # random.shuffle(self.targets)
        # print('after shuffle:', self.targets[:5])

        self.using_LDS = using_LDS
        if using_LDS:
            self.weights = prepare_weights(self.targets,max_target=self.args.max_value+1)

        self.img_list_ori, self.text_list_ori = self.img_list.copy(), self.text_list.copy()
        self.targets_ori = self.targets.copy()
        self.labels_bin = [categorize(icp_value, self.class_thresholds) for icp_value in self.targets]
        self.labels_bin_ori = np.array(self.labels_bin).copy()
        print(f'Class distribution: {np.unique(self.labels_bin, return_counts=True)}')
        if self.using_LDS:
            self.weights_ori = np.array(self.weights, dtype=np.float32)


    def __getitem__(self, index):
        data_path = self.img_list[index]
        icp_value = self.targets[index]
        if self.using_LDS:
            weight = self.weights[index]
        label_bin = self.labels_bin[index]
        label_multi = self.label_list[index]
        label_onehot = np.eye(len(self.class_thresholds)+1)[label_bin]
        # target = int(icp_value >= 12)
        # target_onehot = np.eye(2)[target]

        #icp_value: [0,max]
        # icp_value = (icp_value - self.args.min_value) / (self.args.max_value - self.args.min_value) #[0,1]
        # icp_value = icp_value * self.args.scaler - self.args.offset

        ## Large-Scale-medical
        # if data_path.endswith('.h5'):
        #     file = h5py.File(data_path, 'r')
        #     data = file['img'][()].transpose(2, 1, 0) #[z,y,x]
        # ## fixed slice num
        # # data = data[10:21]
        # npy_normalized = self.transforms(self.augmentation, data, self.args)
        # npy_normalized = npy_normalized[np.newaxis].repeat(3,axis=0)

        ## Biomediclip
        if data_path.endswith('.h5'):
            file = h5py.File(data_path, 'r')
            data = file['img'][()].transpose(2, 1, 0) #[z,x,y] #[0-255]
        else:
            print('not .h5', data_path)
        npy_normalized = self.raw_transform(data)  #[-1,1]
        text_all = self.text_list[index].split('###')[-4]
        text_basic = self.text_list[index].split('###')[-2]
        text_CT = self.text_list[index].split('###')[-3]
        
        # text_categorize = categorize_text(target)
        if self.using_LDS:
            return {
                'image': npy_normalized,
                'label': label_bin,
                'label_multi': torch.tensor(label_multi),
                'icp_value_norm': icp_value,
                'label_onehot': label_onehot,
                'img_path': data_path,
                'text_ct_raw': text_CT,
                'text_basic_raw': text_basic,
                'text_all_raw': text_all,
                'lds_weights': weight # Label distribution smoothing 
                # 'text_categorize': text_categorize
            }
        else:
            return {
                'image': npy_normalized,
                'label': label_bin,
                'label_multi': label_multi,
                'icp_value_norm': icp_value,
                'label_onehot': label_onehot,
                'img_path': data_path,
                'text_ct_raw': text_CT,
                'text_basic_raw': text_basic,
                'text_all_raw': text_all,
                # 'text_categorize': text_categorize
            }

    def __len__(self):
        return len(self.img_list)
    
    # def balance_samples(self):
    #     """
    #     Balance the samples in the dataset.
    #     when the threshold lens is 1
    #     """
    #     assert len(self.class_thresholds) == 1
    #     self.labels = np.array(self.targets_ori) > self.class_thresholds[0]
    #     self.labels = self.labels.astype(np.float32)
    #     self.img_list = np.array(self.img_list_ori)
    #     self.text_list = np.array(self.text_list_ori)

    #     pos_idx = np.where(self.labels == 1)[0]
    #     neg_idx = np.where(self.labels == 0)[0]
    #     if len(neg_idx) > len(pos_idx):
    #         neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
    #     else:
    #         pos_idx = np.random.choice(pos_idx, size=len(neg_idx), replace=False)

    #     idxs = np.concatenate([pos_idx, neg_idx])
    #     self.img_list = self.img_list[idxs]
    #     self.text_list = self.text_list[idxs]
    #     self.labels = self.labels[idxs]

    #     self.img_list = list(self.img_list)
    #     self.text_list = list(self.text_list)
    #     # self.labels = list(self.labels)
    #     if self.using_LDS:
    #         self.weights = self.weights_ori[idxs].tolist()
    #     print(f'Balance samples: {len(self.img_list)}')
   
    def balance_samples(self):
        """
        Balance the samples in the dataset, select the minimum class number of samples from each class.
        """
        cls_num = len(self.class_thresholds) + 1
        assert cls_num > 1
        mini_num = min(np.unique(self.labels_ori, return_counts=True)[1])

        idxs = []
        for cls in range(cls_num):
            cls_idx = np.where(self.labels_ori == cls)[0]
            if len(cls_idx) > mini_num:
                cls_idx = np.random.choice(cls_idx, size=mini_num, replace=False)
            idxs.extend(cls_idx)
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        self.img_list = np.array(self.img_list_ori)[idxs]
        self.text_list = np.array(self.text_list_ori)[idxs]
        self.labels = self.labels_ori[idxs]

        if self.using_LDS:
            self.weights = self.weights_ori[idxs]
        print(f'TrainingSet Balance samples: {len(self.img_list)}')
   
