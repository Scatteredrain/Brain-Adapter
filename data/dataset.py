import os
import nibabel
import numpy as np
from torch.utils.data import Dataset
# from utils.img_util import apply_wl_ww_and_norm
import sys
sys.path.append('..')
from utils.img_util import apply_wl_ww_and_norm
from .augment import transforms
from scipy.ndimage import zoom
from tqdm import tqdm
import random

def resample_image(image, target_shape):
    # Calculate the zoom factors
    zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
    
    # Resample the image using scipy's zoom function
    resampled_image = zoom(image, zoom_factors, order=3)  # order=3 for cubic interpolation
    return resampled_image

class CTDataset(Dataset):
    def __init__(self, mode, img_file_path, text_file_path, transformer, WL_WW=(40, 90),max_value=50):
        self.WL_WW = WL_WW
        # print(img_file_path)
        with open(img_file_path, 'r') as f:
            self.img_list = [line.strip() for line in f]
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r') as f:
                self.text_list = [line.strip() for line in f]
        else:
            self.text_list = ['Abnormal brain CT scan' for _ in range(len(self.img_list))]
        if mode == 'val':
            datas = list(zip(self.img_list,self.text_list))
            random.shuffle(datas)
            self.img_list, self.text_list = zip(*datas)

        self.imgs = []
        self.labels = []
        # stats = {'pmin': None, 'pmax': None, 'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
        stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}
        transformer = transforms.Transformer(transformer, stats)
        self.raw_transform = transformer.raw_transform()
        self.mode = mode
  
        for idx in tqdm(range(len(self.img_list)),position=0):
            img_name = self.img_list[idx]
            label = int(self.img_list[idx].split('_')[-1].split('.')[0])
            if label >= max_value:
                label = 1
            else:
                label = label / max_value
            if label < 0:
                label = 0
            # label = label * 2 - 1 # [-1, 1]
            self.labels.append(label)
            assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            assert img is not None
            img = self.__preprocess_data__(img)
            self.imgs.append(img.transpose(2, 1, 0))
            # tqdm.write("Loading {} with shape {}".format(img_name, self.imgs[-1].shape))
            
    def __len__(self):
        return len(self.imgs)
    
    def __preprocess_data__(self, data): 
        data = data.get_fdata()#[...,10:25]
        data = apply_wl_ww_and_norm(data, self.WL_WW, True) #grayvalue
        data = resample_image(data, (data.shape[0], data.shape[1], 38))
        return data
    
    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        text_all = self.text_list[idx].split('###')[-4]
        text = self.text_list[idx].split('###')[-2]
        text_ct = self.text_list[idx].split('###')[-3]
        label = self.labels[idx]

        img_array = self.raw_transform(img)
        # img_array = self.__nii2tensorarray__(img_array)

        return img_array, text, text_ct, text_all, label