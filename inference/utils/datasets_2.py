import logging
# from PIL import Image
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
from pathlib import Path
import xarray as xr
import rioxarray as rioxr

class DisasterDataset(Dataset):
    def __init__(self, data_dir, img_suffix_list, transform:bool, normalize:bool):
        
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        self.img_suffix_list = img_suffix_list
    
    def __process_image__(self, img):

        img_mean = img.mean(dim=["x", "y"]).values.tolist()
        img_std = img.std(dim=["x", "y"]).values.tolist()

        if self.normalize:
            norm_func = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std)
            ])
        else:
            norm_func = transforms.ToTensor()

        img_arr = np.transpose(np.array(img), (1, 2, 0)).astype(dtype='float64')
        img_norm = norm_func(img_arr)

        return img_norm
    
    def __getitem__(self, i):
        
        img_dir = Path(self.data_dir).joinpath("images")

        img_suffix = self.img_suffix_list[i]

        pre_img = xr.open_rasterio(img_dir.joinpath(f"{img_suffix}_pre_disaster.tif")) / 255.0
        post_img = xr.open_rasterio(img_dir.joinpath(f"{img_suffix}_post_disaster.tif")) / 255.0

        pre_img = self.__process_image__(pre_img)
        post_img = self.__process_image__(post_img)

        return {'pre_image': pre_img, 'post_image': post_img}
    
    
