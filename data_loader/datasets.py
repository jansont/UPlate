import os
from traceback import print_list
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_loader.data_reader import *

class Image_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.mask_paths = os.listdir(mask_dir)
        self.image_paths = os.listdir(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.img_dir + self.image_paths[int(idx)]
        mask = self.mask_dir + self.mask_paths[int(idx)]
        image,mask = Reader(img, mask, display = False)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        if len(mask.shape) == 2:
            mask = mask.reshape((1,)+mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        image, mask = image.transpose(0,2), mask.transpose(0,2)
        return image,mask