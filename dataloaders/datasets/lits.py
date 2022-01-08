from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import pandas as pd

class LiverSegmentation(Dataset):
    """
    LITS dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('lits_liver'),
                 split='train',
                 ):
        """
        :param base_dir: path to lits dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self.root = os.path.join(self._base_dir,'dataset_6')
        self.df = pd.read_csv(os.path.join(self._base_dir,"lits_df.csv"))
        if split != "val":
            self.df = self.df[self.df['liver_mask_empty'] == True]
        self.train_df = self.df[self.df['study_number']<111]
        self.test_df = self.df[self.df['study_number']>=111]
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

    def __len__(self):
        for split in self.split:
            if split == "train":
                return len(self.train_df)
            elif split == 'val' or split == 'vis':
                return len(self.test_df)
                
    def __getitem__(self, index):

        for split in self.split:
            if split == "train":
                imgpath = os.path.join(self.root, os.path.basename(self.train_df.iloc[index]['filepath']))
                _img = Image.open(imgpath).convert('RGB')
                maskpath = os.path.join(self.root, os.path.basename(self.train_df.iloc[index]['liver_maskpath']))
                _target = Image.open(maskpath).convert('L')
                sample = {'image': _img, 'label': _target}
                return self.transform_tr(sample)
            
            elif split == 'val' or split == 'vis':
                imgpath = os.path.join(self.root, os.path.basename(self.test_df.iloc[index]['filepath']))
                _img = Image.open(imgpath).convert('RGB')
                maskpath = os.path.join(self.root, os.path.basename(self.test_df.iloc[index]['liver_maskpath']))
                _target = Image.open(maskpath).convert('L')
                sample = {'image': _img, 'label': _target}
                return self.transform_val(sample)


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(30.0),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.FixedResize(size=self.args.base_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixedResize(size=self.args.base_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'LITS_liver(split=' + str(self.split) + ')'

class TumorSegmentation(Dataset):
    """
    LITS dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('lits_tumor'),
                 split='train',
                 ):
        """
        :param base_dir: path to lits dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self.root = os.path.join(self._base_dir,'dataset_6')
        self.df = pd.read_csv(os.path.join(self._base_dir,"lits_df.csv"))
        if split != "val":
            self.df = self.df[self.df['tumor_mask_empty'] == True]
        self.train_df = self.df[self.df['study_number']<111]
        self.test_df = self.df[self.df['study_number']>=111]
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

    def __len__(self):
        for split in self.split:
            if split == "train":
                return len(self.train_df)
            elif split == 'val' or split == 'vis':
                return len(self.test_df)
                
    def __getitem__(self, index):

        for split in self.split:
            if split == "train":
                imgpath = os.path.join(self.root, os.path.basename(self.train_df.iloc[index]['filepath']))
                _img = Image.open(imgpath).convert('RGB')
                maskpath = os.path.join(self.root, os.path.basename(self.train_df.iloc[index]['tumor_maskpath']))
                _target = Image.open(maskpath).convert('L')
                sample = {'image': _img, 'label': _target}
                return self.transform_tr(sample)
            
            elif split == 'val' or split == 'vis':
                imgpath = os.path.join(self.root, os.path.basename(self.test_df.iloc[index]['filepath']))
                _img = Image.open(imgpath).convert('RGB')
                maskpath = os.path.join(self.root, os.path.basename(self.test_df.iloc[index]['tumor_maskpath']))
                _target = Image.open(maskpath).convert('L')
                sample = {'image': _img, 'label': _target}
                return self.transform_val(sample)


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(30.0),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.FixedResize(size=self.args.base_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixedResize(size=self.args.base_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'LITS_liver(split=' + str(self.split) + ')'