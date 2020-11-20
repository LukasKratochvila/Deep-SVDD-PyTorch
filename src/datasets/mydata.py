#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:55:30 2020

@author: kratochvila
"""
from PIL import Image
import os
import os.path
import json
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from torchvision.datasets.utils import check_integrity

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import torchvision.transforms as transforms
from torch.utils.data import Subset

class MYDATA_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2 # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(1, 5))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-1.3942, 6.5378),
                   (-1.9069, 18.5352),
                   (-2.6035, 29.8313),
                   (-2.3845, 12.4581)]
        
        #For 1225 Industry biscuits base
        #target_transform=transforms.Lambda(lambda x: x)
        #test=list(dataset.test_set)
        #train=list(dataset.train_set)
        #s=[x[0] for x in train if x[1]==1]
        #n=[s.append(x[0]) for x in test if x[1]==1]
        #ma=max([torch.max(x) for x in s])
        #mi=min([torch.min(x) for x in s])

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.Resize((32,32)), #160,120
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                              [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyData(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyData(root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)
        
class MyData(data.Dataset):
    """`MyData Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``mydata-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = '300-batches-py'
    train_list = None#[
    #     ['train_batch', 'f3f5c2da070691006107cc91e1d62b20'],
    # ]

    test_list = None#[
        #['test_a_batch', 'dd03ab9be0ed8e9e7c0d9156ea132d63'],
    #     ['test_b_batch', '551dc5bf4b9ac26bafb99766439aed2a']
    # ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            # self.train_data = []
            # self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                # self.train_data.append(entry['data'])
                # if 'labels' in entry:
                #     self.train_labels.append(entry['labels'])
                # else:
                #     self.train_labels.append(entry['fine_labels'])
                fo.close()
                self.train_data = entry['data']
                self.train_labels = entry['labels']
            # self.train_data = np.concatenate(self.train_data)
            # self.train_data = self.train_data.reshape((100, 640, 480, 3))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        if self.train_list == None:
            assert os.path.exists(os.path.join(root, self.base_folder, 'hashes.json')), 'The hashes.json doesnt exist'
            with open(os.path.join(root, self.base_folder, 'hashes.json')) as hashes_buffer:    
                hashes = json.loads(hashes_buffer.read())
            keys = list(hashes.keys())
            self.train_list = list()
            self.test_list = list()
            for key in keys:
                if 'train' in key:
                    self.train_list.append(list((key,hashes[key])))
                elif 'test' in key:
                    self.test_list.append(list((key,hashes[key])))

        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True