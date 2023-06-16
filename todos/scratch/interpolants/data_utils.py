import pathlib
import os
import pickle
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch
import pickle

## from imagenet downsample ####

from PIL import Image
import os
import os.path
import numpy as np
import sys


# if sys.version_info[0] == 2:
#     import cPickle as pickle
# else:

from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils
import torch.utils.data as data
from torchvision import transforms as T

################################



# from diffusion transformer
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/
    8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])




def get_transform(config):


    Crop = lambda size : T.Lambda(lambda pil_image: center_crop_arr(pil_image, size))
    Flip = T.RandomHorizontalFlip()
    Tens = T.ToTensor()
    Norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    Compose = T.Compose

    if config.dataset in ['mnist','cifar']:
        return Compose([Flip, Tens])
    else:
        return Compose([Crop(config.H), Flip, Tens]) # Norm


def setup_data(trainer, ddp = False, rank = None):
    

    dset = trainer.config.dataset
    path = './data'
    imnet_path = './data/imagenet_1k_val_subset'
    imnet_path = './data/imagenet_tiny'
    flower_path = './data/flowers/jpg'
    config = trainer.config
    config.skip_fid = True


    MNIST = datasets.MNIST
    CIFAR = datasets.CIFAR10



    if dset == 'mnist':
        
        config.C, config.H, config.W = 1, 28, 28
        config.num_classes = 10
        ds = MNIST(path, train=True, download=True, transform=get_transform(config))
    
    elif dset == 'cifar':

        config.C, config.H, config.W = 3, 32, 32
        config.num_classes = 10
        ds = CIFAR(path, train=True, download=True, transform=get_transform(config))
        config.skip_fid = False

    elif dset == 'flowers':
        config.C, config.H, config.W = 3, 128, 128
        ds = ImageFolder(flower_path, transform=get_transform(config))

    elif dset == 'imagenet32':
        assert False
        config.C, config.H, config.W = 3, 32, 32
        ds = ImageNetDownSample(imnet_path, img_size = config.H, train=True)

    elif dset == 'imagenet64':
        assert False
        config.C, config.H, config.W = 3, 64, 64
        ds = ImageNetDownSample(imnet_path, img_size = config.H, train=True)

    elif dset == 'imagenet128':
        assert False
        config.C, config.H, config.W = 3, 128, 128
        ds = ImageFolder(imnet_path, transform = get_transform(config))

    elif dset == 'imagenet256':
        config.C, config.H, config.W = 3, 256, 256
        config.num_classes = 1000
        ds = ImageFolder(imnet_path, transform = get_transform(config))

    elif dset == 'afhq-2':
        assert False
        config.C, config.H, config.W = 3, 512, 512

    else:
        assert False


    if ddp:

        assert rank is not None

        train_sampler = DistributedSampler(
                        ds,
                        num_replicas=dist.get_world_size(),
                        rank=rank,
                        shuffle=True,
                        seed=config.global_seed
        )
        
        train_loader = DataLoader(
                ds,
                batch_size=int(config.bsz // dist.get_world_size()),
                shuffle=False,
                sampler=train_sampler,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=True
        )
    else:
    
        train_loader = DataLoader(
                ds,
                batch_size=config.bsz,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=True
        )



    trainer.train_loader = train_loader

    config.d = config.C * config.H * config.W 
    config.null_token = config.num_classes
  
    if config.use_vae:                                                                                                                            
        assert config.dataset == 'imagenet256'
        config.C_flow = 4
        config.H_flow = config.H // 8
        config.W_flow = config.W // 8
    else:
        config.C_flow = config.C
        config.H_flow = config.H
        config.W_flow = config.W


    print("done setting up data")
    print("set trainer.train_loader")
    print("not doing val/test loaders at the moment")

def setup_generic_ddp(trainer, rank):

    assert False, 'outdated'

    config = trainer.config

    trainer.train_sampler = DistributedSampler(
                    trainer.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=rank,
                    shuffle=True,
                    seed=config.global_seed
    )
    trainer.test_sampler = DistributedSampler(
                    trainer.test_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=rank,
                    shuffle=True,
                    seed=config.global_seed
    )

    trainer.train_loader = DataLoader(
            trainer.train_dataset,
            batch_size=int(config.bsz // dist.get_world_size()),
            shuffle=False,
            sampler=trainer.train_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
    )

    trainer.test_loader = DataLoader(
            trainer.test_dataset,
            batch_size=int(config.bsz // dist.get_world_size()),
            shuffle=False,
            sampler=trainer.test_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
    )


def setup_generic(trainer):


    assert False, 'outdated'

    config = trainer.config

    trainer.train_loader = DataLoader(
            trainer.train_dataset,
            batch_size=config.bsz,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
    )

    trainer.test_loader = DataLoader(
            trainer.test_dataset,
            batch_size=config.bsz,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
    )


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


"""
Simple Dataset class that includes option for 
converting image format, adapted for lucidrains.
"""

class Dataset(Dataset):
    def __init__(
        self,
        folder, ### directory of images, say png 
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)




"""Create tools for Pytorch to load Downsampled ImageNet (32X32,64X64)
Thanks to the cifar.py provided by Pytorch.
Author: Xu Ma.
Date:   Apr/21/2019
Data Preparation:
    1. Download unsampled data from ImageNet website.
    2. Unzip file  to rootPath. eg: /home/xm0036/Datasets/ImageNet64(no train, val folders)
Remark:
This tool is able to automatic recognize downsampled size.
Use this tool like cifar10 in datsets/torchvision.
"""

class ImageNetDownSample(data.Dataset):
    """`DownsampleImageNet`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
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

    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, img_size, train=True,
                 transform=None, target_transform=None, augment_horizontal_flip = False):
        self.root = os.path.expanduser(root)
        # self.transform = transform
        self.transform = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else torch.nn.Identity(),
            T.CenterCrop(img_size),
            T.ToTensor()
        ])
        self.target_transform = None
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.train_labels[:] = [x - 1 for x in self.train_labels]

            self.train_data = np.concatenate(self.train_data)
            [picnum, pixel] = self.train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            [picnum,pixel]= self.test_data.shape
            pixel = int(np.sqrt(pixel/3))

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.test_labels[:] = [x - 1 for x in self.test_labels]
            self.test_data = self.test_data.reshape((picnum, 3, pixel, pixel))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

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
        # print("IMAGE SHAPE:", img.shape)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
