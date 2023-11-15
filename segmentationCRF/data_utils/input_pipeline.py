from typing import List
import time
import os
import sys
import random

import numpy as np
import cv2
from skimage import io
from einops import rearrange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import torch.nn.functional as F

from segmentationCRF.data_utils.fiber_data import * 

class ChannelFirst(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, seg_mask):
        out = rearrange(seg_mask, 'h w c -> c h w')

        return out

class OneHotTransform(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, seg_mask):
        out = F.one_hot(seg_mask, self.n_classes).float()
        out = rearrange(out, 'h w c -> c h w')

        return out
    
class Integerfy(nn.Module):
    def __init__(self):
        super(Integerfy, self).__init__()
        
    def forward(self, seg_mask):
        seg_mask = (seg_mask.squeeze()*255).type(torch.int64)-1
        return seg_mask

class LabelNoiseTransform(nn.Module):
    def __init__(self, num_classes, noise_level=0.0):
        super(LabelNoiseTransform, self).__init__()
        self.num_classes = num_classes
        self.noise_level = noise_level

    def forward(self, seg_mask):
        # Copy the original mask to avoid modifying it in place
        noisy_mask = seg_mask.clone()

        # Determine the number of pixels to add noise to
        num_pixels = int(self.noise_level * noisy_mask.numel())

        # Randomly select pixels to add noise to
        noisy_indices = torch.randint(0, noisy_mask.numel(), (num_pixels,))
        noisy_labels = torch.randint(0, self.num_classes, (num_pixels,))

        # Apply label noise
        noisy_mask.contiguous().view(-1)[noisy_indices] = noisy_labels

        return noisy_mask

class FiberSegDataset(Dataset):
    """Fiber Segmentation dataset."""

    def __init__(self, images_path, segs_path, n_classes, input_width, input_height, output_width, output_height, transform=None, target_transform=None, ordering="channels_last", read_image_type=cv2.IMREAD_COLOR, ignore_segs=False):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied on a image.
            target_transform (callable, optional): Optional transform to be applied on a segmentation mask.
        """
        if ordering=="channels_first":
            self.ordering = ImageOrdering.CHANNELS_FIRST
        elif ordering=="channels_last":
            self.ordering = ImageOrdering.CHANNELS_LAST 
        else:
            raise DataLoaderError("Ordering {0} is undefined".format(file_name, full_dir_entry))
        self.read_image_type = read_image_type
        self.ignore_segs = ignore_segs
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height
        self.n_classes = n_classes
        self.transform = transform
        self.target_transform = target_transform
        if not ignore_segs:
            self.img_list = get_pairs_from_paths(images_path, segs_path)
        else:
            self.img_list = get_image_list_from_path(images_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.ignore_segs:
            im = self.img_list[idx]
            seg = None 
        else:
            im, seg = self.img_list[idx]
            seg = cv2.imread(seg, 1)
            seg = get_segmentation_array(seg, self.n_classes, self.output_width, self.output_height, no_reshape=True)

        im = cv2.imread(im, self.read_image_type)
        im = get_image_array(im, self.input_width, self.input_height, ordering=self.ordering, read_image_type=self.read_image_type).copy()
        
        if self.transform:
            im = self.transform(im)
        if self.target_transform:
            seg = self.target_transform(seg)

        if not self.ignore_segs:
            return  (im, seg)
        else:
            return im

def get_datset(dataset_name, dataset_parameters):
    if dataset_name=='fiber':
        val_images = dataset_parameters['images']
        val_annotations = dataset_parameters['annotations']
        n_classes = dataset_parameters['n_classes']
        n_workers = dataset_parameters['n_workers']
        input_height = dataset_parameters['input_height']
        input_width = dataset_parameters['input_width']
        output_height = dataset_parameters['output_height']
        output_width = dataset_parameters['output_width']
        data_transform = dataset_parameters['data_transform']
        target_transform = dataset_parameters['target_transform']
        read_image_type = dataset_parameters['read_image_type']
        ignore_segs = dataset_parameters['ignore_segs']

        dataset = FiberSegDataset(val_images, val_annotations, n_classes, 
            input_height, input_width, output_height, output_width,
            transform=data_transform, target_transform=target_transform,
            read_image_type=read_image_type, ignore_segs=ignore_segs)
        return dataset
    elif dataset_name=='oxford':
        data_path = dataset_parameters['data_path']
        split = dataset_parameters['split']
        data_transform = dataset_parameters['data_transform']
        target_transform = dataset_parameters['target_transform']
        download = dataset_parameters['download']
        dataset = OxfordIIITPet(root=data_path, split=split, target_types='segmentation', transform=data_transform, target_transform=target_transform, download=download)
        return dataset
    else:
        raise ValueError("Dataset {} is not supported".format(dataset_name))


def get_default_transforms(dataset_name, size, n_classes, noise_level=0.0):
    if dataset_name=='fiber':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            ])
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            ])
    elif dataset_name=='oxford':
        data_mean = np.array((0.485, 0.456, 0.406))
        data_std = np.array((0.229, 0.224, 0.225))

        data_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Normalize(data_mean, data_std)
        ])

        target_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            Integerfy(),
            LabelNoiseTransform(n_classes, noise_level),
            OneHotTransform(n_classes),
        ])
    else:
        raise ValueError("Dataset {} is not supported".format(dataset_name))
    return data_transform, target_transform
        
def main():
    val_images = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/val/img/"
    val_annotations = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/val/ann/"
    batch_size=32
    n_workers = 0
    input_height = 288
    input_width = 288
    output_height = 288
    output_width = 288
    read_image_type=1
    ignore_segs = False
    classes = ('background', 'foreground')
    n_classes = len(classes)
    
    data_transform, target_transform = get_default_transforms('fiber')
    
    dataset_parameters = {
        'images': val_images,
        'annotations': val_annotations,
        'n_classes': n_classes,
        'n_workers': n_workers,
        'input_height': input_height,
        'input_width': input_width,
        'output_height': output_height,
        'output_width': output_width,
        'data_transform': data_transform,
        'target_transform': target_transform,
        'read_image_type': read_image_type,
        'ignore_segs': ignore_segs,
    }

    dataset = get_datset('fiber', dataset_parameters)

    print('number of data', len(dataset), 'image size', dataset[0][0].shape, 'segmentation mask size', dataset[0][1].shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            im, seg = data
            print('dataloader', 'img size', im.shape, 'seg size', seg.shape)
            break

if __name__ == "__main__":
    main()