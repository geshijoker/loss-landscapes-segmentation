# Header files
from __future__ import print_function, absolute_import, division

import os
import shutil
import sys
import json
import time
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

EPS = 1e-7

def check_make_dir(path):

    # You should change 'test' to your preferred folder.
    mydir = os.path.join('./', path)
    check_folder = os.path.isdir(mydir)

    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(mydir)
        print("created folder : ", mydir)
    else:
        print(mydir, "folder already exists.")
    return mydir

def clean_directory(path):
    for root, dirs, files in os.walk(path):
        # Remove .DS_Store files
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")

        # Remove .ipynb_checkpoints directories
        for dir_name in dirs:
            if dir_name == ".ipynb_checkpoints":
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")

        # Remove __pycache__ directories
        for dir_name in dirs:
            if dir_name == "__pycache__":
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
                
# Pascal VOC color palette for labels
_PALETTE = [0, 0, 0,
            128, 0, 0,
            0, 128, 0,
            128, 128, 0,
            0, 0, 128,
            128, 0, 128,
            0, 128, 128,
            128, 128, 128,
            64, 0, 0,
            192, 0, 0,
            64, 128, 0,
            192, 128, 0,
            64, 0, 128,
            192, 0, 128,
            64, 128, 128,
            192, 128, 128,
            0, 64, 0,
            128, 64, 0,
            0, 192, 0,
            128, 192, 0,
            0, 64, 128,
            128, 64, 128,
            0, 192, 128,
            128, 192, 128,
            64, 64, 0,
            192, 64, 0,
            64, 192, 0,
            192, 192, 0]

# _IMAGENET_MEANS = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # RGB mean values

# def get_preprocessed_image(file_name):
#     """
#     Reads an image from the disk, pre-processes it by subtracting mean etc. and
#     returns a numpy array that's ready to be fed into the PyTorch model.

#     Args:
#         file_name:  File to read the image from

#     Returns:
#         A tuple containing:

#         (preprocessed image, img_h, img_w, original width & height)
#     """

#     image = Image.open(file_name)
#     original_size = image.size
#     w, h = original_size
#     ratio = min(500.0 / w, 500.0 / h)
#     image = image.resize((int(w * ratio), int(h * ratio)), resample=Image.BILINEAR)
#     im = np.array(image).astype(np.float32)
#     assert im.ndim == 3, 'Only RGB images are supported.'
#     im = im[:, :, :3]
#     im = im - _IMAGENET_MEANS
#     im = im[:, :, ::-1]  # Convert to BGR
#     img_h, img_w, _ = im.shape

#     pad_h = 500 - img_h
#     pad_w = 500 - img_w
#     im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
#     return np.expand_dims(im.transpose([2, 0, 1]), 0), img_h, img_w, original_size


def get_label_image(probs, is_pred=False, original_size=None):
    """
    Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

    Args:
        probs: Numpy array, probability output of shape (num_labels, height, width)
        original_size: Original image size (width, height)

    Returns:
        Label image as a PIL Image
    """
    if is_pred:
        labels = probs.argmax(axis=0).astype('uint8')
    else:
        labels = probs
    label_im = Image.fromarray(labels, 'P')
    label_im.putpalette(_PALETTE)
    if original_size:
        label_im = label_im.resize(original_size)
    return label_im

# def get_label_images_from_tensor(probs, is_one_hot=False, original_size=None):
#     """
#     Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

#     Args:
#         probs:  Torch Tensor, probability output of shape (batch_size, num_labels, height, width)
#         original_size: Original image size (width, height)

#     Returns:
#         Label image as a PIL Image
#     """
#     if is_one_hot:
#         _, labels = torch.max(probs, 1)
#     else:
#         labels = probs
#     labels = labels.detach().cpu().numpy().astype(np.uint8)
#     label_imgs = []
#     for label in labels:
#         label_im = Image.fromarray(label, 'P')
#         label_im.putpalette(_PALETTE)
#         print(label_im.size)
#         if original_size:
#             label_im = label_im.resize(original_size)
#         label_im = np.expand_dims(label_im, axis=0)
#         label_imgs.append(label_im)
#     label_imgs = np.stack(label_imgs, axis=0)
#     return torch.from_numpy(label_imgs)

def get_label_images_from_tensor(probs, n_classes, is_one_hot=False):
    """
    Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

    Args:
        probs:  Torch Tensor, probability output of shape (batch_size, num_labels, height, width)
        original_size: Original image size (width, height)

    Returns:
        Label image as a PIL Image
    """
    if is_one_hot:
        _, labels = torch.max(probs, 1)
    else:
        labels = probs
    labels = F.one_hot(labels, num_classes=n_classes)
    img_palette = torch.tensor(_PALETTE[:n_classes*3]).reshape((n_classes, 3))
    label_imgs = einsum(labels, img_palette, "b h w c, c nc -> b nc h w")   
    return label_imgs

if __name__ == "__main__":
    target_directory = "../"  # Change this to the root directory where you want to start the cleanup
    clean_directory(target_directory)