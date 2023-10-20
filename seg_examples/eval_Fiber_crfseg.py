from typing import List
import time
import datetime
import random
import sys
import os
import argparse

import numpy as np
import h5py
from torchviz import make_dot

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchinfo import summary

from segmentationCRF import metrics
from segmentationCRF.models import UNet
from segmentationCRF.data_utils import get_datset, get_default_transforms
from segmentationCRF.utils import check_make_dir
from segmentationCRF import test
from segmentationCRF.crfseg import CRF

"""
example command to run:
python seg_examples/eval_Fiber_crfseg.py -d /global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter -r /global/cfs/cdirs/m636/geshi/exp/Fiber/crf/CrossEntropy/1_seed_9999/iter512-10-16-2023-18:50:36.pt -a unet-crf -s 9999 -g 1 --benchmark --verbose
"""

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--data', '-d', type=str, required=True,
                    help='data folder to load data')
parser.add_argument('--resume', '-r', type=str, required=True, 
                    help='resume from checkpoint')
parser.add_argument('--architecture', '-a', type=str, default='unet',
                    help='model architecture')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='which GPU to use, only when disable-cuda not specified')
parser.add_argument('--data_parallel', '-dp', action='store_true',
                    help='using data parallel')
parser.add_argument('--benchmark', action='store_true',
                    help='using benchmark algorithms')
parser.add_argument('--debug', action='store_true',
                    help='using debug mode')
parser.add_argument('--verbose', action='store_true',
                    help='verbose mode')

# load and parse argument
args = parser.parse_args()

model_path = args.resume

if args.gpu<0 or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    if args.gpu<torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cuda") 
print('Using device: {}'.format(device))

# set up the seed
if args.seed:
    seed = args.seed
else:
    seed = torch.seed()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# set up benchmark running
if args.benchmark:
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.benchmark = False
    
if args.debug:
    torch.autograd.set_detect_anomaly(True)
else:
    torch.autograd.set_detect_anomaly(False)

val_images = os.path.join(args.data, "val/img/")
val_annotations = os.path.join(args.data, "val/ann/")
batch_size=32
classes = ('background', 'foreground')
n_classes = len(classes)
n_workers = 0
input_height = 288
input_width = 288
output_height = 288
output_width = 288
read_image_type=1
ignore_segs = False
    
downward_params = {
    'in_channels': 3, 
    'emb_sizes': [32, 64, 128, 256, 512], 
    'out_channels': [32, 64, 128, 256, 512],
    'kernel_sizes': [3, 3, 3 ,3 ,3], 
    'paddings': [1, 1, 1, 1, 1], 
    'batch_norm_first': False,
}
upward_params = {
    'in_channels': [512, 1024, 512, 256, 128],
    'emb_sizes': [1024, 512, 256, 128, 64], 
    'out_channels': [512, 256, 128, 64, 32],
    'kernel_sizes': [3, 3, 3, 3, 3], 
    'paddings': [1, 1, 1, 1, 1], 
    'batch_norm_first': False, 
    'bilinear': True,
}
output_params = {
    'in_channels': 64,
    'n_classes': n_classes,
}

x = torch.rand(batch_size, 3, input_height, input_width)

if args.architecture == 'unet':
    model = UNet(downward_params, upward_params, output_params)
elif args.architecture == 'unet-crf':
    unet = UNet(downward_params, upward_params, output_params)
    model = nn.Sequential(
        unet,
        CRF(n_spatial_dims=2)
    )
out = model(x)
print('output shape', out.shape) 

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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

if args.data_parallel:
    model= nn.DataParallel(model)
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

res = test_Fiber(model, dataloader, classes, device)
print(res)
