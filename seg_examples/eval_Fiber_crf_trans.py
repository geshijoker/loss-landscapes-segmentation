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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from segmentationCRF import metrics
from segmentationCRF.models import UNet
from segmentationCRF.data_utils import get_datset, get_default_transforms
from segmentationCRF.utils import check_make_dir
from segmentationCRF import test
from segmentationCRF.crfseg import CRF

"""
example command to run:
python seg_examples/eval_Fiber_crf_trans.py -d /global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter -rc /global/cfs/cdirs/m636/geshi/exp/Fiber/crf/CrossEntropy/2_seed_9999/iter19-10-26-2023-21:51:01.pt -rn /global/cfs/cdirs/m636/geshi/exp/Fiber/non-crf/CrossEntropy/2_seed_9999/iter19-10-27-2023-00:12:03.pt -s 9999 -g 0 -p 1 -ad 5 -aw 32 -ip 288 -bs 128 --benchmark --verbose
"""

parser = argparse.ArgumentParser(description='Model testing')
parser.add_argument('--data', '-d', type=str, required=True,
                    help='data folder to load data')
parser.add_argument('--resume_crf', '-rc', type=str, required=True, 
                    help='resume from checkpoint of unet-crf')
parser.add_argument('--resume_noncrf', '-rn', type=str, required=True, 
                    help='resume from checkpoint of unet')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='which GPU to use, only when disable-cuda not specified')
parser.add_argument('--percentage', '-p', type=float, default=1.0, 
                    help='the percentage of data used for training')
parser.add_argument('--arc_depth', '-ad', type=int, default=5,
                    help='the depth of the model')
parser.add_argument('--arc_width', '-aw', type=int, default=32,
                    help='the width of the model')
parser.add_argument('--input_size', '-ip', type=int, default=288,
                    help='the size of input')
parser.add_argument('--batch_size', '-bs', type=int, default=32,
                    help='the batch size of the data')
parser.add_argument('--benchmark', action='store_true',
                    help='using benchmark algorithms')
parser.add_argument('--debug', action='store_true',
                    help='using debug mode')
parser.add_argument('--verbose', action='store_true',
                    help='verbose mode')

# load and parse argument
args = parser.parse_args()

unet_crf_path = args.resume_crf
unet_path = args.resume_noncrf
log_path = os.path.dirname(unet_path)
unet_name = os.path.basename(unet_path)

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
    
batch_size=args.batch_size
arc_width = args.arc_width
arc_depth = args.arc_depth
input_size = args.input_size
percentage = args.percentage
assert percentage>0 and percentage<=1, "The percentage is out of range of (0, 1)"

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

data_transform, target_transform = get_default_transforms('fiber', input_size)
    
downward_params = {
    'in_channels': 3, 
    'emb_sizes': [1, 2, 4, 8, 16], 
    'out_channels': [1, 2, 4, 8, 16],
    'kernel_sizes': [3, 3, 3 ,3 ,3], 
    'paddings': [1, 1, 1, 1, 1], 
    'batch_norm_first': False,
}
upward_params = {
    'in_channels': [16, 32, 16, 8, 4],
    'emb_sizes': [32, 16, 8, 4, 2], 
    'out_channels': [16, 8, 4, 2, 1],
    'kernel_sizes': [3, 3, 3, 3, 3], 
    'paddings': [1, 1, 1, 1, 1], 
    'batch_norm_first': False, 
    'bilinear': True,
}
output_params = {
    'in_channels': 2,
    'n_classes': n_classes,
}

downward_params['emb_sizes'] = [downward_params['emb_sizes'][i]*arc_width for i in range(min(len(downward_params['emb_sizes']), arc_depth))]
downward_params['out_channels'] = [downward_params['out_channels'][i]*arc_width for i in range(min(len(downward_params['out_channels']), arc_depth))]
upward_params['in_channels'] = [upward_params['in_channels'][i]*arc_width for i in range(min(len(upward_params['in_channels']), arc_depth))]
upward_params['emb_sizes'] = [upward_params['emb_sizes'][i]*arc_width for i in range(min(len(upward_params['emb_sizes']), arc_depth))]
upward_params['out_channels'] = [upward_params['out_channels'][i]*arc_width for i in range(min(len(upward_params['out_channels']), arc_depth))]
output_params['in_channels'] = output_params['in_channels']*arc_width

x = torch.rand(batch_size, 3, input_height, input_width)

unet = UNet(downward_params, upward_params, output_params)
unet_crf = nn.Sequential(
    UNet(downward_params, upward_params, output_params),
    CRF(n_spatial_dims=2)
)
out = unet(x)
print('unet output shape', out.shape) 
out = unet_crf(x)
print('unet-crf output shape', out.shape) 

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
num = int(round(len(dataset)*percentage))
selected = list(range(num))
dataset = Subset(dataset, selected)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

unet = unet.to(device)
unet_crf = unet_crf.to(device)

checkpoint = torch.load(unet_path, map_location=device)
unet.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(unet_crf_path, map_location=device)
unet_crf.load_state_dict(checkpoint['model_state_dict'])

trans_crf = nn.Sequential(
    unet,
    unet_crf[1]
)

cl_wise_iou, test_stats = test(unet, dataloader, n_classes, device)
print('unet: ', test_stats)
cl_wise_iou, test_stats = test(unet_crf, dataloader, n_classes, device)
print('unet crf: ', test_stats)

cl_wise_iou, test_stats = test(trans_crf, dataloader, n_classes, device)
with SummaryWriter(log_path) as w:
    w.add_hparams({'model_name': unet_name+'_crf_trans', 'bs': batch_size}, test_stats)
print('crf trans: ', test_stats)
