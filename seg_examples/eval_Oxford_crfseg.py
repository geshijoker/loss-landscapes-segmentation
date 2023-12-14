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
import torchvision
from torchvision import transforms, utils
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset


from segmentationCRF import metrics
from segmentationCRF.models import UNet
from segmentationCRF.data_utils import get_datset, get_default_transforms
from segmentationCRF.utils import check_make_dir, get_label_image, get_label_images_from_tensor
from segmentationCRF import test
from segmentationCRF.crfseg import CRF

"""
example command to run:
python seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r /global/cfs/cdirs/m636/geshi/exp/Oxford/crf-late/CrossEntropy/0_seed_9999/iter10-11-18-2023-00:08:14.pt -a unet-crf -s 9999 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose

python seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r /global/cfs/cdirs/m636/geshi/exp/Oxford/non-crf/CrossEntropy/0_seed_9999/iter30-11-17-2023-18:04:01.pt -a unet -s 9999 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose
"""

parser = argparse.ArgumentParser(description='Model testing')
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
log_path = os.path.dirname(model_path)
model_name = os.path.basename(model_path)

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
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# set up benchmark running
if args.benchmark:
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.benchmark = False
    
if args.debug:
    torch.autograd.set_detect_anomaly(True)
else:
    torch.autograd.set_detect_anomaly(False)

data_path = args.data
batch_size=args.batch_size
arc_width = args.arc_width
arc_depth = args.arc_depth
input_size = args.input_size
percentage = args.percentage
assert percentage>0 and percentage<=1, "The percentage is out of range of (0, 1)"

n_workers = 0
classes = ('foreground', 'background', 'border')
n_classes = len(classes)
    
data_transform, target_transform = get_default_transforms('oxford', input_size, n_classes, noise_level=0)

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

x = torch.rand(batch_size, 3, input_size, input_size)

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

dataset_parameters = {
    'data_path': data_path,
    'split': 'test',
    'data_transform': data_transform,
    'target_transform': target_transform,
    'download': True,
}

dataset = get_datset('oxford', dataset_parameters)
num = int(round(len(dataset)*percentage))
selected = list(range(num))
dataset = Subset(dataset, selected)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

if args.data_parallel:
    model= nn.DataParallel(model)
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

cl_wise_iou, test_stats = test(model, dataloader, n_classes, device)

dataiter = iter(dataloader)
images, labels = next(dataiter)
# create grid of images
img_labels = get_label_images_from_tensor(labels, n_classes, is_one_hot=True)
label_grid = torchvision.utils.make_grid(img_labels)

with torch.no_grad():
    preds = model(images.to(device)).detach().cpu()
    img_preds = get_label_images_from_tensor(preds, n_classes, is_one_hot=True)
    preds_grid = torchvision.utils.make_grid(img_preds)

# write to tensorboard
with SummaryWriter(log_path) as w:
    w.add_hparams({'name':model_name, 'condition': 'normal'}, test_stats)
    w.add_image('gtrue labels', label_grid)
    w.add_image('normal preds labels', preds_grid)

print(test_stats)