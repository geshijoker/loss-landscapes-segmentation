# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_u_like, rand_n_like, orthogonal_to
from loss_landscapes.contrib.functions import SimpleWarmupCaller, SimpleLossEvalCaller, log_refined_loss

from pyhessian.utils import *
from pyhessian import hessian, get_esd_plot, density_generate # ESD plot

from segmentationCRF import metrics
from segmentationCRF.models import EncoderDecoder
from segmentationCRF.data_utils import get_datset, get_default_transforms
from segmentationCRF.utils import check_make_dir, get_label_image, get_label_images_from_tensor
from segmentationCRF import test
from segmentationCRF.crfseg import CRF

from typing import List
import time
import datetime
import random
import sys
import os
import copy
import argparse

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils

"""
example command to run:
python /global/u2/g/geshi/loss-landscapes-segmentation/loss_examples/Oxford_ed_hessian.py -d /global/cfs/cdirs/m636/geshi/data/ -r /global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_243/0_lr_0001_seed_243/iter30-03-01-2024-01:52:17.pt -sa /global/u2/g/geshi/loss-landscapes-segmentation/loss_examples/ed-pyhessian-loss-landscapes -s 243 -g 0 -ad 5 -aw 16 -ip 224 -bs 32 --benchmark --debug --verbose
"""

parser = argparse.ArgumentParser(description='Model testing')
parser.add_argument('--data', '-d', type=str, required=True,
                    help='data folder to load data')
parser.add_argument('--resume', '-r', type=str, required=True, 
                    help='resume from checkpoint of ed')
parser.add_argument('--save', '-sa', type=str, required=True, 
                    help='the path to save loss landscapes')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='which GPU to use, only when disable-cuda not specified')
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

# contour plot resolution
STEPS = 31
top_n = 2
NORM = 'layer'
DIST = 0.01

# load and parse argument
args = parser.parse_args()

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
model_path = args.resume
save_path = args.save
batch_size=args.batch_size
arc_width = args.arc_width
arc_depth = args.arc_depth
input_size = args.input_size

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
    'in_channels': [16, 8, 4, 2, 1],
    'emb_sizes': [16, 8, 4, 2, 1], 
    'out_channels': [8, 4, 2, 1, 1],
    'kernel_sizes': [3, 3, 3, 3, 3], 
    'paddings': [1, 1, 1, 1, 1], 
    'batch_norm_first': False, 
    'bilinear': True,
}
output_params = {
    'in_channels': 1,
    'n_classes': n_classes,
}

downward_params['emb_sizes'] = [downward_params['emb_sizes'][i]*arc_width for i in range(min(len(downward_params['emb_sizes']), arc_depth))]
downward_params['out_channels'] = [downward_params['out_channels'][i]*arc_width for i in range(min(len(downward_params['out_channels']), arc_depth))]
upward_params['in_channels'] = [upward_params['in_channels'][i]*arc_width for i in range(min(len(upward_params['in_channels']), arc_depth))]
upward_params['emb_sizes'] = [upward_params['emb_sizes'][i]*arc_width for i in range(min(len(upward_params['emb_sizes']), arc_depth))]
upward_params['out_channels'] = [upward_params['out_channels'][i]*arc_width for i in range(min(len(upward_params['out_channels']), arc_depth))]
output_params['in_channels'] = output_params['in_channels']*arc_width

x = torch.rand(1, 3, input_size, input_size)
model = EncoderDecoder(downward_params, upward_params, output_params)
out = model(x)
print('output shape', out.shape)

dataset_parameters = {
    'data_path': data_path,
    'split': 'trainval',
    'data_transform': data_transform,
    'target_transform': target_transform,
    'download': True,
}

val_dataset_parameters = {
    'data_path': data_path,
    'split': 'test',
    'data_transform': data_transform,
    'target_transform': target_transform,
    'download': True,
}

dataset = get_datset('oxford', dataset_parameters)
val_dataset = get_datset('oxford', val_dataset_parameters)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

model = model.to(device)
model.eval()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
# test(model, val_dataloader, n_classes, device)

criterion = torch.nn.CrossEntropyLoss() # DiceLoss(True), IOULoss(softmax=True)
x, y = iter(dataloader).__next__()
metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))

##### encoderdecoder
model_final = copy.deepcopy(model)

hessian_comp = hessian(model_final,
                       criterion,
                       data=(x.to(device), y.to(device)),
                       device=device)
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

dir_one = ModelParameters(top_eigenvector[0])
dir_two = ModelParameters(top_eigenvector[1])

pll = loss_landscapes.PlanarLossLandscape(model_final, STEPS, deepcopy_model=True)
pll.precomputed(dir_one, dir_two, distance=DIST, normalization=NORM, centered=True)
pll.stats_initializer()

# single batch loss landscape
since = time.time()
pll.warm_up(metric)
print('warmup time cost ', time.time()-since)

since = time.time()
loss_data = pll.compute(metric)
print('compute time cost ', time.time()-since)

model_setting = model_path.split('/')[-2]
epoch_number = model_path.split('/')[-1].split('-')[0]
save_name ='ed_'+model_setting+'_'+epoch_number+'.npy'

with open(os.path.join(save_path, save_name), 'wb') as f:
    np.save(f, loss_data)