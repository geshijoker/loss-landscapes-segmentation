# libraries
import os
import copy
import sys
import time
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import h5py

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm, trange

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_u_like, rand_n_like, orthogonal_to
from loss_landscapes.contrib.functions import SimpleWarmupCaller, SimpleLossEvalCaller, log_refined_loss

from pyhessian.utils import *
from pyhessian import hessian

from segmentationCRF.metrics import IOULoss, DiceLoss
from segmentationCRF.models import UNet
from segmentationCRF.data_utils import get_datset, get_default_transforms
from segmentationCRF.utils import check_make_dir
from segmentationCRF import test
from segmentationCRF.crfseg import CRF

# define, load and parse argument
parser = argparse.ArgumentParser(description='Do Checkpoints X Losses plot')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='the folder to load checkpoints')
parser.add_argument('--architecture', '-a', type=str, default='unet',
                    help='model architecture')
parser.add_argument('--seed', '-s', type=int, default=234, 
                    help='which seed for random number generator to use')
parser.add_argument('--gpu', '-g', default=0, type=int, 
                    help='which GPU to use, only when disable-cuda not specified')
args = parser.parse_args()

# set up the seed
if args.seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
# Device
if args.gpu<0 or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    if args.gpu<torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cuda") 
print('Using device: {}'.format(device))

# Hyper-parameters
train_images = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/train/img/"
train_annotations = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/train/ann/"
val_images = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/val/img/"
val_annotations = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/val/ann/"
batch_size=16
classes = ('background', 'foreground')
n_classes = len(classes)
n_workers = 0
input_height = 288
input_width = 288
output_height = 288
output_width = 288
other_inputs_paths=None
preprocessing=None
read_image_type=1

data_transform, target_transform = get_default_transforms('fiber')

# model architecture hyperparameters
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

# contour plot resolution
STEPS = 20
RANDOM = 'normal'
NORM = 'layer'
DIST = 1.0
MODEL_DIR = args.model # '/global/cfs/cdirs/m636/geshi/exp/Fiber/crf/CrossEntropy/0_seed_9999'
trained_on = MODEL_DIR.split('/')[-2]
seed = MODEL_DIR.split('/')[-1]
use_hessian = False

### define model
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

model = model.to(device)
model.eval()

# Define dataset and load data
dataset_parameters = {
    'images': train_images,
    'annotations': train_annotations,
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
val_dataset = get_datset('fiber', dataset_parameters)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

### Get a small batch of data
x, y = iter(dataloader).__next__()

### define the comparison function to sort
def srotFunc(e):
    return int(e.split('-')[0].split('iter')[1])

### define settings to try
try_models = []
for name in os.listdir(MODEL_DIR):
    if name.endswith('.pt'):
        try_models.append(name)
try_models.sort(key=srotFunc, reverse=True)
try_criterions = (nn.CrossEntropyLoss(reduction='mean'), DiceLoss(True), IOULoss(softmax=True))
try_models = try_models[:1]

# Test the performance of the optimal pretrained model
checkpoint = torch.load(os.path.join(MODEL_DIR, try_models[0]), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
res = test_UNet(model, val_dataloader, classes, device)
print(res)

# Pac Bayes
# optimal_dist = _pacbayes_sigma(
#     model,
#     2,
#     train_loader,
#     accuracy = 0.982,
#     search_depth = 10,
#     montecarlo_samples = 10,
#     accuracy_displacement = 0.1,
#     displacement_tolerance = 1e-2,
#     n_dim = 2,
#     random = 'normal',
#     normalization = 'layer',
#     )

# Use the pretrained direction of the final optimal model
if trained_on == 'CrossEntropy':
    trained_on_idx = 0
elif trained_on == 'DICE':
    trained_on_idx = 1
elif trained_on == 'IOU':
    trained_on_idx = 2
else:
    trained_on_idx = 0
    
if use_hessian:
    hessian_comp = hessian(model,
                           try_criterions[trained_on_idx],
                           data=(x.to(device), y.to(device)),
                           device=device)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

    dir_one = ModelParameters(top_eigenvector[0])
    dir_two = ModelParameters(top_eigenvector[1])
else:
    model_start_wrapper = wrap_model(copy.deepcopy(model))
    start_point = model_start_wrapper.get_module_parameters()
    if RANDOM == 'uniform':
        dir_one = rand_u_like(start_point)
    elif RANDOM == 'normal':
        dir_one = rand_n_like(start_point)
    else:
        raise AttributeError('Unsupported random argument. Supported values are uniform and normal')
    dir_two = orthogonal_to(dir_one, RANDOM)

### set up for plotting
n_rows = len(try_models)
n_cols = len(try_criterions)

fig = plt.figure(figsize=(12, 4*n_rows))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1.0)

X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])

### loop over normalization and distance parmaeters
for i,criterion in enumerate(try_criterions):
    for j,model_name in enumerate(try_models):

        checkpoint = torch.load(os.path.join(MODEL_DIR, model_name), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))

        ### compute random projections and loss
        pll = loss_landscapes.PlanarLossLandscape(model, STEPS, deepcopy_model=True)
        pll.precomputed(dir_one, dir_two, distance=DIST, normalization=NORM, centered=True)
        pll.stats_initializer()

        # single batch loss landscape
        since = time.time()
        pll.warm_up(metric)
        print('warmup time cost ', time.time()-since)

        since = time.time()
        loss_data = pll.compute(metric)
        print('compute time cost ', time.time()-since)

        loss_data_fin = log_refined_loss(loss_data)
        
        ### plot stuff
        # update subplot index
        plot_index = ((j * n_cols) + i)+1
        
        # draw 2d 
        ax = fig.add_subplot(n_rows, n_cols, plot_index)
        ax.contour(loss_data_fin, levels=60)    
        
        # draw 3d
        # ax = fig.add_subplot(n_rows, n_cols, plot_index, projection='3d')
        # ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
        # finalize
        title_color = 'red' if i==trained_on_idx else 'black'
        ax.set_title(f'Iterations={model_name.split("-")[0].split("iter")[1]}, criterion={type(criterion).__name__}', fontweight='bold', color=title_color)


### finalize figure
plt.tight_layout()
# plt.savefig(f"iter-criterion-use_hessian-{use_hessian}-fiber-{args.architecture}-{trained_on}-{seed}.pdf", dpi=150)
plt.savefig(f"fiber-use_hessian-{use_hessian}-{args.architecture}-{trained_on}-{seed}.pdf", dpi=150)

plt.close()
# plt.show()