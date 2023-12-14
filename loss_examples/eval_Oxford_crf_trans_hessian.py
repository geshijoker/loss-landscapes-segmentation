# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_u_like, rand_n_like, orthogonal_to
from loss_landscapes.contrib.functions import SimpleWarmupCaller, SimpleLossEvalCaller, log_refined_loss

from pyhessian.utils import *
from pyhessian import hessian, get_esd_plot, density_generate # ESD plot

from segmentationCRF import metrics
from segmentationCRF.models import UNet
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
python loss_examples/eval_Oxford_crf_trans_hessian.py -d /global/cfs/cdirs/m636/geshi/data/ -rc /global/cfs/cdirs/m636/geshi/exp/Oxford/crf/CrossEntropy/0_seed_9999/iter30-11-17-2023-19:03:26.pt -rn /global/cfs/cdirs/m636/geshi/exp/Oxford/non-crf/CrossEntropy/0_seed_9999/iter30-11-17-2023-18:04:01.pt -s 9999 -g 0 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose
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
STEPS = 20
top_n = 2
NORM = 'layer'
DIST = 0.01

# load and parse argument
args = parser.parse_args()

unet_crf_path = args.resume_crf
unet_path = args.resume_noncrf
unet_crf_log_path = os.path.dirname(unet_crf_path)
unet_log_path = os.path.dirname(unet_path)
unet_crf_name = os.path.basename(unet_crf_path)
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

unet = unet.to(device)
unet.eval()
checkpoint = torch.load(unet_path, map_location=device)
unet.load_state_dict(checkpoint['model_state_dict'])
# test(unet, val_dataloader, n_classes, device)

unet_crf = unet_crf.to(device)
unet_crf.eval()
checkpoint = torch.load(unet_crf_path, map_location=device)
unet_crf.load_state_dict(checkpoint['model_state_dict'])
# test(unet_crf, val_dataloader, n_classes, device)

trans_crf = nn.Sequential(
    unet,
    unet_crf[1]
)

# test(trans_crf, val_dataloader, n_classes, device)

def compute_eigen(model, criterion, x, y, top_n=2, save_path=None, save_name=None):
    hessian_comp = hessian(model,
                           criterion,
                           data=(x, y),
                           device=x.device)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=top_n)

    dir_one = ModelParameters(top_eigenvector[0])
    dir_two = ModelParameters(top_eigenvector[1])
    
    trace = hessian_comp.trace()
    print("The trace of this model is: %.4f"%(np.mean(trace)))
    
    density_eigen, density_weight = hessian_comp.density()
    
    density, grids = density_generate(density_eigen, density_weight)
    
    if save_path and save_name:
        fig = plt.figure()
        plt.semilogy(grids, density + 1.0e-7)
        plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
        plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.axis([np.min(density_eigen) - 1, np.max(density_eigen) + 1, None, None])
        plt.text(
                (np.max(density_eigen)+1)*0.85,
                1 - 0.5,
                ("mean trace %.2f" % np.mean(trace)).lstrip("0"),
                size=30,
                horizontalalignment="right",
            )
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, save_name+'-hessian-spectrum.png'), dpi=150)
        plt.close(fig)
        
    if save_path and save_name:
        with open(os.path.join(save_path, save_name+'-top{}-eigenvalues.npy'.format(top_n)), 'wb') as f:
            np.save(f, top_eigenvalues)
    
    return trace, dir_one, dir_two
        
def create_loss_landscape(model, metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=None, save_name=None):
    # compute loss data
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
    
    if save_path and save_name:
        with open(os.path.join(save_path, save_name+'-pyhessian-loss-landscape.npy'), 'wb') as f:
            np.save(f, loss_data)
    
    if save_path and save_name:
        fig = plt.figure()
        plt.contour(loss_data, levels=50)
        plt.title('Loss Contours around Trained Model')
        plt.show()
        plt.savefig(os.path.join(save_path, save_name+'-pyhessian-2d.png'), dpi=150)
        plt.close(fig)
    
    if save_path and save_name:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Surface Plot of Loss Landscape')
        fig.show()
        plt.savefig(os.path.join(save_path, save_name+'-pyhessian-3d.png'), dpi=150)
        plt.close(fig)
        
    return loss_data

criterion = torch.nn.CrossEntropyLoss() # DiceLoss(True), IOULoss(softmax=True)
x, y = iter(dataloader).__next__()

##### unet crf
model_final = copy.deepcopy(unet_crf)
unet_final = model_final[0]
crf_final = model_final[1]

metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))
crf_metric = loss_landscapes.metrics.CRFPerturbLoss(copy.deepcopy(unet_final), criterion, x.to(device), y.to(device))
unet_metric = loss_landscapes.metrics.BackbonePerturbLoss(copy.deepcopy(crf_final), criterion, x.to(device), y.to(device))

trace, dir_one, dir_two = compute_eigen(model_final, criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_crf_log_path, save_name='unet-crf')
loss_data = create_loss_landscape(model_final, metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_crf_log_path, save_name='unet-crf')

trace, dir_one, dir_two = compute_eigen(copy.deepcopy(unet_final), criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_crf_log_path, save_name='unet-crf-no-crf')
loss_data = create_loss_landscape(copy.deepcopy(unet_final), metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_crf_log_path, save_name='unet-crf-no-crf')

trace, dir_one, dir_two = compute_eigen(copy.deepcopy(unet_final), criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_crf_log_path, save_name='unet-crf-only-unet')
loss_data = create_loss_landscape(copy.deepcopy(unet_final), unet_metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_crf_log_path, save_name='unet-crf-only-unet')

trace, dir_one, dir_two = compute_eigen(copy.deepcopy(crf_final), criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_crf_log_path, save_name='unet-crf-only-crf')
loss_data = create_loss_landscape(copy.deepcopy(crf_final), crf_metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_crf_log_path, save_name='unet-crf-only-crf')

##### crf transplant
model_final = copy.deepcopy(trans_crf)
unet_final = model_final[0]
crf_final = model_final[1]

metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))
crf_metric = loss_landscapes.metrics.CRFPerturbLoss(copy.deepcopy(unet_final), criterion, x.to(device), y.to(device))
unet_metric = loss_landscapes.metrics.BackbonePerturbLoss(copy.deepcopy(crf_final), criterion, x.to(device), y.to(device))

trace, dir_one, dir_two = compute_eigen(model_final, criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_log_path, save_name='unet-transplant-crf')
loss_data = create_loss_landscape(model_final, metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_log_path, save_name='unet-transplant-crf')

trace, dir_one, dir_two = compute_eigen(copy.deepcopy(unet_final), criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_log_path, save_name='unet')
loss_data = create_loss_landscape(copy.deepcopy(unet_final), metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_log_path, save_name='unet')

trace, dir_one, dir_two = compute_eigen(copy.deepcopy(unet_final), criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_log_path, save_name='unet-only-unet')
loss_data = create_loss_landscape(copy.deepcopy(unet_final), unet_metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_log_path, save_name='unet-only-unet')

trace, dir_one, dir_two = compute_eigen(copy.deepcopy(crf_final), criterion, x.to(device), y.to(device), top_n=top_n, save_path=unet_log_path, save_name='unet-only-crf')
loss_data = create_loss_landscape(copy.deepcopy(crf_final), crf_metric, dir_one, dir_two, STEPS, DIST, NORM, save_path=unet_log_path, save_name='unet-only-crf')
