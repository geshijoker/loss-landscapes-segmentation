import copy
import time
import os
import sys
import argparse
import time
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms

from vit_pytorch import ViT
from robustbench.data import load_cifar10c

import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_u_like, rand_n_like, orthogonal_to
from loss_landscapes.contrib.functions import SimpleWarmupCaller, SimpleLossEvalCaller, log_refined_loss

sys.path.append("/global/u2/g/geshi/PyHessian")
from utils import * # get the dataset
from pyhessian import hessian # Hessian computation
from models.resnet import resnet
from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model

def get_data_cifar10(data_dir, batch_size, num_cores=0, shuffle=False):

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                            ]))

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=shuffle)
    
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                               
                            ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_cores, shuffle=False)
    return train_subset_loader, test_subset_loader

def test_accuracy(test_loader, net):
    """Evaluate testset accuracy of a model."""
    net.eval()
    acc_sum, count = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # send data to the GPU if cuda is availabel
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            count += inputs.size(0)
            
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)

            # labels = labels.long()
            acc_sum += torch.sum(preds == labels.data).item()
    return acc_sum/count

# Parsing
parser = argparse.ArgumentParser(description="Parse arguments to load pretrained ResNet and compute loss landscape on CIFAR10")
parser.add_argument('--model', type=str, required=True, help='model name loaded and used used')
parser.add_argument('--data', type=str, required=True, help='the directory to load data from')
parser.add_argument('--output', type=str, required=True, help='the directory to save loss landscape as npy file')
parser.add_argument('--corrupted', action='store_true', help='if used, load CIFAR10C instead of CIFAR10')
parser.add_argument('--hessian', action='store_true', help='if used, use pyhessian directions')
parser.add_argument('--batch', type=int, default=512, help='the batch size to use')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=int, default=0, help='which device to use, if exist, use cuda, if not, use cpu')
parser.add_argument('--distance', type=float, default=1, help='what distance to use in computing loss landscapes')
parser.add_argument('--steps', type=int, default=40, help='how many steps to use in computing loss landscapes')
parser.add_argument('--norm', type=str, default='layer', help='what normalization is used in computing loss landscapes')
parser.add_argument('--random', type=str, default='normal', help='what random mode is used in computing loss landscapes')
args = parser.parse_args()

"""
example:
python loss-lens-Vit-cifar10.py 
--model '/global/cfs/cdirs/m636/JiaqingChen/ViT-models/VIT_model_seed_0_threshold_{00}.pt' 
--data '/global/cfs/cdirs/m636/geshi/data' --output '/global/cfs/cdirs/m636/geniesse/Vit_loss_lens' 
--corrupted --batch 512 --seed 0 --device 0 --distance 0.05 --steps 40 --norm layer --random normal
"""

# Variables
model_name = args.model
data_dir = args.data
output_dir = args.output
use_corrupted = bool(args.corrupted)
use_hessian = bool(args.hessian)
batch_size = int(args.batch)
random_seed = int(args.seed)
DIST = float(args.distance)
STEPS = int(args.steps)
NORM = args.norm
RANDOM = args.random

# Device
if torch.cuda.is_available() and args.device in list(range(torch.cuda.device_count())):
    device = torch.device(f'cuda:{args.device}') 
    use_cuda = True
else:
    device = torch.device('cpu')
    use_cuda = False

# seed setting
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# get the model 
# get the model 
model = ViT(image_size = 32, 
            patch_size = 4, 
            num_classes = 10, 
            dim = 1024, 
            depth = 6, 
            heads = 16, 
            mlp_dim = 2048, 
            dropout = 0.1, 
            emb_dropout = 0.1)

model = model.to(device)

# create loss function
criterion = torch.nn.CrossEntropyLoss()

if use_corrupted:
    x, y = load_cifar10c(n_examples=batch_size, data_dir=data_dir)
else:
    train_loader, test_loader = get_data_cifar10(data_dir, batch_size, num_cores=0)
    x, y = iter(train_loader).__next__()

metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))

pll = loss_landscapes.PlanarLossLandscape(model, STEPS, deepcopy_model=True)

if use_hessian:
    hessian_comp = hessian(model,
                           criterion,
                           data=(x.to(device), y.to(device)),
                           cuda=use_cuda)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

    dir_one = ModelParameters(top_eigenvector[0])
    dir_two = ModelParameters(top_eigenvector[1])

    pll.precomputed(dir_one, dir_two, distance=DIST, normalization=NORM, centered=True)
else:
    # compute loss data
    pll.random_plain(distance=DIST, normalization=NORM, random=RANDOM, centered=True)
pll.stats_initializer()

# single batch loss landscape
since = time.time()
pll.warm_up(metric)
print('warmup time cost ', time.time()-since)

since = time.time()
loss_data = pll.compute(metric)
print('compute time cost ', time.time()-since)
print(loss_data.shape)

output_name = os.path.join(output_dir, f"{model_name.split('/')[-1].split('.')[0]}_hessian_{use_hessian}_corrupted_{use_corrupted}_batch_size_{batch_size}_distance_{DIST}_steps_{STEPS}_norm_{NORM}_random_{RANDOM}.npy")
print('output_name ', output_name)
with open(output_name, 'wb') as f:
    np.save(f, loss_data)
