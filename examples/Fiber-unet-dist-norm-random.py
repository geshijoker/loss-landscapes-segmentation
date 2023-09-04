# libraries
import copy
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import h5py

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics

import loss_landscapes.model_interface.model_parameters as model_parameters
from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import rand_u_like, rand_n_like, orthogonal_to

sys.path.append("/global/u2/g/geshi/Scientific_Segmentation/src")
# sys.path.append("/global/homes/g/geshi/.local/perlmutter/3.9-anaconda-2021.11/lib/python3.9/site-packages/cv2")
from models import SegmentationNet
from data_utils.input_pipeline import FiberSegDataset
from converter import print_model_h5_wegiths, inspect_pytorch_model, load_h5_params

# model architecture hyperparameters
downward_params = {
    'in_channels': 3, 
    'emb_sizes': [64, 128, 256, 512], 
    'kernel_sizes': [3, 3, 3 ,3 ,3], 
    'paddings': [1, 1, 1, 1, 1], 
    'batch_norm_first': False,
}
upward_params = {
    'in_channels': [512, 1536, 768, 384, 192],
    'emb_sizes': [1024, 512, 256, 128, 64], 
    'out_channels': [1024, 512, 256, 128, 64],
    'kernel_sizes': [3, 3, 3, 3, 3], 
    'paddings': [1, 1, 1, 1, 1], 
    'batch_norm_first': False, 
    'bilinear': True,
}
output_params = {
    'in_channels': 64,
    'n_classes': 2,
}

# Hyper-parameters
val_images = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/val/img/"
val_annotations = "/global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter/val/ann/"
batch_size=64
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

# contour plot resolution
STEPS = 20
RANDOM = "normal"

# Device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

### define model
x = torch.rand(1, 3, 288, 288)
model = SegmentationNet(downward_params, upward_params, output_params)
out = model(x)
print('output shape', out.shape)

model = model.to(device)
model.eval()
# stores the initial point in parameter space
model_initial = copy.deepcopy(model)

# load keras weights
keras_weights_file = '/global/cfs/projectdirs/m636/Vis4ML/Fiber/unet_quarter_intern/non_crf/unet_crf_non_crf_quarter_fiber.h5'
model_param_dict = inspect_pytorch_model(model, verbose=False)

# gamma and beta correspond to weight and bias, respectively.
keras2torch_layers= {
    'conv2d':'encoder.blocks.0.conv_proj.conv_proj.0',
    'conv2d_1':'encoder.blocks.0.conv_proj.conv_proj.3',
    'conv2d_2':'encoder.blocks.1.conv_proj.conv_proj.0',
    'conv2d_3':'encoder.blocks.1.conv_proj.conv_proj.3',
    'conv2d_4':'encoder.blocks.2.conv_proj.conv_proj.0',
    'conv2d_5':'encoder.blocks.2.conv_proj.conv_proj.3',
    'conv2d_6':'encoder.blocks.3.conv_proj.conv_proj.0',
    'conv2d_7':'encoder.blocks.3.conv_proj.conv_proj.3',
    'conv2d_8':'decoder.blocks.0.conv_proj.conv_proj.0',
    'conv2d_9':'decoder.blocks.0.conv_proj.conv_proj.3',
    'conv2d_10':'decoder.blocks.1.conv_proj.conv_proj.0',
    'conv2d_11':'decoder.blocks.1.conv_proj.conv_proj.3',
    'conv2d_12':'decoder.blocks.2.conv_proj.conv_proj.0',
    'conv2d_13':'decoder.blocks.2.conv_proj.conv_proj.3',
    'conv2d_14':'decoder.blocks.3.conv_proj.conv_proj.0',
    'conv2d_15':'decoder.blocks.3.conv_proj.conv_proj.3',
    'conv2d_16':'decoder.conv_proj.conv_proj.0',
    'conv2d_17':'decoder.conv_proj.conv_proj.3',
    'conv2d_18':'classifier.classifier',
    'batch_normalization':'encoder.blocks.0.conv_proj.conv_proj.2',
    'batch_normalization_1':'encoder.blocks.0.conv_proj.conv_proj.5',
    'batch_normalization_2':'encoder.blocks.1.conv_proj.conv_proj.2',
    'batch_normalization_3':'encoder.blocks.1.conv_proj.conv_proj.5',
    'batch_normalization_4':'encoder.blocks.2.conv_proj.conv_proj.2',
    'batch_normalization_5':'encoder.blocks.2.conv_proj.conv_proj.5',
    'batch_normalization_6':'encoder.blocks.3.conv_proj.conv_proj.2',
    'batch_normalization_7':'encoder.blocks.3.conv_proj.conv_proj.5',
    'batch_normalization_8':'decoder.blocks.0.conv_proj.conv_proj.2',
    'batch_normalization_9':'decoder.blocks.0.conv_proj.conv_proj.5',
    'batch_normalization_10':'decoder.blocks.1.conv_proj.conv_proj.2',
    'batch_normalization_11':'decoder.blocks.1.conv_proj.conv_proj.5',
    'batch_normalization_12':'decoder.blocks.2.conv_proj.conv_proj.2',
    'batch_normalization_13':'decoder.blocks.2.conv_proj.conv_proj.5',
    'batch_normalization_14':'decoder.blocks.3.conv_proj.conv_proj.2',
    'batch_normalization_15':'decoder.blocks.3.conv_proj.conv_proj.5',
    'batch_normalization_16':'decoder.conv_proj.conv_proj.2',
    'batch_normalization_17':'decoder.conv_proj.conv_proj.5',
}
keras2torch_params = {
    'gamma:0':'weight',
    'beta:0': 'bias',
    'kernel:0': 'weight',
    'bias:0': 'bias',
    'moving_mean:0': 'running_mean',
    'moving_variance:0': 'running_var',
}

# load the final pretrained model
load_h5_params(keras_weights_file, model_param_dict, keras2torch_layers, keras2torch_params, verbose=False) 

# Define dataset and load data
data_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

dataset = FiberSegDataset(val_images, val_annotations, n_classes, 
    input_height, input_width, output_height, output_width,
    transform=data_transform, target_transform = target_transform,
    other_inputs_paths=other_inputs_paths, preprocessing=preprocessing, 
    read_image_type=read_image_type, ignore_segs=False)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

# define criterion
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
model_final = copy.deepcopy(model)

### Get a small batch of data
x, y = iter(dataloader).__next__()
metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))

### define settings to try
try_distance = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3*1e4, 1e5]
try_normalization = ['filter', 'layer', 'model']

### set up for plotting
n_rows = len(try_distance)
n_cols = len(try_normalization)

fig = plt.figure(figsize=(12, 4*n_rows))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1.0)

X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])

def refined_loss(loss):
    return np.log(1.+loss)

### loop over normalization and distance parmaeters
for i,normalization in enumerate(try_normalization):
    for j,distance in enumerate(try_distance):
            
        ### compute random projections and loss
        pll = loss_landscapes.PlanarLossLandscape(model, STEPS, deepcopy_model=True)
        pll.random_plane(distance, normalization, random=RANDOM)
        pll.stats_initializer()
        pll.warm_up(metric)
        loss_data_fin = pll.compute(metric)
        loss_data_fin = refined_loss(loss_data_fin)
        
        ### plot stuff
        # update subplot index
        plot_index = ((j * n_cols) + i)+1
        
        # draw 2d 
        # ax = fig.add_subplot(n_rows, n_cols, plot_index)
        # ax.contour(loss_data_fin, levels=60)    
        
        # draw 3d
        ax = fig.add_subplot(n_rows, n_cols, plot_index, projection='3d')
        ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
        # finalize
        ax.set_title(f'(distance={distance}, normalization="{normalization}")', fontweight='bold')


### finalize figure
plt.tight_layout()
plt.savefig(f"dist-norm-fiber-unet-random-{RANDOM}.pdf", dpi=150)
plt.close()
# plt.show()