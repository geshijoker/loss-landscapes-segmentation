import time
import datetime
import random
import sys
import os
import argparse

import numpy as np
import h5py
import pickle
from torchviz import make_dot
from tqdm import tqdm, trange
from ptflops import get_model_complexity_info
from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchinfo import summary
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from segmentationCRF import metrics
from segmentationCRF.models import UNet
from segmentationCRF.data_utils import get_datset, get_default_transforms
from segmentationCRF.utils import check_make_dir
from segmentationCRF.crfseg import CRF

"""
example command to run:
python seg_examples/train_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -c examples/checkpoints.npy -e /global/cfs/cdirs/m636/geshi/exp/Oxford/non-crf/CrossEntropy -n 2 -a unet -l ce -s 234 -p 0.1 -g 1 --benchmark --verbose
"""

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--data', '-d', type=str, required=True,
                    help='data folder to load data')
parser.add_argument('--checkpoints', '-c', type=str, required=True,
                    help='npy file that save the list of iterations to make checkpoints')
parser.add_argument('--experiment', '-e', type=str, required=True,
                    help='name of the experiment')
parser.add_argument('--name', '-n', type=str, required=True, 
                    help='name of run', )
parser.add_argument('--architecture', '-a', type=str, default='unet',
                    help='model architecture')
parser.add_argument('--loss', '-l', type=str, default='ce',
                    help='the loss function to use')
parser.add_argument('--resume', '-r', type=str, default=None, 
                    help='resume from checkpoint')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--percentage', '-p', type=float, default=1.0, 
                    help='the percentage of data used for training')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='which GPU to use, only when disable-cuda not specified')
parser.add_argument('--benchmark', action='store_true',
                    help='using benchmark algorithms')
parser.add_argument('--debug', action='store_true',
                    help='using debug mode')
parser.add_argument('--verbose', action='store_true',
                    help='verbose mode')

# load and parse argument
args = parser.parse_args()

checkpoints = np.load(args.checkpoints)
assert checkpoints.ndim==1, 'The loaded checkpoints is not a 1D array'
print(checkpoints)

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
    
experiment = args.experiment
run_name = args.name + f'_seed_{seed}'
log_path = os.path.join(experiment, run_name)

if os.path.isdir(log_path):
    sys.exit('The name of the run has alrealy exist')
else:
    check_make_dir(log_path)
sys.stdout = open(os.path.join(log_path, 'log.txt'), 'w')

# set up benchmark running
if args.benchmark:
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.benchmark = False
    
if args.debug:
    torch.autograd.set_detect_anomaly(True)
else:
    torch.autograd.set_detect_anomaly(False)

writer = SummaryWriter(log_path)

data_path = args.data
input_size = 224
batch_size=32
n_workers = 0
classes = ('foreground', 'background', 'border')
n_classes = len(classes)
num_epochs = 15
percentage = args.percentage
assert percentage>0 and percentage<=1, "The percentage is out of range of (0, 1)"

data_transform, target_transform = get_default_transforms('oxford')

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

x = torch.rand(batch_size, 3, input_size, input_size)

if args.architecture == 'unet':
    model = UNet(downward_params, upward_params, output_params)
elif args.architecture == 'unet-crf':
    unet = UNet(downward_params, upward_params, output_params)
    model = nn.Sequential(
        unet,
        CRF(n_spatial_dims=2)
    )
else:
    raise ValueError("Model architecture {} is not supported".format(args.architecture))
out = model(x)
print('output shape', out.shape) 

dataset_parameters = {
    'data_path': data_path,
    'split': 'trainval',
    'data_transform': data_transform,
    'target_transform': target_transform,
    'download': True,
}

dataset = get_datset('oxford', dataset_parameters)
num = int(round(len(dataset)*percentage))
selected = list(range(num))
dataset = Subset(dataset, selected)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

model = model.to(device)

if args.loss == 'iou':
    print('use IOULoss')
    criterion = metrics.IOULoss(softmax=True)
elif args.loss == 'dice':
    print('use DiceLoss')
    criterion = metrics.DiceLoss(softmax=True)
elif args.loss == 'ce':
    print('use CrossEntropy')
    criterion = nn.CrossEntropyLoss() 
else:
    print(f'{args.loss} not supported, use default CrossEntropyLoss')
    criterion = nn.CrossEntropyLoss() 

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20)

since = time.time()
model.train()   # Set model to evaluate mode
start_epoch = 0

def save_checkpoint(checkpoint):
    utctime = datetime.datetime.now(datetime.timezone.utc).strftime("%m-%d-%Y-%H:%M:%S")
    model_path = os.path.join(log_path, f'iter{checkpoint}-' + utctime + '.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'iteration': checkpoint,
        }, model_path)

save_checkpoint(0)

print('Starting training loop; initial compile can take a while...')
pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0)
# Iterate over data.
for epoch in pbar:
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_corrects = 0
    count = 0

    piter = tqdm(dataloader, desc='Batch', unit='batch', position=1, leave=False)
    for i, (inputs, seg_masks) in enumerate(piter):

        inputs = inputs.to(device)
        seg_masks = seg_masks.to(device)
        _, targets = torch.max(seg_masks, 1)
        batch_size = inputs.size(0)
        
        count += batch_size
        cur_iter = epoch * len(piter) + i
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, seg_masks)

        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * batch_size
        running_iou += metrics.iou_coef(targets.flatten(), preds.flatten(), n_classes).item() * batch_size
        running_dice += metrics.dice_coef(targets.flatten(), preds.flatten(), n_classes).item() * batch_size
        running_corrects += ((preds == targets).sum()/np.prod(preds.size())).item() * batch_size
        
        if cur_iter in checkpoints:
            save_checkpoint(cur_iter)

    scheduler.step()
    epoch_loss = running_loss / count
    epoch_iou = running_iou / count
    epoch_dice = running_dice / count
    epoch_acc = running_corrects / count

    if writer is not None:
        writer.add_scalar('train loss', epoch_loss, epoch)
        writer.add_scalar('train iou', epoch_iou, epoch)
        writer.add_scalar('train dice', epoch_dice, epoch)
        writer.add_scalar('train acc', 100. * epoch_acc, epoch)

    pbar.set_postfix(loss = epoch_loss, acc=100. * epoch_acc, iou = epoch_iou, dice = epoch_dice)

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, last epoch loss: {epoch_loss}, acc: {100. * epoch_acc}, Iou: {epoch_iou}, Dice: {epoch_dice}')

writer.flush()
writer.close()
