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
from segmentationCRF import train_epoch, test
from segmentationCRF.crfseg import CRF

"""
example command to run:
python seg_examples/train_Oxford_crf_resume.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/crf-resume/CrossEntropy -n 1 -l ce -r /global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/non-crf/seed_234/0_bs_32_seed_234/iter20-10-31-2023-00:43:24.pt -a unet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose
"""

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--data', '-d', type=str, required=True,
                    help='data folder to load data')
parser.add_argument('--experiment', '-e', type=str, required=True,
                    help='name of the experiment')
parser.add_argument('--name', '-n', type=str, required=True, 
                    help='name of run')
parser.add_argument('--loss', '-l', type=str, default='ce',
                    help='the loss function to use')
parser.add_argument('--resume', '-r', type=str, required=True, 
                    help='resume from checkpoint')
parser.add_argument('--architecture', '-a', type=str, default='unet',
                    help='model architecture')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--percentage', '-p', type=float, default=1.0, 
                    help='the percentage of data used for training')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='which GPU to use, only when disable-cuda not specified')
parser.add_argument('--frequency', '-f', type=int, default=0,
                    help='for every # epochs to save checkpoints')
parser.add_argument('--num_epochs', '-ne', type=int, default=20,
                    help='the number of epochs for training')
parser.add_argument('--label_noise', '-ln', type=float, default=0.00,
                    help='the rate of noisy labels')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001,
                    help='the learning rate of training')
parser.add_argument('--batch_size', '-bs', type=int, default=32,
                    help='the batch size of the data')
parser.add_argument('--arc_depth', '-ad', type=int, default=5,
                    help='the depth of the model')
parser.add_argument('--arc_width', '-aw', type=int, default=32,
                    help='the width of the model')
parser.add_argument('--input_size', '-ip', type=int, default=288,
                    help='the size of input')
parser.add_argument('--test_while_train', '-t', action='store_true',
                    help='using test while train')
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
resume_path = args.resume

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
    
data_path = args.data
lr = args.learning_rate
input_size = args.input_size
batch_size=args.batch_size
arc_width = args.arc_width
arc_depth = args.arc_depth
label_noise = args.label_noise
test_while_train = args.test_while_train
frequency = args.frequency
num_epochs = args.num_epochs
percentage = args.percentage
assert percentage>0 and percentage<=1, "The percentage is out of range of (0, 1)"

writer = SummaryWriter(log_path)

n_workers = 0
classes = ('foreground', 'background', 'border')
n_classes = len(classes)
    
print('n_classes', n_classes)
data_transform, target_transform = get_default_transforms('oxford', input_size, n_classes, noise_level=label_noise)
    
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

train_set = get_datset('oxford', dataset_parameters)
num = int(round(len(train_set)*percentage))
# selected = list(range(num))
selected = random.sample(list(range(len(train_set))), num)
trainset = Subset(train_set, selected)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

if test_while_train:
    dataset_parameters['split'] = 'test'
    test_set = get_datset('oxford', dataset_parameters)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

if args.data_parallel:
    model= nn.DataParallel(model)
model = model.to(device)

checkpoint = torch.load(resume_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

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

optimizer = optim.AdamW(model.parameters(), lr=lr)
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

save_checkpoint('init')

print('Starting training loop; initial compile can take a while...')
since = time.time()
model.train()   # Set model to evaluate mode
start_epoch = 0

pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0)
# Iterate over data.
for epoch in pbar:
    model, epoch_loss, epoch_acc, train_stats = train_epoch(model, train_loader, n_classes, criterion, optimizer, scheduler, device)
    if test_while_train:
        cl_wise_iou, test_stats = test(model, test_loader, n_classes, device)

    if writer:
        writer.add_scalar('time eplased', time.time() - since, epoch)
        for stat in train_stats:
            writer.add_scalar(stat, train_stats[stat], epoch)
        if test_while_train:
            for stat in test_stats:
                writer.add_scalar(stat, test_stats[stat], epoch)
            for cl_i in range(len(cl_wise_iou)):
                writer.add_scalar(f'class_{classes[cl_i]}_iou', cl_wise_iou[cl_i], epoch)

    pbar.set_postfix(loss = epoch_loss, acc = epoch_acc)
        
    if epoch+1==num_epochs or (frequency>0 and (epoch+1)%frequency==0):
        save_checkpoint(epoch+1)

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, last epoch loss: {epoch_loss}, acc: {epoch_acc}')

writer.flush()
writer.close()
