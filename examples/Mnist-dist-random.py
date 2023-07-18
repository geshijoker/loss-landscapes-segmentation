# libraries
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics

import loss_landscapes.model_interface.model_parameters as model_parameters
from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import rand_u_like, rand_n_like, orthogonal_to

### training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
EPOCHS = 25
# contour plot resolution
STEPS = 20
RANDOM = "normal"
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### define model

class MLPSmall(torch.nn.Module):
    """ Fully connected feed-forward neural network with one hidden layer. """
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.bn = torch.nn.BatchNorm1d(32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = self.bn(h)
        return F.softmax(self.linear_2(h), dim=1)


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()    
    

def train(model, optimizer, criterion, train_loader, epochs, device):
    """ Trains the given model with the given optimizer, loss function, etc. """
    model.train()
    # train model
    for _ in tqdm(range(epochs), 'Training'):
        for count, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    model.eval()

# download MNIST and setup data loaders
mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)

# define model
model = MLPSmall(IN_DIM, OUT_DIM)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

### points in parameter space
# stores the initial point in parameter space
train(model, optimizer, criterion, train_loader, EPOCHS, device)

### Get a small batch of data
x, y = iter(train_loader).__next__()
metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))

### define settings to try
try_distance = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
try_normalization = ['filter', 'layer', 'model']


### set up for plotting
n_rows = len(try_distance)
n_cols = len(try_normalization)

fig = plt.figure(figsize=(12, 4*n_rows))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1.0)

X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])

### loop over normalization and distance parmaeters
for i,normalization in enumerate(try_normalization):
    for j,distance in enumerate(try_distance):
        
        ### compute random projections and loss
        pll = loss_landscapes.PlanarLossLandscape(model, STEPS, deepcopy_model=True)
        pll.random_plane(distance, normalization, random=RANDOM)
        pll.stats_initializer()
        loss_data_fin = pll.compute(metric)
        
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
plt.savefig(f"dist-norm-mnist-random-{RANDOM}.pdf", dpi=150)
plt.close()
# plt.show()