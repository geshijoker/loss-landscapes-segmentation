import abc
import typing
import copy

import torch
import torch.nn as nn
from torch.types import Device
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import numpy as np

from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_u_like, rand_n_like
from loss_landscapes.metrics.metric import Metric

def log_refined_loss(loss):
    return np.log(1.+loss)

class SimpleWarmupCaller(object):
    def __init__(self, data_loader: DataLoader, device: typing.Union[None, Device] = None, start=0):
        self.data_loader = data_loader
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start = start

    def __call__(self, model: ModelWrapper):
        model.train()
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader, self.start):
                x, y = batch
                x = x.to(self.device)
                model.forward(x)

class SimpleLossEvalCaller(object):
    def __init__(self, data_loader: DataLoader, criterion: nn.Module, device: typing.Union[None, Device] = None, start=0):
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start = start

    def __call__(self, model: ModelWrapper):
        model.eval()
        count = 0
        loss = 0.
        with torch.no_grad():
            for count, batch in enumerate(self.data_loader, self.start):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.shape[0]

                pred = model.forward(x)
                batch_loss = self.criterion(pred, y)
                                
                loss = count/(count+batch_size)*loss + batch_size/(count+batch_size)*batch_loss.item()
                count+=batch_size
        return loss

def _perturbed_model(
    model: typing.Union[torch.nn.Module, ModelWrapper],
    distance: float,
    n_dim: int,
    random='uniform',
    normalization='filter',
    ) -> ModelWrapper:
    
    model_start_wrapper = wrap_model(copy.deepcopy(model))
    start_point = model_start_wrapper.get_module_parameters()

    if random == 'uniform':
        direction = rand_u_like(start_point)
    elif random == 'normal':
        direction = rand_n_like(start_point)
    else:
        raise AttributeError('Unsupported random argument. Supported values are uniform and normal')

    if normalization == 'model':
        direction.model_normalize_(start_point)
    elif normalization == 'layer':
        direction.layer_normalize_(start_point)
    elif normalization == 'filter':
        direction.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

    direction.mul_((start_point.model_norm() * (n_dim * (distance/2)**2)**(1/n_dim)) / direction.model_norm())
    start_point.sub_(direction)
    return model_start_wrapper

# Adapted from https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(
    model: typing.Union[torch.nn.Module, ModelWrapper],
    distance: float,
    dataloader: DataLoader,
    accuracy: float,
    search_depth: int = 15,
    montecarlo_samples: int = 10,
    accuracy_displacement: float = 0.1,
    displacement_tolerance: float = 1e-2,
    **kwargs,
    ) -> float:

    lower, upper = 0, distance
    sigma = 1
    device = next(model.parameters()).device

    for _ in range(search_depth):
        sigma = (lower + upper) / 2
        accuracy_samples = []
        for _ in range(montecarlo_samples):
            p_model = _perturbed_model(model, sigma, **kwargs)
            loss_estimate = 0
            for data, target in dataloader:
                data = data.to(device)
                target = target.to(device)
                logits = p_model.forward(data)
                pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
                batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
                loss_estimate += batch_correct.sum()
            loss_estimate /= len(dataloader.dataset)
            accuracy_samples.append(loss_estimate)
        displacement = abs(np.mean(accuracy_samples) - accuracy)
        if abs(displacement - accuracy_displacement) < displacement_tolerance:
            break
        elif displacement > accuracy_displacement:
            # Too much perturbation
            upper = sigma
        else:
            # Not perturbed enough to reach target displacement
            lower = sigma
    return sigma

