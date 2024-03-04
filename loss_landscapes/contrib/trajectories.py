"""
Classes and functions for tracking a model's optimization trajectory and computing
a low-dimensional approximation of the trajectory.
"""

import copy
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA

import torch
from torch import nn

from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import ModelParameters, filter_normalize, layer_normalize, model_normalize 

class TrajectoryTracker(ABC):
    """
    A TrajectoryTracker facilitates tracking the optimization trajectory of a
    DL/RL model. Trajectory trackers provide facilities for storing model parameters
    as well as for retrieving and operating on stored parameters.
    """
    def __init__(self, final_model, checkpoints, norm=None, order=2):
        super().__init__()
        self.final_model = final_model
        self.checkpoints = checkpoints
        self.norm = norm
        self.order = order

    @abstractmethod
    def load_checkpoints(self, load_func):
        pass

    @abstractmethod
    def fit(self, n_components):
        pass
    
    @abstractmethod
    def fit_transform(self, n_components):
        pass
    
    @abstractmethod
    def transform(self, parameters):
        pass
    
    @abstractmethod
    def inverse_transform(self, projection):
        pass

class PCATrajectoryTracker(TrajectoryTracker):
    """
    A FullTrajectoryTracker is a tracker which stores a history of points in the tracked
    model's original parameter space, and can be used to perform a variety of computations
    on the trajectory. The tracker spills data into storage rather than keeping everything
    in main memory.
    """
    def __init__(self, final_model, checkpoints, norm=None, order=2):
        super().__init__()
        self.final_model = final_model
        self.checkpoints = checkpoints
        self.norm = norm
        self.order = order
        
        self.final_model_parameters = wrap_model(final_model).get_module_parameters()
        self.architecture = [parameter.shape for parameter in self.final_model_parameters] 
        print('architecture', self.architecture)
        
    def normalize(self, tensor):
        if self.norm == 'model':
            return model_normalize(tensor, self.order)
        elif self.norm == 'layer':
            return layer_normalize(tensor, self.order)
        elif self.norm == 'filter':
            return filter_normalize(tensor, self.order)
        else:
            return tensor
        
    def load_checkpoints(self, load_func):
        self.parameters = []
        model = copy.deepcopy(self.final_model)
        for checkpoint in self.checkpoints:
            load_func(model, checkpoint)
            model_parameters = wrap_model(model).get_module_parameters()
            self.parameters.append(flatten(model_parameters)-flatten(self.final_model_parameters))
    
    def fit(self, n_components=2):
        self.pca = PCA(n_components=2)
        self.pca.fit(np.array(self.parameters))
    
    def fit_transform(self, n_components=2):
        self.pca = PCA(n_components=2)
        projections = self.pca.fit_transform(np.array(self.parameters))
        return projections
    
    def transform(self, parameters):
        self.pca.transform(parameters)
    
    def inverse_transform(self, projection):
        return self.pca.inverse_transform(projection)

    def get_components(self):
        return copy.deepcopy(self.pca.components_)
    
    def get_explained_variance_ratio(self):
        return copy.deepcopy(self.pca.explained_variance_ratio_)

def flatten(model_parameters):
    parameters = [param.detach().cpu().numpy().flatten() for param in model_parameters]
    return np.concatenate(parameters)

def reconstruct(parameters, architecture):
    split = [np.prod(shape) for shape in architecture]
    pos = 0
    model_parameters = []
    for size, shape in zip(split, architecture):
        model_parameters.append(np.resize(parameters[pos:pos+size], shape))
        pos += size
    return ModelParameters(model_parameters)

def propermap(projections, paddings, n_samples):
    ranges = list(zip(projections.min(axis=0), projections.max(axis=0)))
    xs = []
    for n, padding_rate, (low, high) in zip(n_samples, paddings, ranges):
        pad = (high-low)*padding_rate
        xs.append(np.linspace(low-pad, high+pad, n))
    return np.meshgrid(*xs)