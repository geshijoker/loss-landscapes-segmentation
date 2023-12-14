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
from loss_landscapes.model_interface.model_parameters import ModelParameters

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

class PCATrajectoryTracker(object):
    """
    A FullTrajectoryTracker is a tracker which stores a history of points in the tracked
    model's original parameter space, and can be used to perform a variety of computations
    on the trajectory. The tracker spills data into storage rather than keeping everything
    in main memory.
    """
    def __init__(self, final_model, checkpoints):
        super().__init__()
        self.final_model = final_model
        self.checkpoints = checkpoints
        
        self.final_model_parameters = wrap_model(final_model).get_module_parameters()
        self.architecture = [parameter.shape for parameter in self.final_model_parameters] 
        print('architecture', self.architecture)
        
    def load_checkpoints(self, load_func):
        self.parameters = []
        model = copy.deepcopy(self.final_model)
        for checkpoint in self.checkpoints:
            load_func(model, checkpoint)
            model_parameters = wrap_model(model).get_module_parameters()
            self.parameters.append(flatten(model_parameters)-flatten(self.final_model_parameters))

    def dim_reduction(self, n_components=2):
        self.pca = PCA(n_components=2)
        projections = self.pca.fit_transform(np.array(self.parameters))
        proj_parameters = self.pca.inverse_transform(projections)
        return projections
    
    def inverse_project(self, projection):
        return self.pca.inverse_transform(projection)

    def get_components(self):
        return copy.deepcopy(self.pca.components_)
    
    def get_explained_variance_ratio(self):
        return copy.deepcopy(self.pca.explained_variance_ratio_)
