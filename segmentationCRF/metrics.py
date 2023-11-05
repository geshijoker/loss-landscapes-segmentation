import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from segmentationCRF.utils import EPS

def get_class_iou(y_true, y_pred, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        tp[cl] = np.sum((y_pred == cl) * (y_true == cl)).item()
        fp[cl] = np.sum((y_pred == cl) * ((y_true != cl))).item()
        fn[cl] = np.sum((y_pred != cl) * ((y_true == cl))).item()
        class_wise[cl] = tp[cl] / (tp[cl] + fp[cl] + fn[cl] + EPS)
    return class_wise

def get_class_dice(y_true, y_pred, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        tp[cl] = np.sum((y_pred == cl) * (y_true == cl)).item()
        fp[cl] = np.sum((y_pred == cl) * ((y_true != cl))).item()
        fn[cl] = np.sum((y_pred != cl) * ((y_true == cl))).item()
        class_wise[cl] = 2*tp[cl] / (2*tp[cl] + fp[cl] + fn[cl] + EPS)
    return class_wise

def iou_coef(y_true, y_pred, n_classes):
    y_true = F.one_hot(y_true, num_classes=n_classes)
    y_pred = F.one_hot(y_pred, num_classes=n_classes)
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    jac = (intersection + EPS) / (union - intersection + EPS)
    return jac
    
def dice_coef(y_true, y_pred, n_classes):
    y_true = F.one_hot(y_true, num_classes=n_classes)
    y_pred = F.one_hot(y_pred, num_classes=n_classes)
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2 * intersection + EPS) / (union + EPS)
    return dice

class DiceLoss(nn.Module):
    def __init__(self, softmax=False):
        super(DiceLoss, self).__init__()
        self.softmax = softmax
        self.smooth = EPS

    def forward(self, pred, target):
        if self.softmax:
            pred = F.softmax(pred,1)
        pred_flat = pred.contiguous().view(-1)
        true_flat = target.contiguous().view(-1)
        intersection = (pred_flat * true_flat).sum()
        union = torch.sum(pred_flat) + torch.sum(true_flat)
        
        return 1 - ((2. * intersection + self.smooth) / (union + self.smooth) )

class IOULoss(nn.Module):
    def __init__(self, softmax=False):
        super(IOULoss, self).__init__()
        self.softmax = softmax
        self.smooth = EPS

    def forward(self, pred, target):
        if self.softmax:
            pred = F.softmax(pred,1)
        pred_flat = pred.contiguous().view(-1)
        true_flat = target.contiguous().view(-1)
        intersection = (pred_flat * true_flat).sum()
        union = torch.sum(pred_flat) + torch.sum(true_flat)
        
        return 1 - ((intersection + self.smooth) / (union - intersection + self.smooth) )
    
def miss_homo(y_true, y_pred):
    mask = (y_true!=y_pred)
    homo = ((y_pred[:,:-1,:]!=y_pred[:,1:,:])*mask[:,:-1,:]).sum() + \
           ((y_pred[:,1:,:]!=y_pred[:,:-1,:])*mask[:,1:,:]).sum() + \
           ((y_pred[:,:,:-1]!=y_pred[:,:,1:])*mask[:,:,:-1]).sum() + \
           ((y_pred[:,:,1:]!=y_pred[:,:,:-1])*mask[:,:,1:]).sum()
    return homo

def homo(y_true, y_pred):
    homo = (y_pred[:,:-1,:]!=y_pred[:,1:,:]).sum() + \
           (y_pred[:,1:,:]!=y_pred[:,:-1,:]).sum() + \
           (y_pred[:,:,:-1]!=y_pred[:,:,1:]).sum() + \
           (y_pred[:,:,1:]!=y_pred[:,:,:-1]).sum()
    return homo

def true_homo(y_true, y_pred):
    homo = 0
    mask = (y_true!=y_pred)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k]:
                    if j>=1 and y_pred[i,j,k]!=y_pred[i,j-1,k]:
                        homo+=1
                    if j<mask.shape[1]-1 and y_pred[i,j,k]!=y_pred[i,j+1,k]:
                        homo+=1
                    if k>=1 and y_pred[i,j,k]!=y_pred[i,j,k-1]:
                        homo+=1
                    if k<mask.shape[2]-1 and y_pred[i,j,k]!=y_pred[i,j,k+1]:
                        homo+=1
    return homo