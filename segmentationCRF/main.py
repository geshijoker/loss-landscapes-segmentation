import logging
import time
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from segmentationCRF import metrics
from segmentationCRF.utils import EPS

def train_Fiber(model, dataloader, num_classes, criterion, optimizer, scheduler, num_epochs, device, start_epoch=0, writer=None):
    since = time.time()
    model.train()   # Set model to evaluate mode
    
    pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0)
    # Iterate over data.
    for epoch in pbar:
        running_loss = 0.0
        running_ce = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_corrects = 0
        count = 0

        piter = tqdm(dataloader, desc='Batch', unit='batch', position=1, leave=False)
        for inputs, seg_masks in piter:

            inputs = inputs.to(device)
            seg_masks = seg_masks.to(device)
            _, targets = torch.max(seg_masks, 1)
            
            batch_size = inputs.size(0)
            count += batch_size
            
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, seg_masks)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * batch_size
            running_iou += metrics.iou_coef(targets.flatten(), preds.flatten(), num_classes).item() * batch_size
            running_dice += metrics.dice_coef(targets.flatten(), preds.flatten(), num_classes).item() * batch_size
            running_corrects += ((preds == targets).sum()/np.prod(preds.size())).item() * batch_size

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
    
    return model

def test_Fiber(model, dataloader, classes, device):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    corrects = 0
    count = 0

    tp = np.zeros(len(classes))
    fp = np.zeros(len(classes))
    fn = np.zeros(len(classes))
    n_pixels = np.zeros(len(classes))

    # Iterate over data.
    with torch.no_grad():
        pbar = tqdm(dataloader, unit='batch')
        for inputs, seg_masks in pbar:
            pbar.set_description(f"Test ")

            inputs = inputs.to(device)
            _, seg_masks = torch.max(seg_masks, 1)
            seg_masks = seg_masks.to(device)
            
            batch_size = inputs.size(0)
            count += batch_size

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            corrects += torch.sum(preds == seg_masks.data)/np.prod(preds.size())*batch_size

            pr = preds.flatten()
            gt = seg_masks.flatten()

            for cl_i in range(len(classes)):

                tp[cl_i] += torch.sum((pr == cl_i) * (gt == cl_i)).item()
                fp[cl_i] += torch.sum((pr == cl_i) * ((gt != cl_i))).item()
                fn[cl_i] += torch.sum((pr != cl_i) * ((gt == cl_i))).item()
                n_pixels[cl_i] += torch.sum(gt == cl_i).item()

            acc = corrects.double().item() / count
            pbar.set_postfix(accuracy=100. * acc)

    cl_wise_iou = tp / (tp + fp + fn + EPS)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IOU = np.sum(cl_wise_iou*n_pixels_norm)
    mean_IOU = np.mean(cl_wise_iou)

    acc = corrects.double().item() / count

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Test Acc: {100. * acc}, Test Iou: {mean_IOU}')
    
    return {
        "class_wise_IOU": cl_wise_iou,
        "frequency_weighted_IOU": frequency_weighted_IOU,
        "mean_IOU": mean_IOU,
    }

if __name__ == '__main__':
    pass