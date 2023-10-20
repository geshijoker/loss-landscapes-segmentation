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

def train(model, dataloader, num_classes, criterion, optimizer, scheduler, num_epochs, device, start_epoch=0, writer=None):
    since = time.time()
    model.train()   # Set model to evaluate mode
    
    pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0)
    # Iterate over data.
    for epoch in pbar:
        model, epoch_loss, epoch_acc, train_stats = train_epoch(model, dataloader, num_classes, criterion, optimizer, scheduler, device)

        if writer:
            writer.add_scalars('train_stats', train_stats, epoch)

        pbar.set_postfix(loss = epoch_loss, acc = epoch_acc)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, last epoch loss: {epoch_loss}, acc: {100. * epoch_acc}')
    
    return model

def train_epoch(model, dataloader, num_classes, criterion, optimizer, scheduler, device):
    epoch_loss = 0.0
    epoch_acc = 0
    epoch_ce = 0.0
    epoch_iou = 0.0
    epoch_dice = 0.0
    count = 0

    piter = tqdm(dataloader, desc='Batch', unit='batch', position=1, leave=False)
    for inputs, seg_masks in piter:

        inputs = inputs.to(device)
        seg_masks = seg_masks.to(device)
        _, targets = torch.max(seg_masks, 1)

        batch_size = inputs.size(0)
        nxt_count = count+batch_size
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, seg_masks)

        loss.backward()
        optimizer.step()

        # statistics
        epoch_loss += loss.item() * batch_size/nxt_count + epoch_loss * count/nxt_count
        epoch_acc += ((preds == targets).sum()/np.prod(preds.size())).item() * batch_size/nxt_count + epoch_acc * count/nxt_count
        epoch_ce += nn.CrossEntropyLoss(outputs, seg_masks).item() * batch_size/nxt_count + epoch_ce * count/nxt_count
        epoch_iou += metrics.iou_coef(targets.flatten(), preds.flatten(), num_classes).item() * batch_size/nxt_count + epoch_iou * count/nxt_count
        epoch_dice += metrics.dice_coef(targets.flatten(), preds.flatten(), num_classes).item() * batch_size/nxt_count + epoch_dice * count/nxt_count
        
        count = nxt_count

    epoch_acc *= 100.
    scheduler.step()
    train_stats = {
        'epoch_loss': epoch_loss,
        'epoch_acc': epoch_acc,
        'epoch_ce': epoch_ce,
        'epoch_iou': epoch_iou,
        'epoch_dice': epoch_dice,
    }
    
    return model, epoch_loss, epoch_acc, train_stats

def test(model, dataloader, num_classes, device):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    
    corrects = 0
    count = 0

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    n_pixels = np.zeros(num_classes)

    # Iterate over data.
    with torch.no_grad():
        piter = tqdm(dataloader, unit='batch')
        for inputs, seg_masks in pbar:
            piter.set_description(f"Test ")

            inputs = inputs.to(device)
            _, targets = torch.max(seg_masks, 1)
            targets = targets.to(device)
            
            batch_size = inputs.size(0)
            count += batch_size

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            corrects += torch.sum(preds == targets.data)/np.prod(preds.size())*batch_size

            pr = preds.flatten()
            gt = targets.flatten()

            for cl_i in range(num_classes):

                tp[cl_i] += torch.sum((pr == cl_i) * (gt == cl_i)).item()
                fp[cl_i] += torch.sum((pr == cl_i) * ((gt != cl_i))).item()
                fn[cl_i] += torch.sum((pr != cl_i) * ((gt == cl_i))).item()
                n_pixels[cl_i] += torch.sum(gt == cl_i).item()

            acc = corrects.double().item() / count
            piter.set_postfix(accuracy=100. * acc)

    cl_wise_iou = tp / (tp + fp + fn + EPS)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IOU = np.sum(cl_wise_iou*n_pixels_norm)
    mean_IOU = np.mean(cl_wise_iou)

    acc = corrects.double().item() / count

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Test Acc: {100. * acc}, Test Iou: {mean_IOU}')
    
    test_stats = {
        "class_wise_IOU": cl_wise_iou,
        "frequency_weighted_IOU": frequency_weighted_IOU,
        "mean_IOU": mean_IOU,
    }

    return cl_wise_iou, test_stats

if __name__ == '__main__':
    pass