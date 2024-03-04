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
    
class RegularizedCRFLoss(torch.nn.Module):
    """
    This module provides an implementation of the Gated CRF Loss for Weakly Supervised Semantic Image Segmentation.
    This loss function promotes consistent label assignment guided by input features, such as RGBXY.
    Please consider using the following bibtex for citation:
    @article{obukhov2019gated,
        author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
        title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
        journal={CoRR},
        volume={abs/1906.04651},
        year={2019},
        url={http://arxiv.org/abs/1906.04651},
    }
    https://github.com/meng-tang/rloss/blob/master/pytorch/pytorch-deeplab_v3_plus/DenseCRFLoss.py
    https://github.com/Farbdrucker/GatedCRFLoss/blob/master/models/model_loss_semseg_gatedcrf.py
    """
    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
            compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask.
        :param mask_dst: (optional) Destination mask.
        :param compatibility: (optional) Classes compatibility matrix, defaults to Potts model.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :return: Loss function value.
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape
        device = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )

        denom = N * height_pred * width_pred

        def resize_fix_mask(mask, name):
            assert mask.dim() == 4 and mask.shape[:2] == (N, 1) and mask.dtype == torch.float32, \
                f'{name} mask must be a NCHW batch with C=1 and dtype float32'
            if mask.shape[2:] != (height_pred, width_pred):
                mask = ModelLossSemsegGatedCRF._downsample(
                    mask, 'mask', height_pred, width_pred, custom_modality_downsamplers
                )
            mask[mask != mask] = 0.0    # handle NaN
            mask[mask < 1.0] = 0.0      # handle edges of mask after interpolation
            return mask

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)

        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)

        if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            loss = -(product_kernel_x_y_hat * y_hat_softmax).sum()
            loss = kernels.sum() + loss  # comment out to save computation, total loss may go below 0
        else:
            assert compatibility.shape == (C, C), f'Compatibility matrix expected shape [{C}x{C}]'
            assert (compatibility < 0).int().sum() == 0, 'Compatibility matrix must not have negative values'
            assert compatibility.diag.sum() == 0, 'Compatibility matrix diagonal must be 0'
            compat = (C-1) * F.normalize(compatibility.float().to(device), p=1, dim=1)
            y_hat_CxNHW = y_hat_softmax.permute(1, 0, 2, 3).contiguous().view(C, -1)
            product_kernel_x_y_hat_NHWxC = product_kernel_x_y_hat.permute(0, 2, 3, 1).contiguous().view(-1, C)
            product_CxC = torch.mm(y_hat_CxNHW, product_kernel_x_y_hat_NHWxC)
            loss = (compat * product_CxC).sum()

        out = {
            'loss': loss / denom,
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_input, width_input, height_pred, width_pred
            )

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = ModelLossSemsegGatedCRF._get_mesh(N, height_pred, width_pred, device)
                else:
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]
                    feature = ModelLossSemsegGatedCRF._downsample(
                        feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * ModelLossSemsegGatedCRF._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = ModelLossSemsegGatedCRF._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3], 0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis