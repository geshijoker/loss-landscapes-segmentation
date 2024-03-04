import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

class UnetDoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, out_channels: int, kernel_size: int, padding: int, batch_norm_first: bool=True):
        super().__init__()
        if batch_norm_first:
            self.conv_proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, padding=padding, bias=True, padding_mode='zeros'),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=n_filters, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True, padding_mode='zeros'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, padding=padding, bias=True, padding_mode='zeros'),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(n_filters),
                nn.Conv2d(in_channels=n_filters, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True, padding_mode='zeros'),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_proj(x)

class UnetDownBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, out_channels: int, kernel_size: int, padding: int, batch_norm_first: bool=True):
        super().__init__()
        self.conv_proj = UnetDoubleConvBlock(in_channels, n_filters, out_channels, kernel_size, padding, batch_norm_first)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor, pool=True):
        x = self.conv_proj(x)
        if pool:
            p = self.pool(x)
            return x, p
        else:
            return x

class UnetUpBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, out_channels: int, bilinear: bool=True, **kwargs):
        super().__init__()
        self.conv_proj = UnetDoubleConvBlock(in_channels, n_filters, out_channels, **kwargs)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kwargs['kernel_size'], stride=2, bias=False, padding=1, output_padding=1)

    def forward(self, x1: Tensor, x2: Tensor):
        x1 = self.conv_proj(x1)
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return x

class UnetOutputBlock(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, n_classes, (1, 1), padding_mode='zeros')

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, out_channels: int, kernel_size: int, padding: int, batch_norm_first: bool=True):
        super().__init__()
        self.conv_proj = UnetDoubleConvBlock(in_channels, n_filters, out_channels, kernel_size, padding, batch_norm_first)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor):
        x = self.conv_proj(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, out_channels: int, bilinear: bool=True, **kwargs):
        super().__init__()
        self.conv_proj = UnetDoubleConvBlock(in_channels, n_filters, out_channels, **kwargs)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kwargs['kernel_size'], stride=2, bias=False, padding=1, output_padding=1)

    def forward(self, x: Tensor):
        x = self.conv_proj(x)
        x = self.up(x)
        return x