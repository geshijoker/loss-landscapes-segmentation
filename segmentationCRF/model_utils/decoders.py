from typing import Any, Optional, Type, TYPE_CHECKING, Union, Callable, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchviz import make_dot
from torchinfo import summary

from segmentationCRF.model_utils.blocks import UnetDoubleConvBlock, UnetUpBlock, UnetOutputBlock, UpBlock

"""
    All decoders so far are based on UNET. There are different UNET decorders
    depending on the type of encoder. This was done due to the dimensionality
    constraints from using certain encoder architectures.
"""

class UnetDecoder(nn.Module):
    def __init__(self, in_channels: List[int], emb_sizes: List[int], out_channels: List[int], kernel_sizes: List[int], paddings: List[int], batch_norm_first: bool=True, bilinear: bool=True):
        super().__init__()
        # self.blocks = nn.ModuleList([
        #     UnetUpBlock(in_channel, emb_size, out_channel, bilinear, kernel_size=kernel_size, padding=padding, batch_norm_first=batch_norm_first) 
        #     for i, (in_channel, emb_size, out_channel, kernel_size, padding) in enumerate(zip(in_channels[:-1], emb_sizes[:-1], out_channels[:-1], kernel_sizes[:-1], paddings[:-1]))
        # ])
        self.blocks = nn.ModuleList([
            UnetUpBlock(in_channel, emb_size, out_channel, bilinear, kernel_size=kernel_size, padding=padding, batch_norm_first=batch_norm_first) 
            for i, (in_channel, emb_size, out_channel, kernel_size, padding) in enumerate(zip(in_channels, emb_sizes, out_channels, kernel_sizes, paddings))
        ])
        # self.conv_proj = UnetDoubleConvBlock(in_channels[-1], emb_sizes[-1], kernel_sizes[-1], paddings[-1], batch_norm_first=batch_norm_first)

    def forward(self, levels: List[Tensor]) -> Tensor:
        assert len(levels)==len(self.blocks)+1, "The size of downsampled results doesn't match the number of upward blocks"
        levels = levels[::-1]
        x = levels[0]
        levels = levels[1:]
        for level, block in zip(levels, self.blocks):
            x = block(x, level)
        # x = self.conv_proj(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels: List[int], emb_sizes: List[int], out_channels: List[int], kernel_sizes: List[int], paddings: List[int], batch_norm_first: bool=True, bilinear: bool=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            UpBlock(in_channel, emb_size, out_channel, bilinear, kernel_size=kernel_size, padding=padding, batch_norm_first=batch_norm_first) 
            for i, (in_channel, emb_size, out_channel, kernel_size, padding) in enumerate(zip(in_channels, emb_sizes, out_channels, kernel_sizes, paddings))
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x
