from typing import Any, Optional, Type, TYPE_CHECKING, Union, Callable, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from segmentationCRF.model_utils.blocks import UnetDownBlock, DownBlock

class UnetEncoder(nn.Module):
    def __init__(self, in_channels: int, emb_sizes: List[int], out_channels: List[int], kernel_sizes: List[int], paddings: List[int], batch_norm_first: bool=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            UnetDownBlock(in_channels, emb_sizes[0], out_channels[0], kernel_sizes[0], paddings[0], batch_norm_first),
            *[UnetDownBlock(in_channel, emb_size, out_channel, kernel_size, padding, batch_norm_first)
            for i, (in_channel, emb_size, out_channel, kernel_size, padding) in enumerate(zip(out_channels[:-1], emb_sizes[1:], out_channels[1:], kernel_sizes[1:], paddings[1:]))]
        ])

    def forward(self, img_input: Tensor) -> Tensor:
        x = img_input
        levels = []
        for block in self.blocks:
            prev, x = block(x, True)
            levels.append(prev)
        levels.append(x)
        return img_input, levels
    
class Encoder(nn.Module):
    def __init__(self, in_channels: int, emb_sizes: List[int], out_channels: List[int], kernel_sizes: List[int], paddings: List[int], batch_norm_first: bool=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            DownBlock(in_channels, emb_sizes[0], out_channels[0], kernel_sizes[0], paddings[0], batch_norm_first),
            *[DownBlock(in_channel, emb_size, out_channel, kernel_size, padding, batch_norm_first)
            for i, (in_channel, emb_size, out_channel, kernel_size, padding) in enumerate(zip(out_channels[:-1], emb_sizes[1:], out_channels[1:], kernel_sizes[1:], paddings[1:]))]
        ])

    def forward(self, img_input: Tensor) -> Tensor:
        x = img_input
        for block in self.blocks:
            x = block(x)
        return x