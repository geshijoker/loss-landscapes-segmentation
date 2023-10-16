import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchviz import make_dot
from torchinfo import summary

from segmentationCRF.model_utils import UnetDoubleConvBlock, UnetUpBlock, UnetOutputBlock, UnetEncoder, UnetDecoder

class UNet(nn.Module):
    def __init__(self, downward_params, upward_params, output_params):
        super().__init__()
        self.encoder = UnetEncoder(**downward_params)
        self.decoder = UnetDecoder(**upward_params)
        self.classifier = UnetOutputBlock(**output_params)

    def forward(self, img_input: Tensor):
        img_input, levels = self.encoder(img_input)
        x = self.decoder(levels)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    downward_params = {
        'in_channels': 3, 
        'emb_sizes': [32, 64, 128, 256, 512], 
        'out_channels': [32, 64, 128, 256, 512],
        'kernel_sizes': [3, 3, 3 ,3 ,3], 
        'paddings': [1, 1, 1, 1, 1], 
        'batch_norm_first': False,
    }
    upward_params = {
        'in_channels': [512, 1024, 512, 256, 128],
        'emb_sizes': [1024, 512, 256, 128, 64], 
        'out_channels': [512, 256, 128, 64, 32],
        'kernel_sizes': [3, 3, 3, 3, 3], 
        'paddings': [1, 1, 1, 1, 1], 
        'batch_norm_first': False, 
        'bilinear': True,
    }
    output_params = {
        'in_channels': 64,
        'n_classes': 2,
    }

    x = torch.rand(1, 3, 288, 288)
    model = UNet(downward_params, upward_params, output_params)
    out = model(x)
    print('output shape', out.shape)
    summary(model, input_size=(1, 3, 288, 288))
    
    print('layer weights')
    for name, param in model.named_parameters():
        print('layer', name, 'param', param.shape)
    
    print('state dict')
    for name, param in model.state_dict().items():
        print('state', name, 'param', param.shape)

    # make_dot(out, params=dict(list(model.named_parameters()))).render("../examples/unet_torchviz", format="png")
