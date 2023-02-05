from typing import Tuple

import torch
from torch import nn

from shared import *
from hyper_params import *
from symmetry_transforms import identity

class VAE(nn.Module):
    '''
    Huge thanks to AntixK. 
    See https://github.com/AntixK/PyTorch-VAE
    '''
    def __init__(self, hyperParams: HyperParams) -> None:
        super().__init__()
        self.hParams = hyperParams
        channels = [hyperParams.datasetDef.img_n_channels, *hyperParams.vae_channels]
        if hyperParams.relu_leak:
            MyRelu = nn.LeakyReLU
        else:
            MyRelu = nn.ReLU
        def computeConvNeck():
            layer_width = hyperParams.datasetDef.img_resolution
            if isinstance(layer_width, int):
                layer_width = [layer_width, layer_width]
            else:
                layer_width = [*layer_width]
            for kernel_size, padding, stride in zip(
                hyperParams.vae_kernel_sizes, 
                hyperParams.vae_paddings, 
                hyperParams.vae_strides, 
            ):
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                if isinstance(padding, int):
                    padding = (padding, padding)
                if isinstance(stride, int):
                    stride = (stride, stride)
                for dim in range(2):
                    layer_width[dim] = (
                        layer_width[dim] 
                        + padding[dim] * 2 - kernel_size[dim]
                    ) // stride[dim] + 1
            return layer_width
        self.conv_neck = computeConvNeck()

        def zipLayerParams():
            return zip(
                channels[0:], 
                channels[1:], 
                hyperParams.vae_kernel_sizes, 
                hyperParams.vae_paddings, 
                hyperParams.vae_strides, 
            )
        modules = []
        for c0, c1, kernel_size, padding, stride in zipLayerParams():
            sequential = []
            sequential.append(
                nn.Conv2d(
                    c0, c1, 
                    kernel_size=kernel_size, 
                    stride=stride, padding=padding, 
                ),
            )
            if hyperParams.encoder_batch_norm:
                sequential.append(
                    nn.BatchNorm2d(c1),
                )
            sequential.append(
                MyRelu(),
            )
            modules.append(nn.Sequential(*sequential))
        
        self.conv_neck_dim = (
            channels[-1] * self.conv_neck[0] * self.conv_neck[1]
        )

        self.encoder = nn.Sequential(*modules)
        self.fcMu  = nn.Linear(
            self.conv_neck_dim, hyperParams.symm.latent_dim, 
        )
        if not self.hParams.vae_is_actually_ae:
            self.fcVar = nn.Linear(
                self.conv_neck_dim, hyperParams.symm.latent_dim, 
            )

        modules = []
        prev_layer_width = hyperParams.symm.latent_dim
        for layer_width in hyperParams.vae_fc_before_decode:
            modules.append(nn.Linear(
                prev_layer_width, layer_width, 
            ))
            prev_layer_width = layer_width
            modules.append(MyRelu())
        modules.append(nn.Linear(
            prev_layer_width, self.conv_neck_dim, 
        ))  
        # Yes, linear + conv (without activation layer) 
        # is reducible to linear, but it's fine. 
        del prev_layer_width
        self.fcBeforeDecode = nn.Sequential(*modules)

        modules = []
        for c0, c1, kernel_size, padding, stride in reversed([
            *zipLayerParams(), 
        ]):
            modules.extend([
                nn.ConvTranspose2d(
                    c1, c0, 
                    kernel_size=kernel_size, 
                    stride=stride, padding=padding, 
                ),
                # nn.BatchNorm2d(c1), 
                MyRelu(), 
            ])
        assert isinstance(modules[-1], MyRelu)
        modules.pop(-1)
        if hyperParams.vae_sigmoid_after_decode:
            modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

        print('VAE # of params:', sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ), flush=True)

        if hyperParams.lossWeightTree['vicreg'].weight:
            if hyperParams.vicreg_expander_identity:
                self.expander = identity
            else:
                modules = []
                last_width = hyperParams.symm.latent_dim
                for width in hyperParams.vicreg_expander_widths:
                    modules.append(nn.Linear(last_width, width))
                    last_width = width
                    modules.append(nn.BatchNorm1d(width))
                    modules.append(MyRelu())
                self.expander = nn.Sequential(*modules[:-2])

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        `x` is (batch_size, IMAGE_CHANNELS, height, width)
        '''
        t: torch.Tensor = self.encoder(x)
        t = t.flatten(1)
        # print(t.shape[1], self.conv_neck_dim)
        mu      = self.fcMu (t)
        if self.hParams.vae_is_actually_ae:
            log_var = None
        else:
            log_var = self.fcVar(t)
        return mu, log_var
    
    def decode(self, z):
        '''
        `z` is (batch_size, latent_dim)
        '''
        t: torch.Tensor = self.fcBeforeDecode(z)
        t = t.view(
            -1, self.hParams.vae_channels[-1], 
            *self.conv_neck, 
        )
        t = self.decoder(t)
        return t
