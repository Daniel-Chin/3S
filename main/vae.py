from typing import Tuple

import torch
from torch import nn

from shared import *

STRIDE = 2
PADDING = 1

class VAE(nn.Module):
    '''
    Huge thanks to AntixK. 
    See https://github.com/AntixK/PyTorch-VAE
    '''
    def __init__(self, hyperParams: HyperParams) -> None:
        super().__init__()
        self.hParams = hyperParams
        channels = [IMG_N_CHANNELS, *hyperParams.vae_channels]
        if hyperParams.relu_leak:
            MyRelu = nn.LeakyReLU
        else:
            MyRelu = nn.ReLU
        def computeConvNeck():
            layer_width = RESOLUTION
            for _ in hyperParams.vae_channels:
                layer_width = (
                    layer_width + PADDING * 2 
                    - hyperParams.vae_kernel_size
                ) // STRIDE + 1
            return layer_width
        self.conv_neck_width = computeConvNeck()

        modules = []
        for c0, c1 in zip(
            channels[0:], 
            channels[1:], 
        ):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        c0, c1, 
                        kernel_size=hyperParams.vae_kernel_size, 
                        stride=STRIDE, padding=PADDING, 
                    ),
                    # nn.BatchNorm2d(c),
                    MyRelu(),
                )
            )
        
        self.conv_neck_dim = (
            channels[-1] * self.conv_neck_width ** 2
        )

        self.encoder = nn.Sequential(*modules)
        self.fcMu  = nn.Linear(
            self.conv_neck_dim, hyperParams.symm.latent_dim, 
        )
        self.fcVar = nn.Linear(
            self.conv_neck_dim, hyperParams.symm.latent_dim, 
        )

        if hyperParams.deep_spread:
            self.fcBeforeDecode = nn.Sequential(
                nn.Linear(
                    hyperParams.symm.latent_dim, 
                    16, 
                ), 
                MyRelu(), 
                nn.Linear(16, 32), 
                MyRelu(), 
                nn.Linear(32, 64), 
                MyRelu(), 
                nn.Linear(
                    64, 
                    self.conv_neck_dim, 
                ), 
            )
        else:
            self.fcBeforeDecode = nn.Linear(
                hyperParams.symm.latent_dim, 
                self.conv_neck_dim, 
            )
        modules = []
        for c0, c1 in zip(
            channels[  :0:-1], 
            channels[-2: :-1], 
        ):
            modules.extend([
                nn.ConvTranspose2d(
                    c0, c1, 
                    kernel_size=hyperParams.vae_kernel_size, 
                    stride=STRIDE, padding=PADDING, 
                ),
                # nn.BatchNorm2d(c1), 
                MyRelu(), 
            ])
        modules[-1] = nn.Sigmoid()
        self.decoder = nn.Sequential(*modules)

        # print('VAE # of params:', sum(
        #     p.numel() for p in self.parameters() 
        #     if p.requires_grad
        # ), flush=True)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        `x` is (batch_size, IMAGE_CHANNELS, height, width)
        '''
        t: torch.Tensor = self.encoder(x)
        t = t.flatten(1)
        # print(t.shape[1], self.conv_neck_dim)
        mu      = self.fcMu (t)
        log_var = self.fcVar(t)
        return mu, log_var
    
    def decode(self, z):
        '''
        `z` is (batch_size, latent_dim)
        '''
        t: torch.Tensor = self.fcBeforeDecode(z)
        t = t.view(
            -1, self.hParams.vae_channels[-1], 
            self.conv_neck_width, self.conv_neck_width, 
        )
        t = self.decoder(t)
        return t
