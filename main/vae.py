from typing import Tuple

import torch
from torch import nn

from shared import *

class VAE(nn.Module):
    '''
    Huge thanks to AntixK. 
    See https://github.com/AntixK/PyTorch-VAE
    '''
    def __init__(self, hyperParams: HyperParams) -> None:
        super().__init__()
        self.hParams = hyperParams
        channels = [IMG_N_CHANNELS, *hyperParams.vae_channels]
        self.conv_neck_len = RESOLUTION // 2 ** len(
            hyperParams.vae_channels, 
        )

        modules = []
        for c0, c1 in zip(
            channels[0:], 
            channels[1:], 
        ):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        c0, c1, 
                        kernel_size=3, stride=2, padding=1
                    ),
                    # nn.BatchNorm2d(c),
                    nn.LeakyReLU(),
                )
            )
        
        self.conv_neck_dim = (
            channels[-1] * self.conv_neck_len ** 2
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
                    8, 
                ), 
                nn.LeakyReLU(), 
                nn.Linear(
                    8, 
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
                    kernel_size=3, stride=2, padding=1,
                    output_padding=1, 
                ),
                # nn.BatchNorm2d(c1), 
                nn.LeakyReLU(), 
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
            self.conv_neck_len, self.conv_neck_len, 
        )
        t = self.decoder(t)
        return t
