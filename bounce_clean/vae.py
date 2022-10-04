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
        channels = hyperParams.vae_channels
        self.conv_neck_len = RESOLUTION // 2 ** len(channels)

        modules = []
        last_c = IMG_N_CHANNELS
        for c in channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        last_c, c,
                        kernel_size=3, stride=2, padding=1
                    ),
                    # nn.BatchNorm2d(c),
                    nn.LeakyReLU(),
                )
            )
            last_c = c
        
        self.conv_neck_dim = (
            channels[-1] * self.conv_neck_len ** 2
        )

        self.encoder = nn.Sequential(*modules)
        self.fcMu  = nn.Linear(
            self.conv_neck_dim, LATENT_DIM, 
        )
        self.fcVar = nn.Linear(
            self.conv_neck_dim, LATENT_DIM, 
        )

        if hyperParams.deep_spread:
            self.fcBeforeDecode = nn.Sequential(
                nn.Linear(
                    LATENT_DIM, 
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
                LATENT_DIM, 
                self.conv_neck_dim, 
            )
        modules = []
        for c0, c1 in zip(
            channels[  :0:-1], 
            channels[-2: :-1], 
        ):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c0, c1, 
                        kernel_size=3, stride=2, padding=1,
                        output_padding=1, 
                    ),
                    # nn.BatchNorm2d(c1), 
                    nn.LeakyReLU(), 
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.finalLayer = nn.Sequential(
            nn.ConvTranspose2d(
                channels[0],
                channels[0],
                kernel_size=3, stride=2, padding=1,
                output_padding=1, 
            ),
            # nn.BatchNorm2d(CHANNELS[0]),
            nn.LeakyReLU(),
            nn.Conv2d(
                channels[0], out_channels=IMG_N_CHANNELS,
                kernel_size=3, padding=1,
            ),
            nn.Sigmoid(), 
        )

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
        `z` is (batch_size, LATENT_DIM)
        '''
        t: torch.Tensor = self.fcBeforeDecode(z)
        t = t.view(
            -1, self.hParams.vae_channels[-1], 
            self.conv_neck_len, self.conv_neck_len, 
        )
        t = self.decoder(t)
        t = self.finalLayer(t)
        return t
        
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        reconstructions = self.decode(z)
        return reconstructions, mu, log_var, z
