import torch
from torch import nn
import torch.nn.functional as F

from shared import *

CHANNELS = [8, 16, 16]
LATENT_DIM = 3

CONV_NECK_LEN = RESOLUTION // 2 ** len(CHANNELS)

class VAE(nn.Module):
    '''
    Huge thanks to AntixK. 
    See https://github.com/AntixK/PyTorch-VAE
    '''
    def __init__(self, deep_spread) -> None:
        super().__init__()

        modules = []
        last_c = IMG_N_CHANNELS
        for c in CHANNELS:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        last_c, c,
                        kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(c),
                    nn.LeakyReLU(),
                )
            )
            last_c = c
        
        self.conv_neck_dim = (
            CHANNELS[-1] * CONV_NECK_LEN * CONV_NECK_LEN
        )

        self.encoder = nn.Sequential(*modules)
        self.fcMu  = nn.Linear(
            self.conv_neck_dim, LATENT_DIM, 
        )
        self.fcVar = nn.Linear(
            self.conv_neck_dim, LATENT_DIM, 
        )

        if deep_spread:
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
            CHANNELS[  :0:-1], 
            CHANNELS[-2: :-1], 
        ):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c0, c1, 
                        kernel_size=3, stride=2, padding=1,
                        output_padding=1, 
                    ),
                    nn.BatchNorm2d(c1), 
                    nn.LeakyReLU(), 
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.finalLayer = nn.Sequential(
            nn.ConvTranspose2d(
                CHANNELS[0],
                CHANNELS[0],
                kernel_size=3, stride=2, padding=1,
                output_padding=1, 
            ),
            nn.BatchNorm2d(CHANNELS[0]),
            nn.LeakyReLU(),
            nn.Conv2d(
                CHANNELS[0], out_channels=IMG_N_CHANNELS,
                kernel_size=3, padding=1,
            ),
            nn.Tanh(), 
        )

        # print('VAE # of params:', sum(
        #     p.numel() for p in self.parameters() 
        #     if p.requires_grad
        # ))

    def encode(self, x):
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
        t = self.fcBeforeDecode(z)
        t = t.view(
            -1, CHANNELS[-1], CONV_NECK_LEN, CONV_NECK_LEN, 
        )
        t = self.decoder(t)
        t = self.finalLayer(t)
        return t
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructions = self.decode(z)
        return reconstructions, mu, log_var, z
    
    def computeLoss(
        self, x, reconstructions, mu, log_var, beta, 
    ):
        reconstruct_loss = F.mse_loss(reconstructions, x)
        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1, 
        ), dim=0)
        return (
            reconstruct_loss + beta * kld_loss, 
            reconstruct_loss.detach().item(), 
            kld_loss.        detach().item(), 
        )
