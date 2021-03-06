from argparse import ArgumentError
import os
import torch
import numpy as np
from PIL import Image
from betaSched import Constant

TRAIN_PATH    = './dataset/train'
VALIDATE_PATH = './dataset/validate'
SEQ_LEN = 20
RESOLUTION = 32
IMG_N_CHANNELS = 3
TRAIN_SET_SIZE, VALIDATE_SET_SIZE = 256, 64

class Config:
    keys = [
        'beta', 'vae_loss_coef', 'img_pred_loss_coef', 
        'do_symmetry', 'variational_rnn', 'rnn_width', 
        'deep_spread', 'vae_channels', 'vvrnn', 'vvrnn_static', 
        'rnn_min_context', 'z_pred_loss_coef', 
        'T', 'R', 'TR', 'I', 'lr', 'residual', 'grad_clip', 
    ]
    defaults=[
        Constant(1e-5), 1, 1, 
        True, True, 32, 
        False, [16, 32, 64], False, -5, 
        7, 0.005, 0, 0, 1, 0, 0.001, True, 1, 
    ]
    def __init__(self, *a, **kw):
        for k, v in zip(self.keys, self.defaults):
            self.__setattr__(k, v)
        for i, v in enumerate(a):
            self.__setattr__(self.keys[i], v)
        self.prefix_len = len(a)
        for k, v in kw.items():
            key_i = self.keys.index(k)
            if key_i < self.prefix_len:
                raise ArgumentError(
                    'keyword arg goes backwards.', 
                )
            if key_i == self.prefix_len:
                self.prefix_len += 1
            self.__setattr__(k, v)
        
        do_symmetry = self.T + self.R + self.TR > 0
        assert self.do_symmetry == do_symmetry
    
    def getPrefix(self):
        return [
            self.__getattribute__(k) 
            for k in self.keys[:self.prefix_len]
        ]
    
    def __repr__(self):
        params = []
        for k in self.keys:
            params.append(f'{k}={self.__getattribute__(k) }')
        return f'Config({", ".join(params)})'
    
    def __iter__(self):
        for k in self.keys:
            yield self.__getattribute__(k)

def torch2PIL(torchImg: torch.Tensor):
    return Image.fromarray((
        torchImg.cpu().detach().clamp(0, 1)
        .permute(1, 2, 0) * 255
    ).round().numpy().astype(np.uint8), 'RGB')

EXPERIMENTS_PATH = './experiments'
def renderExperimentPath(
    rand_init_i, config: Config, 
):
    return os.path.join(
        EXPERIMENTS_PATH, f'''{(
            rand_init_i, *config.getPrefix(), 
        )}'''
    )

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu
