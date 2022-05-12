import os
from collections import namedtuple
import torch
import numpy as np
from PIL import Image

TRAIN_PATH    = './dataset/train'
VALIDATE_PATH = './dataset/validate'
SEQ_LEN = 20
RESOLUTION = 32
IMG_N_CHANNELS = 3
TRAIN_SET_SIZE, VALIDATE_SET_SIZE = 256, 64

Config = namedtuple('Config', [
    'beta', 'vae_loss_coef', 'rnn_loss_coef', 'do_symmetry', 
    'variational_rnn', 'rnn_width', 'deep_spread', 
    'vae_channels', 'vvrnn', 
])

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
            rand_init_i, *config, 
        )}'''
    )

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu
