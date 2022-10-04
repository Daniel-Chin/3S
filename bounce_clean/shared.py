from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchWork import *

TRAIN_PATH    = './dataset/train'
VALIDATE_PATH = './dataset/validate'
TRAJ_FILENAME = 'trajectory.pickle'
SEQ_LEN = 20
RESOLUTION = 32
IMG_N_CHANNELS = 3
SPACE_DIM = 3
LATENT_DIM = 3

class HyperParams(BaseHyperParams):
    def __init__(self) -> None:
        super().__init__()

        self.I = None
        self.T = None
        self.R = None
        self.TR = None
        self.supervise_rnn: bool = None
        self.supervise_vae: bool = None

        self.variational_rnn: bool = None
        self.vvrnn: bool = None
        self.vvrnn_static: float = None
        self.rnn_min_context = None

        self.rnn_width = None
        self.vae_channels: List[int] = None
        self.deep_spread: bool = None
        self.residual: bool = None

        self.lr = None
        self.grad_clip: Optional[float] = None

        self.image_loss: str = None
        self.teacher_forcing_duration = None

        self.imgCriterion = None
    
    def ready(self):
        assert self.supervise_rnn == (
            self.lossWeightTree['supervise']['rnn'].weight != 0
        )
        assert self.supervise_vae == (
            self.lossWeightTree['supervise']['vae']['encode'].weight != 0 or
            self.lossWeightTree['supervise']['vae']['decode'].weight != 0
        )
        self.imgCriterion = {
            'mse': F.mse_loss, 
            'bce': torch.nn.BCELoss(), 
        }[self.image_loss]

def torch2PIL(torchImg: torch.Tensor):
    return Image.fromarray((
        torchImg.cpu().detach().clamp(0, 1)
        .permute(1, 2, 0) * 255
    ).round().numpy().astype(np.uint8), 'RGB')

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu
