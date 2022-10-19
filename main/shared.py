__all__ = [
    'TRAJ_FILENAME', 
    'SEQ_LEN', 
    'RESOLUTION', 
    'IMG_N_CHANNELS', 
    'ENTIRE_DATASET_IN_DEVICE', 
    'SLOW_EVAL_EPOCH_INTERVAL', 
    
    'HyperParams', 'torch2PIL', 'reparameterize', 
]

from typing import Callable, List, Optional
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchWork import *

TRAJ_FILENAME = 'trajectory.json'
SEQ_LEN = 20
RESOLUTION = 32
IMG_N_CHANNELS = 3
ENTIRE_DATASET_IN_DEVICE = True
SLOW_EVAL_EPOCH_INTERVAL = 5000

class HyperParams(BaseHyperParams):
    def __init__(self) -> None:
        super().__init__()

        self.latent_dim: int = None
        
        self.I: int = None
        self.T: int = None
        self.R: int = None
        self.TR: int = None
        self.supervise_rnn: bool = None
        self.supervise_vae: bool = None

        self.variational_rnn: bool = None
        self.vvrnn: bool = None
        self.vvrnn_static: float = None
        self.rnn_min_context: int = None

        self.rnn_width: int = None
        self.residual: bool = None
        self.vae_channels: List[int] = None
        self.deep_spread: bool = None

        self.batch_size: int = None
        self.grad_clip: Optional[float] = None
        self.optim_name: str = None

        self.image_loss: str = None
        self.teacher_forcing_duration: int = None

        self.train_set_size: int = None

        self.imgCriterion: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor, 
        ] = None
    
    def ready(self):
        assert self.supervise_rnn == (
            self.lossWeightTree['supervise']['rnn'].weight != 0
        )
        assert self.supervise_vae == (
            self.lossWeightTree['supervise']['vae']['encode'].weight != 0 or
            self.lossWeightTree['supervise']['vae']['decode'].weight != 0
        )
        assert self.vvrnn == (self.vvrnn_static is None)
        if self.supervise_rnn:
            assert self.lossWeightTree['predict']['z'].weight != 0
        self.imgCriterion = {
            'mse': F.mse_loss, 
            'bce': torch.nn.BCELoss(), 
        }[self.image_loss]
        self.OptimClass = {
            'adam': torch.optim.Adam, 
        }[self.optim_name]
    
    def getTeacherForcingRate(self, epoch):
        try:
            return 1 - min(
                1, epoch / self.teacher_forcing_duration, 
            )
        except ZeroDivisionError:
            return 0
    
    @contextmanager
    def eval(self):
        # enter eval mode
        saved = self.teacher_forcing_duration
        self.teacher_forcing_duration = 0
        try:
            yield
        finally:
            self.teacher_forcing_duration = saved

def torch2PIL(torchImg: torch.Tensor):
    return Image.fromarray((
        torchImg.cpu().detach().clamp(0, 1)
        .permute(1, 2, 0) * 255
    ).round().numpy().astype(np.uint8), 'RGB')

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu