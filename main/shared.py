__all__ = [
    'TRAJ_FILENAME', 
    'RESOLUTION', 
    'IMG_N_CHANNELS', 
    'ENTIRE_DATASET_IN_DEVICE', 
    'SLOW_EVAL_EPOCH_INTERVAL', 
    
    'HyperParams', 'torch2PIL', 'torch2np', 
    'reparameterize', 

    'ScheduledSampling', 'LinearScheduledSampling', 
    'SigmoidScheduledSampling', 
]

from typing import Callable, List, Optional
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from symmetry_transforms import SymmetryAssumption

from torchWork import *

TRAJ_FILENAME = 'trajectory.json'
RESOLUTION = 32
IMG_N_CHANNELS = 3
ENTIRE_DATASET_IN_DEVICE = True
SLOW_EVAL_EPOCH_INTERVAL = 1000

class HyperParams(BaseHyperParams):
    def __init__(self) -> None:
        super().__init__()

        self.symm: SymmetryAssumption = None
        self.K: int = 1
        
        self.I: int = None
        self.T: int = None
        self.R: int = None
        self.TR: int = None
        self.supervise_rnn: bool = None
        self.supervise_vae: bool = None
        self.supervise_vae_only_xy: bool = None

        self.variational_rnn: bool = None
        self.vvrnn: bool = None
        self.vvrnn_static: float = None
        self.rnn_min_context: int = None
        self.energy_noise_std: float = None

        self.rnn_width: int = None
        self.residual: bool = None
        self.jepa_stop_grad_encoder: bool = None
        self.dropout: float = None
        self.vae_channels: List[int] = None
        self.deep_spread: bool = None
        self.relu_leak: bool = None
        self.vae_kernel_size: int = None

        self.batch_size: int = None
        self.grad_clip: Optional[float] = None
        self.optim_name: str = None
        self.lr_diminish: Optional[Callable[
            [int, int], float, 
        ]] = None

        self.image_loss: str = None
        self.sched_sampling: Optional[ScheduledSampling] = None
        # Deprecated:
        self.teacher_forcing_duration: int = None

        self.train_set_size: int = None
        self.max_epoch: int = None

        self.imgCriterion: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor, 
        ] = None
        self.n_batches_per_epoch: int = None

    def fillDefaults(self):
        '''
        This is necessary when we want to load old 
        experiments (with less hyper params) without 
        checking out old commits.  
        The default values should guarantee old behaviors.  
        '''
        if self.jepa_stop_grad_encoder is None:
            self.jepa_stop_grad_encoder = False
        if self.supervise_vae_only_xy is None:
            self.supervise_vae_only_xy = False
        if self.energy_noise_std is None:
            self.energy_noise_std = 1
        if self.dropout is None:
            self.dropout = 0.0
        if self.relu_leak is None:
            self.relu_leak = True
        if self.vae_kernel_size is None:
            self.vae_kernel_size = 3
        if self.lr_diminish is None:
            self.lr_diminish = None
        if self.teacher_forcing_duration is None:
            self.sched_sampling = None
        if isinstance(self.teacher_forcing_duration, int):
            self.sched_sampling = LinearScheduledSampling(self.teacher_forcing_duration)
        if 'symm_self_consistency' not in self.lossWeightTree:
            self.lossWeightTree['symm_self_consistency'].weight = 0
        if 'seq_energy' not in self.lossWeightTree:
            self.lossWeightTree['seq_energy'] = LossWeightTree('seq_energy', 0, [
                LossWeightTree('real', 0, None), 
                LossWeightTree('fake', 0, None), 
            ])
    
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
        self.n_batches_per_epoch = self.train_set_size // self.batch_size
    
    def copyOneParam(self, k: str, v):
        if v is None:
            return True, None
        if k in ('imgCriterion', 'sched_sampling'):
            return True, v
        if k == 'symm':
            assert isinstance(v, SymmetryAssumption)
            return True, v.copy()
        return super().copyOneParam(k, v)
    
    @contextmanager
    def eval(self):
        # enter eval mode
        saved = self.sched_sampling
        self.sched_sampling = None
        try:
            yield
        finally:
            self.sched_sampling = saved

def torch2np(torchImg: torch.Tensor) -> np.ndarray:
    return (
        torchImg.cpu().detach().clamp(0, 1)
        .permute(1, 2, 0) * 255
    ).round().numpy().astype(np.uint8)

def torch2PIL(torchImg: torch.Tensor):
    return Image.fromarray(torch2np(torchImg), 'RGB')

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

class ScheduledSampling(metaclass=ABCMeta):
    @abstractmethod
    def get(self, epoch: int, hParams: HyperParams) -> float:
        raise NotImplemented

class LinearScheduledSampling(ScheduledSampling):
    def __init__(self, duration: int) -> None:
        self.duration = duration
    
    def get(self, epoch: int, hParams: HyperParams):
        try:
            return 1 - min(
                1, epoch / self.duration, 
            )
        except ZeroDivisionError:
            return 0
    
    def __repr__(self):
        return f'LinearScheduledSampling(duration={self.duration})'

class SigmoidScheduledSampling(ScheduledSampling):
    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta
    
    def get(self, epoch: int, hParams: HyperParams):
        xj_batch_i_approx = 4 * epoch
        return self.alpha / (self.alpha + np.exp(
            (xj_batch_i_approx + self.beta) / self.alpha, 
        ))
    
    def __repr__(self):
        return f'SigmoidScheduledSampling({self.alpha}, {self.beta})'
