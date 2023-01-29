__all__ = [
    'TRAJ_FILENAME', 
    'RESOLUTION', 
    'IMG_N_CHANNELS', 
    'ENTIRE_DATASET_IN_DEVICE', 
    
    'HyperParams', 'torch2PIL', 'torch2np', 
    'reparameterize', 

    'ScheduledSampling', 'LinearScheduledSampling', 
    'SigmoidScheduledSampling', 'ScheduledImageLoss', 
]

from typing import Callable, List, Optional, Dict, Tuple
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from PIL import Image
from symmetry_transforms import SymmetryAssumption

from torchWork import *

TRAJ_FILENAME = 'trajectory.json'
RESOLUTION = 32
IMG_N_CHANNELS = 3
ENTIRE_DATASET_IN_DEVICE = True

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
        self.vvrnn_static: Optional[float] = None
        self.rnn_min_context: int = None
        self.energy_noise_std: float = None

        self.rnn_width: int = None
        self.rnn_depth: int = None
        self.residual: Optional[float] = None
        self.jepa_stop_grad_l_encoder: bool = None
        self.jepa_stop_grad_r_encoder: bool = None
        # "l"eft, "r"ight refers to Fig.1 of SimSiam (Chen & He, 2020).  
        self.dropout: float = None
        self.rnn_ensemble: int = None

        self.vae_channels: List[int] = None
        self.deep_spread: bool = None
        self.relu_leak: bool = None
        self.vae_kernel_size: int = None
        self.vae_is_actually_ae: bool = None
        self.encoder_batch_norm: bool = None

        self.batch_size: int = None
        self.grad_clip: Optional[float] = None
        self.optim_name: str = None
        self.lr_diminish: Optional[Callable[
            [int, int], float, 
        ]] = None

        self.sched_image_loss: ScheduledImageLoss = None
        self.sched_sampling: Optional[ScheduledSampling] = None
        self.teacher_forcing_duration: int = None   # Deprecated

        self.train_set_size: int = None
        self.max_epoch: int = None

        self.vicreg_expander_identity: bool = None
        self.vicreg_expander_widths: List[int] = None
        self.vicreg_invariance_on_Y: bool = None
        # That "Y" follows the symbols defined in the vicreg paper. 
        # In this source code, vicreg's Y is `z`, 
        # and vicreg's Z is `emb_l` and `emb_r`. 
        self.vicreg_cross_traj: bool = None

        
        self.imgCriterion: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor, 
        ] = None    # Deprecated
        self.n_batches_per_epoch: int = None
        self.vicreg_emb_dim: int = None

        self.experiment_globals: Dict = None

    def fillDefaults(self):
        '''
        This is necessary when we want to load old 
        experiments (with less hyper params) without 
        checking out old commits.  
        The default values should guarantee old behaviors.  
        '''
        try:
            if self.jepa_stop_grad_encoder is None:
                raise AttributeError
        except AttributeError:
            self.jepa_stop_grad_encoder = False
        if self.jepa_stop_grad_l_encoder is None:
            self.jepa_stop_grad_l_encoder = self.jepa_stop_grad_encoder
        if self.jepa_stop_grad_r_encoder is None:
            self.jepa_stop_grad_r_encoder = self.jepa_stop_grad_encoder
        del self.jepa_stop_grad_encoder
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
        if isinstance(self.teacher_forcing_duration, int):
            self.sched_sampling = LinearScheduledSampling(self.teacher_forcing_duration)
        if 'symm_self_consistency' not in self.lossWeightTree:
            self.lossWeightTree['symm_self_consistency'].weight = 0
        if 'seq_energy' not in self.lossWeightTree:
            self.lossWeightTree['seq_energy'] = LossWeightTree('seq_energy', 0, [
                LossWeightTree('real', 0, None), 
                LossWeightTree('fake', 0, None), 
            ])
        if 'vicreg' not in self.lossWeightTree:
            self.lossWeightTree['vicreg'] = LossWeightTree('vicreg', 0, [
                LossWeightTree('variance', 0, None), 
                LossWeightTree('invariance', 0, None), 
                LossWeightTree('covariance', 0, None), 
            ])
            self.vicreg_expander_identity = None
            self.vicreg_expander_widths = None
            self.vicreg_invariance_on_Y = None
        if self.rnn_depth is None:
            self.rnn_depth = 1

    def ready(self, experiment_globals):
        self.experiment_globals = experiment_globals
        assert self.lr_diminish is None # See below comments
        # The current implementation of lr_diminish is via loss scaling. 
        # In Adam, that is not equivalent to lr scaling. Use pytorch's lr sched instead. 
        assert self.supervise_rnn == (
            self.lossWeightTree['supervise']['rnn'].weight != 0
        )
        assert self.supervise_vae == (
            self.lossWeightTree['supervise']['vae']['encode'].weight != 0 or
            self.lossWeightTree['supervise']['vae']['decode'].weight != 0
        )
        assert not (self.vvrnn and (self.vvrnn_static is not None))
        if self.supervise_rnn:
            assert self.lossWeightTree['predict']['z'].weight != 0
        if self.lossWeightTree['vicreg'].weight:
            if self.vicreg_expander_identity:
                assert self.vicreg_expander_widths is None
            assert self.lossWeightTree['predict']['z'].weight == 0
            assert not self.jepa_stop_grad_l_encoder
            assert not self.jepa_stop_grad_r_encoder
        else:
            assert self.vicreg_expander_identity is None
            assert self.vicreg_expander_widths is None
            assert self.vicreg_invariance_on_Y is None
            assert self.lossWeightTree['vicreg']['variance'].weight == 0
            assert self.lossWeightTree['vicreg']['invariance'].weight == 0
            assert self.lossWeightTree['vicreg']['covariance'].weight == 0
        if self.vae_is_actually_ae:
            assert self.lossWeightTree['kld'].weight == 0
        if self.variational_rnn:
            assert not self.vae_is_actually_ae
        x = self.batch_size / self.K
        assert abs(x - int(x)) < 1e-6
        self.OptimClass = {
            'adam': torch.optim.Adam, 
        }[self.optim_name]
        self.n_batches_per_epoch = self.train_set_size // self.batch_size
        if self.lossWeightTree['vicreg'].weight != 0:
            if self.vicreg_expander_identity:
                self.vicreg_emb_dim = self.symm.latent_dim
            else:
                self.vicreg_emb_dim = self.vicreg_expander_widths[-1]
    
    def copyOneParam(self, k: str, v):
        if v is None:
            return True, None
        if k in ('sched_image_loss', 'sched_sampling'):
            return True, v
        if k == 'symm':
            assert isinstance(v, SymmetryAssumption)
            return True, v.copy()
        return super().copyOneParam(k, v)
    
    @contextmanager
    def eval(self):
        # enter eval mode
        saved_sched_sampling = self.sched_sampling
        self.sched_sampling = None
        saved_batch_size = self.batch_size
        self.batch_size = self.experiment_globals['VALIDATE_SET_SIZE']
        saved_vicreg_emb_dim = self.vicreg_emb_dim
        if self.vicreg_emb_dim is None:
            self.vicreg_emb_dim = self.symm.latent_dim
        try:
            yield
        finally:
            self.sched_sampling = saved_sched_sampling
            self.batch_size = saved_batch_size
            self.vicreg_emb_dim = saved_vicreg_emb_dim

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

class ScheduledImageLoss:
    def __init__(self, *schedule: Tuple[int, str]) -> None:
        self.schedule = schedule
    
    def __repr__(self):
        return '__'.join([f'{epoch}_{loss_name}' for (epoch, loss_name) in self.schedule])
    
    def get(self, epoch: int) -> Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor, 
    ]:
        loss_name = None
        for _epoch, _loss_name in self.schedule:
            if _epoch > epoch:
                break
            loss_name = _loss_name
        return {
            'mse': torch.nn.MSELoss(), 
            'l1' : torch.nn.L1Loss(), 
            'bce': torch.nn.BCELoss(), 
        }[loss_name]
