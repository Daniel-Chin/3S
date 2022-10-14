from functools import lru_cache
from torchWork import LossWeightTree, ExperimentGroup, DEVICE

from shared import *

EXP_NAME = 'supervised_rnn_width'
N_RAND_INITS = 2

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams
    
    @lru_cache(1)
    def name(self):
        return f'rnn_width={self.hyperParams.rnn_width}'

GROUPS = []

hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 0, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 0, [
        LossWeightTree('z', 0, None), 
        LossWeightTree('image', 0, None), 
    ]), 
    LossWeightTree('supervise', 1, [
        LossWeightTree('rnn', 1, None), 
        LossWeightTree('vae', 1, [
            LossWeightTree('encode', 1, None), 
            LossWeightTree('decode', 1, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = True
hP.supervise_vae = True
hP.variational_rnn = True
hP.vvrnn = False
hP.vvrnn_static = -25
hP.rnn_min_context = 4
hP.rnn_width = 32
hP.residual = True
hP.vae_channels = [16, 32, 64]
hP.deep_spread = False
hP.batch_size = 128
hP.grad_clip = .03
hP.optim_name = 'adam'
hP.image_loss = 'mse'
hP.train_set_size = 256
hP.teacher_forcing_duration = 0
hP.ready()

hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 0, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 0, [
        LossWeightTree('z', 0, None), 
        LossWeightTree('image', 0, None), 
    ]), 
    LossWeightTree('supervise', 1, [
        LossWeightTree('rnn', 1, None), 
        LossWeightTree('vae', 1, [
            LossWeightTree('encode', 1, None), 
            LossWeightTree('decode', 1, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = True
hP.supervise_vae = True
hP.variational_rnn = True
hP.vvrnn = False
hP.vvrnn_static = -25
hP.rnn_min_context = 4
hP.rnn_width = 64
hP.residual = True
hP.vae_channels = [16, 32, 64]
hP.deep_spread = False
hP.batch_size = 128
hP.grad_clip = .03
hP.optim_name = 'adam'
hP.image_loss = 'mse'
hP.train_set_size = 256
hP.teacher_forcing_duration = 0
hP.ready()

hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 0, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 0, [
        LossWeightTree('z', 0, None), 
        LossWeightTree('image', 0, None), 
    ]), 
    LossWeightTree('supervise', 1, [
        LossWeightTree('rnn', 1, None), 
        LossWeightTree('vae', 1, [
            LossWeightTree('encode', 1, None), 
            LossWeightTree('decode', 1, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = True
hP.supervise_vae = True
hP.variational_rnn = True
hP.vvrnn = False
hP.vvrnn_static = -25
hP.rnn_min_context = 4
hP.rnn_width = 128
hP.residual = True
hP.vae_channels = [16, 32, 64]
hP.deep_spread = False
hP.batch_size = 128
hP.grad_clip = .03
hP.optim_name = 'adam'
hP.image_loss = 'mse'
hP.train_set_size = 256
hP.teacher_forcing_duration = 0
hP.ready()
