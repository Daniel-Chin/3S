from functools import lru_cache
from torchWork import LossWeightTree, ExperimentGroup, DEVICE

from shared import *

EXP_NAME = 'teacher_forcing_duration'
N_RAND_INITS = 1

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams
    
    @lru_cache(1)
    def name(self):
        x = self.hyperParams.teacher_forcing_duration
        return f'''teacher_f_time={x:.1E}'''

GROUPS = []

hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', .005, None), 
        LossWeightTree('image', 1, None), 
    ]), 
    LossWeightTree('supervise', 0, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 0, [
            LossWeightTree('encode', 0, None), 
            LossWeightTree('decode', 0, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = False
hP.supervise_vae = False
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
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', .005, None), 
        LossWeightTree('image', 1, None), 
    ]), 
    LossWeightTree('supervise', 0, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 0, [
            LossWeightTree('encode', 0, None), 
            LossWeightTree('decode', 0, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = False
hP.supervise_vae = False
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
hP.teacher_forcing_duration = 1e2
hP.ready()

hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', .005, None), 
        LossWeightTree('image', 1, None), 
    ]), 
    LossWeightTree('supervise', 0, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 0, [
            LossWeightTree('encode', 0, None), 
            LossWeightTree('decode', 0, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = False
hP.supervise_vae = False
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
hP.teacher_forcing_duration = 1e3
hP.ready()

hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', .005, None), 
        LossWeightTree('image', 1, None), 
    ]), 
    LossWeightTree('supervise', 0, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 0, [
            LossWeightTree('encode', 0, None), 
            LossWeightTree('decode', 0, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = False
hP.supervise_vae = False
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
hP.teacher_forcing_duration = 1e4
hP.ready()


hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', .005, None), 
        LossWeightTree('image', 1, None), 
    ]), 
    LossWeightTree('supervise', 0, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 0, [
            LossWeightTree('encode', 0, None), 
            LossWeightTree('decode', 0, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = False
hP.supervise_vae = False
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
hP.teacher_forcing_duration = 1e5
hP.ready()


hP = HyperParams()
GROUPS.append(MyExpGroup(hP))
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', .005, None), 
        LossWeightTree('image', 1, None), 
    ]), 
    LossWeightTree('supervise', 0, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 0, [
            LossWeightTree('encode', 0, None), 
            LossWeightTree('decode', 0, None), 
        ]), 
    ]), 
])
hP.lr = 0.001
hP.I = 1
hP.T = 0
hP.R = 0
hP.TR = 0
hP.supervise_rnn = False
hP.supervise_vae = False
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
hP.teacher_forcing_duration = 1e6
hP.ready()
