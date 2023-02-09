from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree, ExperimentGroup

from shared import *
from symmetry_transforms import *
from hyper_params import *
from dataset_definitions import twoBody as datasetDef

SLOW_EVAL_EPOCH_INTERVAL = 100

EXP_NAME = '2b_sps'
N_RAND_INITS = 8

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'symm'
        self.variable_value = (
            'yes' if hyperParams.symm == ours.symm else 'no', 
            hyperParams.train_set_size, 
        )[0]
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.datasetDef = datasetDef
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1.31072, None), 
    LossWeightTree('kld', 3.2e-7, None), 
    LossWeightTree('seq_energy', 0, [
        LossWeightTree('real', 0, None), 
        LossWeightTree('fake', 0, None), 
    ]), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', 3.84e-3, None), 
        LossWeightTree('image', 2.62144, None), 
    ]), 
    LossWeightTree('supervise', 0, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 0, [
            LossWeightTree('encode', 0, None), 
            LossWeightTree('decode', 0, None), 
        ]), 
    ]), 
    LossWeightTree('vicreg', 0, [
        LossWeightTree('variance', 0, None), 
        LossWeightTree('invariance', 0, None), 
        LossWeightTree('covariance', 0, None), 
    ]), 
    LossWeightTree('symm_self_consistency', 0, None), 
    LossWeightTree('cycle', 0, None), 
])
template.lr = 0.001
template.symm = SymmetryAssumption(
    6, [
        (SAMPLE_TRANS, [Translate(3, 1), Rotate(3)], {Slice(0, 3), Slice(3, 6)}), 
    ], 
    .1, 
)
template.supervise_rnn = False
template.supervise_vae = False
template.supervise_vae_only_xy = False
template.variational_rnn = True
template.vvrnn = False
template.vvrnn_static = None
template.rnn_min_context = 5
template.energy_noise_std = 1
template.rnn_width = 32
template.rnn_depth = 1
template.residual = False
template.jepa_stop_grad_l_encoder = False
template.jepa_stop_grad_r_encoder = False
template.dropout = 0.0
template.rnn_ensemble = 1
template.vae_channels = [64, 128, 256]
template.vae_kernel_sizes = [4, 4, 4]
template.vae_strides = [2, 2, 2]
template.vae_paddings = [1, 1, 1]
template.vae_fc_before_decode = [16, 32, 64]
template.relu_leak = False
template.vae_is_actually_ae = False
template.encoder_batch_norm = True
template.batch_size = 16
template.grad_clip = None
template.optim_name = 'adam'
template.weight_decay = 0
template.lr_diminish = None
template.train_set_size = 64
template.sched_image_loss = ScheduledImageLoss((0, 'mse'))
template.sched_sampling = LinearScheduledSampling(18000)
template.max_epoch = template.sched_sampling.duration
template.vicreg_expander_identity = None
template.vicreg_expander_widths = None
template.vicreg_invariance_on_Y = None
template.vicreg_cross_traj = None

# modifying template
template.rnn_width = 128
template.train_set_size = 512
template.max_epoch = 8000
template.sched_sampling = LinearScheduledSampling(template.max_epoch)
template.batch_size = 64
template.weight_decay = 0
SLOW_EVAL_EPOCH_INTERVAL = 2000

ours = deepcopy(template)
ours.symm = SymmetryAssumption(
    3, [
        (SAMPLE_TRANS, [Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        (SAMPLE_TRANS, [Trivial()], {Slice(2, 3)}), 
    ], 
    .1, 
)

baseline = deepcopy(template)
baseline.symm = SymmetryAssumption(
    3, [
        (SAMPLE_TRANS, [Trivial()], {Slice(0, 3)}), 
    ], 
    .1, 
)

for s in (
    ours, 
    baseline, 
):
    # for tss in (64, 128, 256, 512):
        tss = 256
        hP = deepcopy(template)
        hP.symm = s.symm
        hP.train_set_size = tss
        hP.max_epoch = 512 * 8000 // tss
        hP.sched_sampling = LinearScheduledSampling(hP.max_epoch)

        hP.ready(globals())
        GROUPS.append(MyExpGroup(hP))
