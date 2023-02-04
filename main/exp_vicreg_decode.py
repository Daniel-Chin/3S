from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree, ExperimentGroup

from shared import *
from symmetry_transforms import *
from hyper_params import *
from dataset_definitions import bounceSingleColor as datasetDef

SLOW_EVAL_EPOCH_INTERVAL = 2000

EXP_NAME = 'vicreg_decode'
N_RAND_INITS = 6

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'nickname'
        self.variable_value = hyperParams.nickname
    
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
    3, [
        (SAMPLE_TRANS, [Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        (SAMPLE_TRANS, [Trivial()], {Slice(2, 3)}), 
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

# vicreg is different from template
vicreg = deepcopy(template)
vicreg.lossWeightTree['vicreg'].weight = 1
vicreg.lossWeightTree['vicreg']['variance'].weight = 25
vicreg.lossWeightTree['vicreg']['invariance'].weight = 25
vicreg.lossWeightTree['vicreg']['covariance'].weight = 1
vicreg.vicreg_expander_identity = False
vicreg.vicreg_expander_widths = [64, 64, 64]
vicreg.vicreg_invariance_on_Y = False
vicreg.vicreg_cross_traj = False
vicreg.weight_decay = 1e-9  # to tweak
# vicreg.batch_size = 32
# vicreg.lossWeightTree['self_recon'].weight = 0
vicreg.lossWeightTree['kld'].weight = 0
# vicreg.lossWeightTree['predict'].weight = 0
# vicreg.lossWeightTree['predict']['image'].weight = 0
vicreg.lossWeightTree['predict']['z'].weight = 0
vicreg.vae_is_actually_ae = True
vicreg.variational_rnn = False

# modify vicreg from vanilla
vicreg.lossWeightTree['vicreg']['variance'].weight = 35
vicreg.lossWeightTree['vicreg']['invariance'].weight = 25
vicreg.vicreg_expander_identity = True
vicreg.vicreg_expander_widths = None
# vicreg.train_set_size = 512
# vicreg.batch_size = 512
# vicreg.max_epoch = 32000
# vicreg.sched_sampling = LinearScheduledSampling(vicreg.max_epoch)

# modify vicreg to combine w/ decode
vicreg.vicreg_expander_identity = False
vicreg.vicreg_expander_widths = [64, 64, 64]

# for vw in (
#     3.84e-3 / 25, 
#     .0003, .001, .003
# ):

hP = deepcopy(vicreg)
hP.nickname = 'no_vicreg'
hP.lossWeightTree['vicreg'].weight = 0
hP.lossWeightTree['vicreg']['variance'].weight = 0
hP.lossWeightTree['vicreg']['invariance'].weight = 0
hP.lossWeightTree['vicreg']['covariance'].weight = 0
hP.vicreg_expander_identity = None
hP.vicreg_expander_widths = None
hP.vicreg_invariance_on_Y = None
hP.batch_size = 64
hP.sched_sampling = LinearScheduledSampling(18000 * 4)
hP.max_epoch = template.sched_sampling.duration
hP.ready(globals())
GROUPS.append(MyExpGroup(hP))

hP = deepcopy(vicreg)
hP.nickname = 'no_vicreg_bsize=16'
hP.lossWeightTree['vicreg'].weight = 0
hP.lossWeightTree['vicreg']['variance'].weight = 0
hP.lossWeightTree['vicreg']['invariance'].weight = 0
hP.lossWeightTree['vicreg']['covariance'].weight = 0
hP.vicreg_expander_identity = None
hP.vicreg_expander_widths = None
hP.vicreg_invariance_on_Y = None
hP.ready(globals())
GROUPS.append(MyExpGroup(hP))

hP = deepcopy(template)
hP.nickname = 'vanilla'
hP.ready(globals())
GROUPS.append(MyExpGroup(hP))
