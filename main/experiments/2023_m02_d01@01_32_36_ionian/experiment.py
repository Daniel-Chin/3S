from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree, ExperimentGroup

from shared import *
from symmetry_transforms import *
from dataset_instances import IonianScales_fr3gm as DATASET_INSTANCE
from load_dataset import MusicDataset

def getDataset(
    is_train_not_validate: bool, size: Optional[int], device, 
):
    dataset = MusicDataset(
        DATASET_INSTANCE.songBox, 
        DATASET_INSTANCE.config, 
        is_train_not_validate, size, device, 
    )
    if not is_train_not_validate:
        assert DATASET_INSTANCE.VALIDATE_SET_SIZE == dataset.size
    return dataset

SLOW_EVAL_EPOCH_INTERVAL = 30

EXP_NAME = 'ionian'
N_RAND_INITS = 8

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = '0'
        self.variable_value = 0
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
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
    1, [
        (SAMPLE_TRANS, [Translate(1, 1)], {Slice(0, 1)}), 
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
template.signal_resolution = (
    DATASET_INSTANCE.config.N_BINS, 
    DATASET_INSTANCE.config.ENCODE_STEP, 
)
template.signal_n_channels = DATASET_INSTANCE.IMG_N_CHANNELS
template.vae_channels = [64, 64, 128, 128]
template.vae_kernel_sizes = [
    (5, DATASET_INSTANCE.config.ENCODE_STEP), 
    (8, 1), 
    (4, 1), 
    (4, 1), 
]
template.vae_strides = [2, 4, 2, 2]
template.vae_paddings = [
    (1, 0), 
    (2, 0), 
    (1, 0), 
    (1, 0), 
]
template.vae_fc_before_decode = [16, 64, 256, 1024]
template.relu_leak = False
template.vae_is_actually_ae = False
template.encoder_batch_norm = True
template.batch_size = 16
template.grad_clip = None
template.optim_name = 'adam'
template.weight_decay = 0
template.lr_diminish = None
template.train_set_size = None
template.sched_image_loss = ScheduledImageLoss((0, 'mse'))
template.sched_sampling = LinearScheduledSampling(600)
template.max_epoch = template.sched_sampling.duration
template.vicreg_expander_identity = None
template.vicreg_expander_widths = None
template.vicreg_invariance_on_Y = None
template.vicreg_cross_traj = None

# modifying template
# template.xxx = xxx

hP = deepcopy(template)
hP.ready(globals())
GROUPS.append(MyExpGroup(hP))
