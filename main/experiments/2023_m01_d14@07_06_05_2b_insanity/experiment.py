'''
Why are 
2023_m01_d12@20_20_40_2b_sps, and
2023_m01_d11@00_13_45_2b_weight_decay 
give inconsistent results?
'''
from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/two_body/train'
VALIDATE_SET_PATH = '../datasets/two_body/validate'
VALIDATE_SET_SIZE = 64
SEQ_LEN = 25
ACTUAL_DIM = 6

EXP_NAME = '2b_insanity'
N_RAND_INITS = 6

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'weight_decay,train_set_size'
        self.variable_value = (
            hyperParams.weight_decay, 
            hyperParams.train_set_size, 
        )
    
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
])
template.lr = 0.001
template.symm = SymmetryAssumption(
    6, [
        (SAMPLE_TRANS, [Translate(3, 1), Rotate(3)], {Slice(0, 3), Slice(3, 6)}), 
    ], 
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
template.residual = False
template.jepa_stop_grad_l_encoder = False
template.jepa_stop_grad_r_encoder = False
template.dropout = 0.0
template.rnn_ensemble = 1
template.vae_channels = [64, 128, 256]
template.deep_spread = True
template.relu_leak = False
template.vae_kernel_size = 4
template.vae_is_actually_ae = False
template.encoder_batch_norm = True
template.batch_size = 16
template.grad_clip = None
template.optim_name = 'adam'
template.weight_decay = 0   # 1e-6
template.lr_diminish = None
template.train_set_size = 64
template.sched_image_loss = ScheduledImageLoss((0, 'mse'))
template.sched_sampling = LinearScheduledSampling(18000)
template.max_epoch = template.sched_sampling.duration
template.vicreg_expander_identity = None
template.vicreg_expander_widths = None
template.vicreg_invariance_on_Y = None

# modifying template
# template.xxx = xxx

for wd, tss in [
    # (0, 64), 
    # (0, 1024), 
    (1e-9, 64), 
    (1e-9, 1024), 
]:
    hP = template.copy()
    hP.weight_decay = wd
    hP.train_set_size = tss
    hP.max_epoch = 16000 * 64 // tss
    hP.sched_sampling = LinearScheduledSampling(hP.max_epoch)
    hP.ready(globals())
    GROUPS.append(MyExpGroup(hP))
