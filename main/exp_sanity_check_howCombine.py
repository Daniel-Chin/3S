from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/bounce/train'
VALIDATE_SET_PATH = '../datasets/bounce/validate'
VALIDATE_SET_SIZE = 64
SEQ_LEN = 20
ACTUAL_DIM = 3

EXP_NAME = 'sanity_check_howCombine'
N_RAND_INITS = 30

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'symm'
        self.variable_value = hyperParams.symm
    
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
    LossWeightTree('symm_self_consistency', 0, None), 
])
template.lr = 0.001
template.symm = SymmetryAssumption(
    3, [
        (SAMPLE_TRANS, [Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        (SAMPLE_TRANS, [Trivial()], {Slice(2, 3)}), 
    ], 
)
template.supervise_rnn = False
template.supervise_vae = False
template.supervise_vae_only_xy = False
template.variational_rnn = True
template.vvrnn = False
template.vvrnn_static = -25
template.rnn_min_context = 5
template.energy_noise_std = 1
template.rnn_width = 32
template.residual = False
template.jepa_stop_grad_l_encoder = False
template.jepa_stop_grad_r_encoder = False
template.dropout = 0.0
template.vae_channels = [64, 128, 256]
template.deep_spread = True
template.relu_leak = False
template.vae_kernel_size = 4
template.batch_size = 16
template.grad_clip = None
template.optim_name = 'adam'
template.lr_diminish = None
template.train_set_size = 128
template.image_loss = 'mse'
template.sched_sampling = LinearScheduledSampling(9000)
template.max_epoch = template.sched_sampling.duration

# modifying template
# template.xxx = xxx

hP = template.copy()
hP.symm = GusMethod()
hP.ready()
GROUPS.append(MyExpGroup(hP))

hP = template.copy()
hP.symm = SymmetryAssumption(
    3, [
        (SAMPLE_TRANS, [Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        (SAMPLE_TRANS, [Trivial()], {Slice(2, 3)}), 
    ], 
)
hP.ready()
GROUPS.append(MyExpGroup(hP))

assert len(GROUPS) == 2
