from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/bounce/train'
VALIDATE_SET_PATH = '../datasets/bounce/validate'
VALIDATE_SET_SIZE = 64
SEQ_LEN = 20
ACTUAL_DIM = 3

EXP_NAME = 'self_recon_weight'
N_RAND_INITS = 4

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'self_recon_w'
        self.variable_value = hyperParams.lossWeightTree[
            'self_recon'
        ].weight
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', None, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('seq_energy', 0, [
        LossWeightTree('real', 0, None), 
        LossWeightTree('fake', 0, None), 
    ]), 
    LossWeightTree('predict', 1, [
        LossWeightTree('z', 0, None), 
        LossWeightTree('image', 1, None), 
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
        ([Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        ([Trivial()], {Slice(2, 3)}), 
    ], 
)
template.supervise_rnn = False
template.supervise_vae = False
template.supervise_vae_only_xy = False
template.variational_rnn = True
template.vvrnn = False
template.vvrnn_static = -25
template.rnn_min_context = 4
template.energy_noise_std = 1
template.rnn_width = 16
template.residual = True
template.jepa_stop_grad_encoder = False
template.dropout = 0.0
template.vae_channels = [16, 32, 64]
template.deep_spread = False
template.relu_leak = True
template.vae_kernel_size = 4
template.batch_size = 128
template.grad_clip = 1
template.optim_name = 'adam'
template.lr_diminish = None
template.train_set_size = 128
template.image_loss = 'mse'
template.sched_sampling = LinearScheduledSampling(40000)
template.max_epoch = template.sched_sampling.duration
template.ready()

# modifying template
# template.xxx = xxx

hP = template.copy()
hP.lossWeightTree[
    'self_recon'
].weight = 1
hP.ready()
GROUPS.append(MyExpGroup(hP))

hP = template.copy()
hP.lossWeightTree[
    'self_recon'
].weight = 2
hP.ready()
GROUPS.append(MyExpGroup(hP))

hP = template.copy()
hP.lossWeightTree[
    'self_recon'
].weight = 3
hP.ready()
GROUPS.append(MyExpGroup(hP))

assert len(GROUPS) == 3
