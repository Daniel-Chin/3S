from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/bounce/train'
VALIDATE_SET_PATH = '../datasets/bounce/validate'
VALIDATE_SET_SIZE = 64
ACTUAL_DIM = 3

EXP_NAME = 'tf_time'
N_RAND_INITS = 1

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'tf_time'
        self.variable_value = hyperParams.teacher_forcing_duration
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
    LossWeightTree('seq_energy', 0, None), 
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
template.rnn_width = 16
template.residual = True
template.jepa_stop_grad_encoder = False
template.vae_channels = [16, 32, 64]
template.deep_spread = False
template.batch_size = 256
template.grad_clip = 1
template.optim_name = 'adam'
template.train_set_size = 256
template.image_loss = 'mse'
template.teacher_forcing_duration = None
template.max_epoch = template.teacher_forcing_duration
template.ready()

# modifying template
# template.xxx = xxx

TFs = [*range(10000, 40001, 5000)]
for tf in TFs:
    hP = template.copy()
    hP.teacher_forcing_duration = tf
    hP.max_epoch = hP.teacher_forcing_duration
    hP.ready()
    GROUPS.append(MyExpGroup(hP))

assert len(GROUPS) == len(TFs)
