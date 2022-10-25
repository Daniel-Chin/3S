from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/bounce/train'
VALIDATE_SET_PATH = '../datasets/bounce/validate'
VALIDATE_SET_SIZE = 64
ACTUAL_DIM = 3

EXP_NAME = 'supvise_vae'
N_RAND_INITS = 3

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'kld_weight'
        self.variable_value = hyperParams.lossWeightTree[
            'kld'
        ].weight
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 0, None), 
    LossWeightTree('kld', None, None), 
    LossWeightTree('predict', 0, [
        LossWeightTree('z', 0, None), 
        LossWeightTree('image', 0, None), 
    ]), 
    LossWeightTree('supervise', 1, [
        LossWeightTree('rnn', 0, None), 
        LossWeightTree('vae', 1, [
            LossWeightTree('encode', 1, None), 
            LossWeightTree('decode', 1, None), 
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
template.supervise_vae = True
template.supervise_vae_only_xy = False
template.variational_rnn = True
template.vvrnn = False
template.vvrnn_static = -25
template.rnn_min_context = 19    # skip
template.rnn_width = 16
template.residual = True
template.jepa_stop_grad_encoder = True
template.vae_channels = [16, 32, 64]
template.deep_spread = False
template.batch_size = 256
template.grad_clip = .03
template.optim_name = 'adam'
template.train_set_size = 256
template.image_loss = 'mse'
template.teacher_forcing_duration = 20000
template.max_epoch = template.teacher_forcing_duration
template.ready()

# modifying template
# template.xxx = xxx

hP = template.copy()
hP.lossWeightTree['kld'].weight = 0
hP.ready()
GROUPS.append(MyExpGroup(hP))

hP = template.copy()
hP.lossWeightTree['kld'].weight = 0.005
hP.ready()
GROUPS.append(MyExpGroup(hP))

assert len(GROUPS) == 2
