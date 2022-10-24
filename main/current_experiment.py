from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/bounce/train'
VALIDATE_SET_PATH = '../datasets/bounce/validate'
VALIDATE_SET_SIZE = 64
ACTUAL_DIM = 3

EXP_NAME = 'test'
N_RAND_INITS = 1

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'ssc_loss_weight'
        self.variable_value = (
            hyperParams.lossWeightTree['symm_self_consistency']
        )
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
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
template.variational_rnn = True
template.vvrnn = False
template.vvrnn_static = -25
template.rnn_min_context = 4
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
template.teacher_forcing_duration = 40000
template.max_epoch = template.teacher_forcing_duration
template.ready()

# Modifying template
template.jepa_stop_grad_encoder = False

hP = template.copy()
hP.lossWeightTree['symm_self_consistency'].weight = 1
hP.ready()
GROUPS.append(MyExpGroup(hP))

hP = template.copy()
hP.lossWeightTree['symm_self_consistency'].weight = .1
hP.ready()
GROUPS.append(MyExpGroup(hP))

hP = template.copy()
hP.lossWeightTree['symm_self_consistency'].weight = .01
hP.ready()
GROUPS.append(MyExpGroup(hP))

hP = template.copy()
hP.lossWeightTree['symm_self_consistency'].weight = .001
hP.ready()
GROUPS.append(MyExpGroup(hP))

assert len(GROUPS) == 4
