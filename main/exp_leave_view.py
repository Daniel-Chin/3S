from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/bounce_leave_view/train'
VALIDATE_SET_PATH = '../datasets/bounce_leave_view/validate'
VALIDATE_SET_SIZE = 64
ACTUAL_DIM = 3

EXP_NAME = 'leave_view'
N_RAND_INITS = 4

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = '0'
        self.variable_value = 0
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

hP = HyperParams()
hP.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
    LossWeightTree('kld', 1e-5, None), 
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
])
hP.lr = 0.001
hP.symm = SymmetryAssumption(
    3, [
        ([Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        ([Trivial()], {Slice(2, 3)}), 
    ], 
)
hP.supervise_rnn = False
hP.supervise_vae = False
hP.variational_rnn = True
hP.vvrnn = False
hP.vvrnn_static = -25
hP.rnn_min_context = 4
hP.rnn_width = 16
hP.residual = True
hP.vae_channels = [16, 32, 64]
hP.deep_spread = False
hP.batch_size = 256
hP.grad_clip = .03
hP.optim_name = 'adam'
hP.train_set_size = 256
hP.image_loss = 'mse'
hP.teacher_forcing_duration = 30000
hP.max_epoch = hP.teacher_forcing_duration
hP.ready()
GROUPS.append(MyExpGroup(hP))
