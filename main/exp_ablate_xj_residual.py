from functools import lru_cache
from symmetry_transforms import *
from torchWork import LossWeightTree, ExperimentGroup

from shared import *

TRAIN_SET_PATH    = '../datasets/xj_bounce/train'
VALIDATE_SET_PATH = '../datasets/xj_bounce/validate'
VALIDATE_SET_SIZE = 64
SEQ_LEN = 20
ACTUAL_DIM = 3

EXP_NAME = 'ablate_xj_residual'
N_RAND_INITS = 8

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'xj_new_dan'
        self.variable_value = hyperParams.xj_new_dan
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('self_recon', 1, None), 
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

xJ = template.copy()
xJ.xj_new_dan = 0
# 0: xj, 1: new (ablate), 2: dan. 
hP = template.copy()
hP.xj_new_dan = 1

xJ.rnn_width = 256
hP.rnn_width = 256

xJ.vae_channels = [64, 128, 256]
hP.vae_channels = [64, 128, 256]

xJ.deep_spread = True
hP.deep_spread = True

xJ.relu_leak = False
hP.relu_leak = False

xJ.batch_size = 32
hP.batch_size = 32

xJ.lossWeightTree['self_recon'].weight = (
    RESOLUTION ** 2 * SEQ_LEN * xJ.batch_size
)
hP.lossWeightTree['self_recon'].weight = (
    RESOLUTION ** 2 * SEQ_LEN * xJ.batch_size
)
xJ.lossWeightTree['kld'].weight = 0.01 * xJ.batch_size
hP.lossWeightTree['kld'].weight = 0.01 * xJ.batch_size
xJ.lossWeightTree['predict']['z'].weight = (
    2 * xJ.symm.latent_dim * SEQ_LEN * xJ.batch_size
)
hP.lossWeightTree['predict']['z'].weight = (
    2 * hP.symm.latent_dim * SEQ_LEN * xJ.batch_size
)
xJ.lossWeightTree['predict']['image'].weight = (
    2 * RESOLUTION ** 2 * SEQ_LEN * xJ.batch_size
)
hP.lossWeightTree['predict']['image'].weight = (
    2 * RESOLUTION ** 2 * SEQ_LEN * xJ.batch_size
)

xJ.K = 2    # one for translate, one for rotate
hP.K = 2    # one for translate, one for rotate

xJ.symm = GusMethod()
hP.symm = GusMethod()

def f(epoch, batch_i, hParams: HyperParams):
    return 0.99999 ** (epoch * hParams.n_batches_per_epoch + batch_i)
xJ.lr_diminish = f
hP.lr_diminish = f

xJ.grad_clip = None
hP.grad_clip = None

xJ.rnn_min_context = 5
hP.rnn_min_context = 5

xJ.sched_sampling = SigmoidScheduledSampling(alpha=2200, beta=8000)
hP.sched_sampling = SigmoidScheduledSampling(alpha=2200, beta=8000)

xJ.max_epoch = 150001 // xJ.batch_size
hP.max_epoch = 150001 // xJ.batch_size

xJ.residual = False
# ablated

xJ.image_loss = 'bce'
hP.image_loss = 'bce'

hP.ready()
GROUPS.append(MyExpGroup(hP))
xJ.ready()
GROUPS.append(MyExpGroup(xJ))

assert len(GROUPS) == 2
