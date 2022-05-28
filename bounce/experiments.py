from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 2

EXPERIMENTS = [
    ('.02', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.02, 
    )), 
]
