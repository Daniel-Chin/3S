from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 4
EXPERIMENTS = [
    ('nothing', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
    )), 
    ('deep spread', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=True, vae_channels=[16, 32, 64], 
    )), 
    ('deep spread + strong vae', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=True, vae_channels=[64, 128, 256], 
    )), 
]
