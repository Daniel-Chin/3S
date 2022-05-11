from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 1
EXPERIMENTS = [
    ('16', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=16, 
        deep_spread=False, vae_channels=[16, 32, 64], 
    )), 
    ('32', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
    )), 
    ('64', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=64, 
        deep_spread=False, vae_channels=[16, 32, 64], 
    )), 
]
