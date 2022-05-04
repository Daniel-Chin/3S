from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 5
EXPERIMENTS = [
    ('AE', Config(
        Constant(0), 1, 0, do_symmetry=False, variational_rnn=False, 
        rnn_width=16, deep_spread=False, 
    )), 
    # ('AE+RNN+symm', Config(
    #     Constant(0), 1, 1, do_symmetry=True, variational_rnn=False, 
    #     rnn_width=16, deep_spread=False, 
    # )), 
    # ('VAE+RNN+symm', Config(
    #     Constant(0.001), 1, 1, do_symmetry=True, variational_rnn=False, 
    #     rnn_width=16, deep_spread=False, 
    # )), 
    # ('VAE+VRNN+symm', Config(
    #     Constant(0.001), 1, 1, do_symmetry=True, variational_rnn=True, 
    #     rnn_width=16, deep_spread=False, 
    # )), 
]
