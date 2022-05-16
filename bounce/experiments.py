from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 3
# EXPERIMENTS = [
#     ('-10', Config(
#         Constant(1e-5), 1, 1, do_symmetry=True, 
#         variational_rnn=True, rnn_width=32, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-10, 
#     )), 
#     ('-7', Config(
#         Constant(1e-5), 1, 1, do_symmetry=True, 
#         variational_rnn=True, rnn_width=32, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-7, 
#     )), 
#     ('-4', Config(
#         Constant(1e-5), 1, 1, do_symmetry=True, 
#         variational_rnn=True, rnn_width=32, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-4, 
#     )), 
#     ('-2', Config(
#         Constant(1e-5), 1, 1, do_symmetry=True, 
#         variational_rnn=True, rnn_width=32, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-2, 
#     )), 
# ]

# EXPERIMENTS = [
#     ('3', Config(
#         Constant(1e-5), 1, 1, do_symmetry=True, 
#         variational_rnn=True, rnn_width=32, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-20, rnn_min_context=3, 
#     )), 
#     ('5', Config(
#         Constant(1e-5), 1, 1, do_symmetry=True, 
#         variational_rnn=True, rnn_width=32, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-20, rnn_min_context=5, 
#     )), 
# ]

EXPERIMENTS = [
    ('0:2', Config(
        Constant(1e-5), 0, 2, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-20, rnn_min_context=5, 
    )), 
    # ('1:1', Config(
    #     Constant(1e-5), 1, 1, do_symmetry=True, 
    #     variational_rnn=True, rnn_width=32, 
    #     deep_spread=False, vae_channels=[16, 32, 64], 
    #     vvrnn=False, vvrnn_static=-20, rnn_min_context=5, 
    # )), 
]
