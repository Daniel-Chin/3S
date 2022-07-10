from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 3

EXPERIMENTS = [
    ('TR', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=1, I=0, lr=0.001, residual=True, 
        grad_clip=.03, 
    )), 
    ('T=1 R=1', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=1, R=1, TR=0, I=0, lr=0.001, residual=True, 
        grad_clip=.03, 
    )), 
    ('T=4 R=4 I=1', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=4, R=4, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, 
    )), 
]

EXPERIMENTS = [
    ('-5', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=1, I=0, lr=0.001, residual=True, 
        grad_clip=.03, 
    )), 
    ('-4', Config(
        Constant(1e-4), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=1, I=0, lr=0.001, residual=True, 
        grad_clip=.03, 
    )), 
    ('-3', Config(
        Constant(1e-3), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=1, I=0, lr=0.001, residual=True, 
        grad_clip=.03, 
    )), 
    ('-2', Config(
        Constant(1e-3), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=1, I=0, lr=0.001, residual=True, 
        grad_clip=.03, 
    )), 
]
