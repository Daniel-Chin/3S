from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 1

EXPERIMENTS = [
    ('0', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=0, 
    )), 
    ('1e2', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e2, 
    )), 
    ('1e3', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e3, 
    )), 
    ('1e4', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e4, 
    )), 
    ('1e5', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e5, 
    )), 
    ('1e6', Config(
        Constant(1e-5), 1, 1, do_symmetry=True, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e6, 
    )), 
]
