from train import Config
from betaSched import Constant, WarmUp, Osc

RAND_INIT_TIMES = 2

# Symmetry is off!

EXPERIMENTS = [
    ('0', Config(
        Constant(1e-5), 1, 1, do_symmetry=False, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=0, 
    )), 
    ('1e2', Config(
        Constant(1e-5), 1, 1, do_symmetry=False, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e2, 
    )), 
    ('1e3', Config(
        Constant(1e-5), 1, 1, do_symmetry=False, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e3, 
    )), 
    ('1e4', Config(
        Constant(1e-5), 1, 1, do_symmetry=False, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e4, 
    )), 
    ('1e5', Config(
        Constant(1e-5), 1, 1, do_symmetry=False, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e5, 
    )), 
    ('1e6', Config(
        Constant(1e-5), 1, 1, do_symmetry=False, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=.005, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=1e6, 
    )), 
]
# supervised rnn
# EXPERIMENTS = [
#     ('.005', Config(
#         beta=Constant(1e-5), vae_loss_coef=0, 
#         img_pred_loss_coef=0, do_symmetry=False, 
#         variational_rnn=True, rnn_width=128, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
#         z_pred_loss_coef=.005, 
#         T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
#         grad_clip=.03, BCE_not_MSE=False, 
#         teacher_forcing_duration=0, 
#         supervised_rnn=True, 
#     )), 
#     ('1', Config(
#         beta=Constant(1e-5), vae_loss_coef=0, 
#         img_pred_loss_coef=0, do_symmetry=False, 
#         variational_rnn=True, rnn_width=128, 
#         deep_spread=False, vae_channels=[16, 32, 64], 
#         vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
#         z_pred_loss_coef=1, 
#         T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
#         grad_clip=.03, BCE_not_MSE=False, 
#         teacher_forcing_duration=0, 
#         supervised_rnn=True, 
#     )), 
# ]

# supervised rnn supervised vae
EXPERIMENTS = [
    ('128', Config(
        beta=Constant(0), vae_loss_coef=0, 
        img_pred_loss_coef=0, do_symmetry=False, 
        variational_rnn=True, rnn_width=128, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=1, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=0, 
        supervised_rnn=True, skip_vae=False, 
        supervised_vae=True, vae_supervision_loss_coef=1, 
    )), 
    ('64', Config(
        beta=Constant(0), vae_loss_coef=0, 
        img_pred_loss_coef=0, do_symmetry=False, 
        variational_rnn=True, rnn_width=64, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=1, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=0, 
        supervised_rnn=True, skip_vae=False, 
        supervised_vae=True, vae_supervision_loss_coef=1, 
    )), 
    ('32', Config(
        beta=Constant(0), vae_loss_coef=0, 
        img_pred_loss_coef=0, do_symmetry=False, 
        variational_rnn=True, rnn_width=32, 
        deep_spread=False, vae_channels=[16, 32, 64], 
        vvrnn=False, vvrnn_static=-25, rnn_min_context=4, 
        z_pred_loss_coef=1, 
        T=0, R=0, TR=0, I=1, lr=0.001, residual=True, 
        grad_clip=.03, BCE_not_MSE=False, 
        teacher_forcing_duration=0, 
        supervised_rnn=True, skip_vae=False, 
        supervised_vae=True, vae_supervision_loss_coef=1, 
    )), 
]
