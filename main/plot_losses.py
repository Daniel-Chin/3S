from os import path

from matplotlib import pyplot as plt
from torchWork.plot_losses import plotLosses, LossType
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME

EXP_PATH = path.join('./experiments/', '''
2022_m10_d27@18_20_46_grad_clip
'''.strip())

fig = plotLosses(
    path.join(EXP_PATH, EXPERIMENT_PY_FILENAME), 
    [
        # LossType('validate', 'loss_root.supervise.vae'), 
        # LossType('validate', 'loss_root.supervise.rnn'), 

        # LossType('validate', 'loss_root.supervise.vae.encode'), 
        # LossType('validate', 'loss_root.supervise.vae.decode'), 
        # LossType('train',    'loss_root.supervise.vae.encode'), 
        # LossType('train',    'loss_root.supervise.vae.decode'), 
        # LossType('train',    'loss_root.self_recon'), 
        LossType('validate', 'loss_root.self_recon'), 

        # LossType('train',    'loss_root.predict.z'), 
        # LossType('validate', 'loss_root.predict.z'), 
        # LossType('train',    'loss_root.predict.image'), 
        LossType('validate', 'loss_root.predict.image'), 

        # LossType('validate', 'loss_root.seq_energy'), 

        # LossType('train',    'linear_proj_mse'), 
        LossType('validate', 'linear_proj_mse'), 
    ], 
    average_over=300, epoch_start=3000, 
    which_legend=0, linewidth=1, 
)

plt.savefig(path.join(EXP_PATH, 'auto_plot_loss.pdf'))
plt.show()
