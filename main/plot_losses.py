from os import path

from matplotlib import pyplot as plt
from torchWork.plot_losses import plotLosses, LossType
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME

try:
    from workspace import EXP_PATH
except ImportError:
    EXP_PATH = input('EXP_PATH=')

lossTypes = [
    # LossType('validate', 'loss_root.supervise.vae'), 
    # LossType('validate', 'loss_root.supervise.rnn'), 

    # LossType('validate', 'loss_root.supervise.vae.encode'), 
    # LossType('validate', 'loss_root.supervise.vae.decode'), 
    # LossType('train',    'loss_root.supervise.vae.encode'), 
    # LossType('train',    'loss_root.supervise.vae.decode'), 
    # LossType('train',    'loss_root.self_recon'), 
    LossType('validate', 'loss_root.self_recon'), 

    # LossType('train',    'loss_root.predict.z'), 
    LossType('validate', 'loss_root.predict.z'), 
    # LossType('train',    'loss_root.predict.image'), 
    LossType('validate', 'loss_root.predict.image'), 

    # LossType('validate', 'loss_root.seq_energy.real'), 
    # LossType('validate', 'loss_root.seq_energy.fake'), 

    # LossType('train',    'linear_proj_mse'), 
    LossType('validate', 'linear_proj_mse'), 
]
fig = plotLosses(
    path.join(EXP_PATH, EXPERIMENT_PY_FILENAME), 
    lossTypes, 
    average_over=50, epoch_start=1000, 
    which_legend=0, linewidth=1, 
)

# given dataset coord std ~= 1, mse should be in (0, 1). 
# when z collapses, mse is unstable and can give 1e+34. 
# so cap it to (0, 1) in the plot.
assert lossTypes[-1].loss_name == 'linear_proj_mse'
fig.axes[-1].set_ylim(0, 1)

plt.savefig(path.join(EXP_PATH, 'auto_plot_loss.pdf'))
plt.show()
