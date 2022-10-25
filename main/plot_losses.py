from os import path

from matplotlib import pyplot as plt
from torchWork.plot_losses import plotLosses, LossType
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME

fig = plotLosses(
    path.join('./experiments/', '''
2022_m10_d24@23_57_39_two_body_nozl
'''.strip(), EXPERIMENT_PY_FILENAME), 
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

        # LossType('train',    'linear_proj_mse'), 
        LossType('validate', 'linear_proj_mse'), 
    ], 
    average_over=300, epoch_start=3000, 
    which_legend=0, linewidth=1, 
)

plt.show()
