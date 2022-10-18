from os import path

from matplotlib import pyplot as plt
from torchWork.plot_losses import plotLosses, LossType
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME

fig = plotLosses(
    path.join('./experiments/', '''
symm_rnn_width_2022_Oct_17_23;54;09
'''.strip(), EXPERIMENT_PY_FILENAME), 
    [
        # LossType('validate', 'loss_root.supervise.vae'), 
        # LossType('validate', 'loss_root.supervise.rnn'), 

        # LossType('validate', 'loss_root.supervise.vae.encode'), 
        # LossType('validate', 'loss_root.supervise.vae.decode'), 
        # LossType('train',    'loss_root.supervise.vae.encode'), 
        # LossType('train',    'loss_root.supervise.vae.decode'), 
        # LossType('train',    'loss_root.self_recon'), 

        LossType('train',    'loss_root.predict.image'), 
        LossType('validate', 'loss_root.predict.image'), 
    ], 
    average_over=300, epoch_start=3000, 
)

plt.show()
