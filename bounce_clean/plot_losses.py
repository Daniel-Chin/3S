from os import path

from matplotlib import pyplot as plt
from torchWork.plot_losses import plotLosses, LossType
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME

fig = plotLosses(
    path.join('./experiments/', '''
supervised_rnn_width_2022_Oct_13_22;48;34
'''.strip(), EXPERIMENT_PY_FILENAME), 
    [
        # LossType('validate', 'loss_root.supervise.vae'), 
        # LossType('validate', 'loss_root.supervise.rnn'), 

        LossType('validate', 'loss_root.supervise.vae.encode'), 
        LossType('train', 'loss_root.supervise.vae.encode'), 
        # LossType('validate', 'loss_root.supervise.vae.decode'), 
    ], 
    average_over=300, epoch_start=3000, 
)

plt.show()
