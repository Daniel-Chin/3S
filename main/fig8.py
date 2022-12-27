from os import path
from typing import List

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from matplotlib import pyplot as plt

from vae import VAE
from load_dataset import Dataset
from template_bounce import MyExpGroup
from linearity_metric import projectionMSE, projectionMSELockHeight

class USING_METRIC:
    name = ''
    method = projectionMSE
    suptitle = ''

# class USING_METRIC:
#     name = '_lock_height'
#     method = projectionMSELockHeight
#     suptitle = '\n(height locked)'

EXPERIMENT_PATH = path.join('./experiments', '''
2022_m12_d19@07_44_07_train_size_c_max_batch
'''.strip())
LOCK_EPOCH = None

def main(experiment_path, lock_epoch):
    exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
        experiment_path, EXPERIMENT_PY_FILENAME, 
    ))
    groups: List[MyExpGroup]
    print(f'{exp_name = }')
    dataset = Dataset(
        experiment.VALIDATE_SET_PATH, 
        experiment.VALIDATE_SET_SIZE, experiment.SEQ_LEN, 
        experiment.ACTUAL_DIM, DEVICE, 
    )
    _shape = dataset.video_set.shape
    image_set = dataset.video_set.view(
        _shape[0] * _shape[1], _shape[2], _shape[3], _shape[4], 
    )
    _shape = dataset.label_set.shape
    traj_set = dataset.label_set.view(
        _shape[0] * _shape[1], _shape[2], 
    )
    Y = [[] for _ in groups]
    for i_group, group in enumerate(groups):
        print(group.name())
        group.hyperParams.print(depth=1)
        for rand_init_i in range(n_rand_inits):
            print(f'{rand_init_i = }')
            epoch, models = loadLatestModels(experiment_path, group, rand_init_i, dict(
                vae=VAE, 
            ), lock_epoch)
            vae: VAE = models['vae']
            vae.eval()
            with torch.no_grad():
                Z, _ = vae.encode(image_set)
                mse = USING_METRIC.method(Z, traj_set)
            Y[i_group].append(mse)
    fig, axes = plt.subplots(1, 5, sharey=True)
    X = [1, 2]
    for col_i, ax in enumerate(axes):
        Ys = (Y[col_i], Y[col_i + 5])
        # for x, y in zip(X, Ys):
        #     ax.plot(
        #         [x] * n_rand_inits, y, linestyle='none', 
        #         markerfacecolor='none', markeredgecolor='k', 
        #         marker='o', markersize=10, 
        #     )
        ax.boxplot(Ys)
        ax.set_xticks(X)
        ax.set_xticklabels(['symm', 'no symm', ])
        assert 'yes' in groups[col_i    ].variable_value
        assert 'no'  in groups[col_i + 5].variable_value
        ax.set_xlim(.8, 2.2)
        ax.set_title(f'|train set| = {groups[col_i].hyperParams.train_set_size}')
        # ax.set_title(f'vae channels = \n{groups[col_i].hyperParams.vae_channels}')
    axes[0].set_ylabel('MSE')
    plt.suptitle('Linear projection MSE (↓)' 
        + USING_METRIC.suptitle
    )
    plt.tight_layout()
    plt.savefig(path.join(
        experiment_path, f'auto_fig8{USING_METRIC.name}.pdf', 
    ))
    plt.show()

if __name__ == '__main__':
    main(EXPERIMENT_PATH, LOCK_EPOCH)
