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

try:
    from workspace import EXP_PATH, LOCK_EPOCH
except ImportError:
    EXP_PATH = input('EXP_PATH=')
    LOCK_EPOCH = None

def COL_TITLE(group: MyExpGroup):
    # return f'rnn,batch = {(group.hyperParams.rnn_width, group.hyperParams.batch_size)}', ''
    # return f'{group.hyperParams.rnn_width}, {group.hyperParams.batch_size}', '\n rnn_width, batch_size'
    return f'|train set| = {group.hyperParams.train_set_size}', ''
    # return f'vae channels = \n{group.hyperParams.vae_channels}', ''

def main(experiment_path, lock_epoch):
    exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
        experiment_path, EXPERIMENT_PY_FILENAME, 
    ))
    groups: List[MyExpGroup]
    n_groups_per_symm = len(groups) // 2

    # permute
    # groups = [*groups[::2], *groups[1::2]]

    print(f'{exp_name = }')
    print(*[x.variable_value for x in groups], sep='\n')

    fig, axes = plt.subplots(1, n_groups_per_symm, sharey=True)
    X = [1, 2]
    for col_i, ax in enumerate(axes):
        # change together: 3g958hpf598
        assert 'yes' in groups[col_i                    ].variable_value
        assert 'no'  in groups[col_i + n_groups_per_symm].variable_value
        title, var_names = COL_TITLE(groups[col_i])
        assert title == COL_TITLE(groups[col_i + n_groups_per_symm])[0]
        ax.set_title(title)

    dataset = Dataset(
        experiment.datasetDef, 
        is_train_not_validate=False, size=None, 
        device=DEVICE, 
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
                vae=(VAE, 1), 
            ), lock_epoch)
            vae: VAE = models['vae'][0]
            vae.eval()
            with torch.no_grad():
                Z, _ = vae.encode(image_set)
                mse = USING_METRIC.method(Z, traj_set)
            Y[i_group].append(mse)

    for col_i, ax in enumerate(axes):
        Ys = (Y[col_i], Y[col_i + n_groups_per_symm])
        # for x, y in zip(X, Ys):
        #     ax.plot(
        #         [x] * n_rand_inits, y, linestyle='none', 
        #         markerfacecolor='none', markeredgecolor='k', 
        #         marker='o', markersize=10, 
        #     )
        ax.boxplot(Ys)
        
        ax.set_xticks(X)
        # change together: 3g958hpf598
        ax.set_xticklabels([
            'symm', 
            'no symm', 
        ])
        ax.set_xlim(.8, 2.2)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('MSE')
    plt.suptitle('Linear projection MSE (↓)' 
        + USING_METRIC.suptitle
        + var_names
    )
    plt.tight_layout()
    plt.savefig(path.join(
        experiment_path, f'auto_fig8{USING_METRIC.name}.pdf', 
    ))
    plt.show()

if __name__ == '__main__':
    main(EXP_PATH, LOCK_EPOCH)
