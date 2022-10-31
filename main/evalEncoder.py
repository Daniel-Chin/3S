from os import path
from typing import List

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from matplotlib import pyplot as plt

from vae import VAE
from load_dataset import Dataset
from template_bounce import MyExpGroup
from linearity_metric import projectionMSE

EXPERIMENT_PATH = path.join('./experiments', '''
2022_m10_d29@15_32_18_tf_time
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
        experiment.VALIDATE_SET_SIZE, experiment.ACTUAL_DIM, DEVICE, 
    )
    _shape = dataset.video_set.shape
    image_set = dataset.video_set.view(
        _shape[0] * _shape[1], _shape[2], _shape[3], _shape[4], 
    )
    _shape = dataset.label_set.shape
    traj_set = dataset.label_set.view(
        _shape[0] * _shape[1], _shape[2], 
    )
    X = range(len(groups))
    Y = [[] for _ in range(n_rand_inits)]
    for group in groups:
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
                mse = projectionMSE(Z, traj_set)
            Y[rand_init_i].append(mse)
    for Y_i in Y:
        plt.plot(
            X, Y_i, linestyle='none', 
            markerfacecolor='none', markeredgecolor='k', 
            marker='o', markersize=10, 
        )
    plt.ylabel('Linear projection MSE (↓)')
    plt.xlabel(group.variable_name)
    plt.xticks(X, [g.variable_value for g in groups])
    plt.suptitle(exp_name)
    plt.savefig(path.join(experiment_path, 'auto_eval_encoder.pdf'))
    plt.show()

if __name__ == '__main__':
    main(EXPERIMENT_PATH, LOCK_EPOCH)
