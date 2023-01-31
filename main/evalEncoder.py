from os import path
from typing import List
from numbers import Number

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from matplotlib import pyplot as plt

from vae import VAE
from load_dataset import getImageSet
from template_bounce import MyExpGroup
from linearity_metric import projectionMSE

try:
    from workspace import EXP_PATH, LOCK_EPOCH
except ImportError:
    EXP_PATH = input('EXP_PATH=')
    LOCK_EPOCH = None

def main(experiment_path, lock_epoch):
    exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
        experiment_path, EXPERIMENT_PY_FILENAME, 
    ))
    groups: List[MyExpGroup]
    print(f'{exp_name = }')
    validateSet = experiment.getDataset(
        is_train_not_validate=False, size=None, 
        device=DEVICE, 
    )
    image_set, traj_set = getImageSet(validateSet)
    X = [g.variable_value for g in groups]
    if not all([isinstance(x, Number) for x in X]):
        X = range(len(groups))
    Y: List[List[torch.Tensor]] = [[] for _ in range(n_rand_inits)]
    for group in groups:
        print(group.name())
        group.hyperParams.print(depth=1, exclude=['experiment_globals'])
        group.hyperParams.fillDefaults()
        for rand_init_i in range(n_rand_inits):
            print(f'{rand_init_i = }')
            epoch, models = loadLatestModels(experiment_path, group, rand_init_i, dict(
                vae=(VAE, 1), 
            ), lock_epoch)
            vae: VAE = models['vae'][0]
            vae.eval()
            with torch.no_grad():
                Z, _ = vae.encode(image_set)
                mse = projectionMSE(Z, traj_set)
            Y[rand_init_i].append(mse)
    print('data:')
    data = [[] for _ in Y[0]]
    for Y_i in Y:
        plt.plot(
            X, Y_i, linestyle='none', 
            markerfacecolor='none', markeredgecolor='k', 
            marker='o', markersize=10, 
        )
        for i, y in enumerate(Y_i):
            data[i].append(y.item())
    for d, g in zip(data, groups):
        print(g.variable_value)
        print(*d, sep='\n')
        print()
    plt.ylabel('Linear projection MSE (↓)')
    plt.xlabel(group.variable_name)
    plt.xticks(X, [g.variable_value for g in groups])
    plt.ylim(0, 1)
    print(*[g.variable_value for g in groups], sep='\n')
    plt.suptitle(exp_name)
    plt.savefig(path.join(experiment_path, 'auto_eval_encoder.pdf'))
    plt.show()

if __name__ == '__main__':
    main(EXP_PATH, LOCK_EPOCH)
