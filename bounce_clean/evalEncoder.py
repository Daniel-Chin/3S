from os import path
from typing import List

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from shared import VALIDATE_PATH, VALIDATE_SET_SIZE
from vae import VAE
from load_dataset import Dataset
from current_experiment import MyExpGroup

EXPERIMENT_PATH = path.join('./experiments', '''
teacher_forcing_duration_2022_Oct_16_23;05;22
'''.strip())
LOCK_EPOCH = None

def main():
    dataset = Dataset(VALIDATE_PATH, VALIDATE_SET_SIZE, DEVICE)
    _shape = dataset.video_set.shape
    image_set = dataset.video_set.view(
        _shape[0] * _shape[1], _shape[2], _shape[3], _shape[4], 
    )
    _shape = dataset.label_set.shape
    traj_set = dataset.label_set.view(
        _shape[0] * _shape[1], _shape[2], 
    )
    exp_name, n_rand_inits, groups = loadExperiment(path.join(
        EXPERIMENT_PATH, EXPERIMENT_PY_FILENAME, 
    ))
    groups: List[MyExpGroup]
    print(f'{exp_name = }')
    X = range(len(groups))
    Y = [[] for _ in range(n_rand_inits)]
    for group in groups:
        print(group.name())
        group.hyperParams.print(depth=1)
        for rand_init_i in range(n_rand_inits):
            print(f'{rand_init_i = }')
            vae = loadLatestModels(EXPERIMENT_PATH, group, rand_init_i, dict(
                vae=VAE, 
            ), LOCK_EPOCH)['vae']
            vae.eval()
            with torch.no_grad():
                mse = evalOne(vae, image_set, traj_set)
            Y[rand_init_i].append(mse)
    for Y_i in Y:
        plt.plot(
            X, Y_i, linestyle='none', markerfacecolor='none', 
            marker='o', markersize=10, 
        )
    plt.ylabel('Linear projection MSE (↓)')
    plt.xlabel = group.variable_name
    plt.xticks(X, [g.variable_value for g in groups])
    plt.show()

def evalOne(vae: VAE, image_set, traj_set: torch.Tensor):
    Z, _ = vae.encode(image_set)
    Y = traj_set
    Y /= Y.std(dim=0)
    err = getErr(Z, Y)
    # x_mse = err[:, 0].square().mean().item()
    # y_mse = err[:, 1].square().mean().item()
    # z_mse = err[:, 2].square().mean().item()

    # xyz_mse = (x_mse + z_mse + y_mse) / 3

    mse = err.square().mean().item()
    return mse

def getErr(X, Y) -> torch.Tensor:
    regression = LinearRegression().fit(X, Y)
    return Y - regression.predict(X)

main()
