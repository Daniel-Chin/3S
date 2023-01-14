from os import path
from typing import List

import torch
from torch.nn import functional as F
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from matplotlib import pyplot as plt
from tqdm import tqdm

from load_dataset import Dataset
from vae import VAE
from info_probe import probe, InfoProbeDataset
from template_bounce import MyExpGroup

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

    # Manual filtering
    # n_rand_inits = 6

    max_dataset_size = 0
    for group in groups:
        max_dataset_size = max(
            max_dataset_size, 
            group.hyperParams.train_set_size, 
        )
    trainSet    = Dataset(
        experiment.TRAIN_SET_PATH,    max_dataset_size,  
        experiment.SEQ_LEN, 
        experiment.ACTUAL_DIM, DEVICE, 
    )
    validateSet = Dataset(
        experiment.VALIDATE_SET_PATH, experiment.VALIDATE_SET_SIZE, 
        experiment.SEQ_LEN, 
        experiment.ACTUAL_DIM, DEVICE, 
    )

    infoProbeDatasets = []
    for group in groups:
        infoProbeDatasets_g = []
        infoProbeDatasets.append(infoProbeDatasets_g)
        for rand_init_i in range(n_rand_inits):
            epoch, models = loadLatestModels(experiment_path, group, rand_init_i, dict(
                vae=(VAE, 1), 
            ), lock_epoch)
            vae: VAE = models['vae'][0]
            
            infoProbeTrainSet = InfoProbeDataset(
                experiment, group.hyperParams, vae, trainSet, 
            )
            infoProbeValidateSet = InfoProbeDataset(
                experiment, group.hyperParams, vae, validateSet, 
            )
            infoProbeDatasets_g.append((
                infoProbeTrainSet, infoProbeValidateSet, 
            ))
    collapse_mse = collapseBaselineMSE(infoProbeValidateSet)
    for i in range(99):
        kw = dict(n_epochs=200)
        if i == 0:
            pass
        else:
            def nE(x):
                kw['n_epochs'] = x
            from console import console
            console({**globals(), **locals()})
        print('training info probes...')
        for group, infoProbeDatasets_g in zip(groups, infoProbeDatasets):
            for rand_init_i, (
                infoProbeTrainSet, infoProbeValidateSet, 
            ) in tqdm(
                *[enumerate(infoProbeDatasets_g)], 
                desc=group.variable_value, 
            ):
                train_losses, validate_losses = probe(
                    experiment, group.hyperParams, 
                    infoProbeTrainSet, infoProbeValidateSet, 
                    **kw, 
                )
                lineTrain, = plt.plot(
                    train_losses,    
                    c='r', linewidth=1, 
                )
                lineValid, = plt.plot(
                    validate_losses, 
                    c='b', linewidth=1, 
                )
        lineCollapse = plt.axhline(
            collapse_mse, c='k', linewidth=1, 
        )
        plt.axhline(
            0, c='g', linewidth=1, 
        )
        plt.legend(
            [lineTrain, lineValid, lineCollapse], 
            ['train', 'validate', 'full collapse'], 
        )
        # plt.ylim(0, 4)
        print('plot showing...')
        plt.show()

def collapseBaselineMSE(dataset: InfoProbeDataset):
    traj = dataset.traj[0, :, :]
    return F.mse_loss(
        traj.mean(dim=0).unsqueeze(0).repeat((traj.shape[0], 1)), 
        traj, 
    )

if __name__ == '__main__':
    main(EXP_PATH, LOCK_EPOCH)
