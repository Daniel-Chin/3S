from os import path
from typing import List

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from matplotlib import pyplot as plt

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

    for i in range(99):
        kw = dict(n_epochs=10)
        if i == 0:
            pass
        else:
            from console import console
            console({**globals(), **locals()})
        for group, infoProbeDatasets_g in zip(groups, infoProbeDatasets):
            for rand_init_i, (
                infoProbeTrainSet, infoProbeValidateSet, 
            ) in enumerate(infoProbeDatasets_g):
                train_losses, validate_losses = probe(
                    experiment, group.hyperParams, 
                    infoProbeTrainSet, infoProbeValidateSet, 
                    **kw, 
                )
                plt.plot(   train_losses, c='r', label='train'),
                plt.plot(validate_losses, c='b', label='validate')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main(EXP_PATH, LOCK_EPOCH)
