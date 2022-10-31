from os import path

import torchWork
from torchWork import DEVICE
from torchWork.experiment_control import (
    runExperiment, loadExperiment, ExperimentGroup, 
    EXPERIMENT_PY_FILENAME, 
)

from shared import *
from load_dataset import Dataset
from one_epoch import oneEpoch
from vae import VAE
from rnn import PredRNN, EnergyRNN

EXP_DIR = '2022_m10_d29@14_11_09_dataset_size'

def continueTrain():
    exp_py_path = path.join(
        './experiments', 
        EXP_DIR, EXPERIMENT_PY_FILENAME, 
    )
    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(exp_py_path)
    print(
        'Experiment:', experiment_name, ',', 
        len(groups), 'x', n_rand_inits, 
    )
    max_dataset_size = 0
    for group in groups:
        group: ExperimentGroup
        hParams: HyperParams = group.hyperParams
        max_dataset_size = max(
            max_dataset_size, 
            hParams.train_set_size, 
        )
    trainSet    = Dataset(
        experiment.TRAIN_SET_PATH,    max_dataset_size,  
        experiment.ACTUAL_DIM, DEVICE, 
    )
    validateSet = Dataset(
        experiment.VALIDATE_SET_PATH, experiment.VALIDATE_SET_SIZE, 
        experiment.ACTUAL_DIM, DEVICE, 
    )
    runExperiment(exp_py_path, oneEpoch, {
        'vae': VAE, 
        'predRnn': PredRNN, 
        'energyRnn': EnergyRNN, 
    }, trainSet, validateSet, continue_from=EXP_DIR)

continueTrain()
