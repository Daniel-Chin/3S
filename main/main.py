import argparse

import torchWork
from torchWork import DEVICE
from torchWork.experiment_control import (
    runExperiment, loadExperiment, ExperimentGroup, 
)

from shared import *
from load_dataset import Dataset
from one_epoch import oneEpoch
from vae import VAE
from rnn import RNN

DEFAULT_EXP = './current_experiment.py'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_py_path", type=str, nargs='?', default=DEFAULT_EXP, 
        help="the python script that defines the experiment", 
    )
    args = parser.parse_args()
    exp_py_path = args.exp_py_path
    print(f'{exp_py_path = }')

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
        'rnn': RNN, 
    }, trainSet, validateSet)

main()
