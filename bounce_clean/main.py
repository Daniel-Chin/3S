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

CURRENT_EXP = './current_experiment.py'

def main():
    (
        experiment_name, n_rand_inits, groups, 
    ) = loadExperiment(CURRENT_EXP)
    print('Experiment:', experiment_name, 'x', n_rand_inits)
    max_dataset_size = 0
    for group in groups:
        group: ExperimentGroup
        hParams: HyperParams = group.hyperParams
        max_dataset_size = max(
            max_dataset_size, 
            hParams.train_set_size, 
        )
    trainSet    = Dataset(
        TRAIN_PATH,    max_dataset_size,  DEVICE, 
    )
    validateSet = Dataset(
        VALIDATE_PATH, VALIDATE_SET_SIZE, DEVICE, 
    )
    runExperiment(CURRENT_EXP, oneEpoch, {
        'vae': VAE, 
        'rnn': RNN, 
    }, trainSet, validateSet)

main()
