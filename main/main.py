import torchWork
from torchWork import DEVICE
from torchWork.experiment_control import (
    runExperiment, loadExperiment, ExperimentGroup, 
)
from torchWork.profiler import GPUUtilizationReporter

from shared import *
from load_dataset import Dataset
from one_epoch import oneEpoch
from vae import VAE
from rnn import PredRNN, EnergyRNN
from arg_parser import ArgParser

def main():
    args = ArgParser()
    print(f'{args.exp_py_path = }')

    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(args.exp_py_path)
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
        experiment.SEQ_LEN, 
        experiment.ACTUAL_DIM, DEVICE, 
    )
    validateSet = Dataset(
        experiment.VALIDATE_SET_PATH, experiment.VALIDATE_SET_SIZE, 
        experiment.SEQ_LEN, 
        experiment.ACTUAL_DIM, DEVICE, 
    )
    with GPUUtilizationReporter(interval=10):
        runExperiment(
            args.exp_py_path, requireModelClasses, oneEpoch, 
            trainSet, validateSet, 
        )

def requireModelClasses(hParams: HyperParams):
    x = {}
    x['vae'] = (VAE, 1)
    x['predRnn'] = (PredRNN, hParams.rnn_ensemble)
    x['energyRnn'] = (EnergyRNN, (
        1 if hParams.lossWeightTree['seq_energy'].weight 
        else 0
    ))
    return x

main()
