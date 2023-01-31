from os import path

import torchWork
from torchWork import DEVICE
from torchWork.experiment_control import (
    runExperiment, loadExperiment, ExperimentGroup, 
    EXPERIMENT_PY_FILENAME, 
)
from torchWork.profiler import GPUUtilizationReporter

from shared import *
from one_epoch import oneEpoch
from vae import VAE
from rnn import PredRNN, EnergyRNN
from arg_parser import ArgParser

def main(continue_exp_dir=None):
    if continue_exp_dir is None:
        args = ArgParser()
        print(f'{args.exp_py_path = }')
        exp_py_path = args.exp_py_path
    else:
        exp_py_path = path.join(
            './experiments', 
            continue_exp_dir, EXPERIMENT_PY_FILENAME, 
        )

    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(exp_py_path)
    print(
        'Experiment:', experiment_name, ',', 
        len(groups), 'x', n_rand_inits, 
    )
    assert groups
    max_dataset_size = None
    for group in groups:
        group: ExperimentGroup
        hParams: HyperParams = group.hyperParams
        if hParams.train_set_size is not None:
            max_dataset_size = max_dataset_size or 0
            max_dataset_size = max(
                max_dataset_size, 
                hParams.train_set_size, 
            )
    trainSet    = experiment.getDataset(
        is_train_not_validate=True,  size=max_dataset_size, 
        device=DEVICE, 
    )
    validateSet = experiment.getDataset(
        is_train_not_validate=False, size=None, 
        device=DEVICE, 
    )
    with GPUUtilizationReporter(interval=10):
        runExperiment(
            exp_py_path, requireModelClasses, oneEpoch, 
            trainSet, validateSet, continue_from=continue_exp_dir, 
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

if __name__ == '__main__':
    main()
