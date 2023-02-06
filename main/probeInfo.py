from os import path
from typing import List
from numbers import Number

import torch
from torch.nn import functional as F
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import (
    EXPERIMENT_PY_FILENAME, loadLatestModels, ExperimentGroup, 
)
from matplotlib import pyplot as plt
import tqdm

from shared import *
from hyper_params import *
from load_dataset import Dataset
from vae import VAE
from info_probe import probe, InfoProbeDataset
from template_bounce import MyExpGroup

try:
    from workspace import (
        EXP_PATH, LOCK_EPOCH, INFO_PROBE_N_EPOCHS, 
    )
except ImportError:
    EXP_PATH = input('EXP_PATH=')
    LOCK_EPOCH = None

CHECK_OVERFIT = 'CHECK_OVERFIT'
COMPARE_GROUPS = 'COMPARE_GROUPS'

def main(experiment_path, lock_epoch):
    exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
        experiment_path, EXPERIMENT_PY_FILENAME, 
    ))
    groups: List[MyExpGroup]
    print(f'{exp_name = }')

    # Manual filtering
    # n_rand_inits = 1

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
    trainSet    = Dataset(
        experiment.datasetDef, 
        is_train_not_validate=True,  size=max_dataset_size, 
        device=DEVICE, 
    )
    validateSet = Dataset(
        experiment.datasetDef, 
        is_train_not_validate=False, size=None, 
        device=DEVICE, 
    )

    for i in range(99):
        kw = dict(n_epochs=INFO_PROBE_N_EPOCHS)
        if i == 0:
            pass
        else:
            def nE(x):
                kw['n_epochs'] = x
            from console import console
            console({**globals(), **locals()})
        print('training info probes...')
        Y_train: List[List[torch.Tensor]] = [[] for _ in range(n_rand_inits)]
        Y_valid: List[List[torch.Tensor]] = [[] for _ in range(n_rand_inits)]
        for rand_init_i in tqdm.trange(n_rand_inits):
            print()
            for group_i, group in enumerate(groups):
                infoProbeTrainSet, infoProbeValidateSet = encodeEverything(
                    experiment, experiment_path, lock_epoch, 
                    trainSet, validateSet, 
                    group_i, group, rand_init_i, 
                )
                collapse_mse = collapseBaselineMSE(infoProbeValidateSet)
                train_losses, validate_losses = probe(
                    experiment, group.hyperParams, 
                    infoProbeTrainSet, infoProbeValidateSet, 
                    **kw, 
                )
                Y_train[rand_init_i].append(   train_losses)
                Y_valid[rand_init_i].append(validate_losses)
        for info_probe_mode in (CHECK_OVERFIT, COMPARE_GROUPS):
            legend_handles = []
            legend_labels = []
            if info_probe_mode is CHECK_OVERFIT:
                for rand_init_i in range(n_rand_inits):
                    for group_i, group in enumerate(groups):
                        lineTrain, = plt.plot(
                            Y_train[rand_init_i][group_i],
                            c='r', linewidth=1, 
                        )
                        lineValid, = plt.plot(
                            Y_valid[rand_init_i][group_i],
                            c='b', linewidth=1, 
                        )
                plt.xlabel('Epoch')
                legend_handles.append(lineTrain)
                legend_labels.append('train')
                legend_handles.append(lineValid)
                legend_labels.append('validate')
            elif info_probe_mode is COMPARE_GROUPS:
                X = [g.variable_value for g in groups]
                if not all([isinstance(x, Number) for x in X]):
                    X = range(len(groups))
                for rand_init_i in range(n_rand_inits):
                    for which, Y, c in (
                        ('train',    Y_train, 'y'), 
                        ('validate', Y_valid, 'k'), 
                    ): 
                        y = []
                        for losses in Y[rand_init_i]:
                            y.append(losses[-1])
                        line = plt.plot(
                            X, y, linestyle='none', 
                            markerfacecolor='none', markeredgecolor=c, 
                            marker='o', markersize=10, 
                        )
                        if rand_init_i == 0:
                            legend_handles.append(line[0])
                            legend_labels.append(which)
                plt.xticks(X, [g.variable_value for g in groups])
                plt.xlabel(group.variable_name)
            plt.ylabel('Info Probe Losses')
            plt.suptitle(exp_name)
            lineCollapse = plt.axhline(
                collapse_mse, c='k', linewidth=1, 
            )
            plt.axhline(
                0, c='g', linewidth=1, 
            )
            plt.legend(
                [*legend_handles, lineCollapse], 
                [*legend_labels, 'full collapse'], 
            )
            # plt.ylim(0, 4)
            if info_probe_mode is COMPARE_GROUPS:
                plt.savefig(path.join(
                    experiment_path, f'auto_info_probe.pdf', 
                ))
            plt.show()

cached_collapse_baseline_mse = None
def collapseBaselineMSE(dataset: InfoProbeDataset):
    global cached_collapse_baseline_mse
    if cached_collapse_baseline_mse is None:
        traj = dataset.traj
        cached_collapse_baseline_mse = F.mse_loss(
            traj.mean(dim=0).unsqueeze(0).repeat((traj.shape[0], 1)), 
            traj, 
        )
    return cached_collapse_baseline_mse

cache_encodeEverything = {}
def encodeEverything(
    experiment, experiment_path, lock_epoch, 
    trainSet, validateSet, 
    group_i, group: MyExpGroup, rand_init_i, 
):
    cache_key = (group_i, rand_init_i)
    if cache_key in cache_encodeEverything:
        print('Warning f3qp8947gh4: cache hit. ')
        print('You are probably in an interactive session where the same plot is drawn over and over again.')
        print('If not, beware: the caching only checks for group_i and rand_init_i. Make sure other inputs are also equal!')
    else:
        epoch, models = loadLatestModels(
            experiment_path, group, rand_init_i, dict(
                vae=(VAE, 1), 
            ), lock_epoch, 
        )
        vae: VAE = models['vae'][0]
        
        infoProbeTrainSet = InfoProbeDataset(
            experiment, group.hyperParams, vae, trainSet, 
        )
        infoProbeValidateSet = InfoProbeDataset(
            experiment, group.hyperParams, vae, validateSet, 
        )
        result = (infoProbeTrainSet, infoProbeValidateSet)
        cache_encodeEverything[cache_key] = result
    return cache_encodeEverything[cache_key]

if __name__ == '__main__':
    main(EXP_PATH, LOCK_EPOCH)
