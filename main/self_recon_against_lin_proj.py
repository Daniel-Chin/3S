# Can we early stop by looking at self recon? 

import os
from os import path
from typing import List, Tuple, Dict
import itertools
import pickle

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, getTrainerPath
from torchWork.plot_losses import LossAcc, LossType, OnChangeOrEnd
from torchWork.loss_logger import Decompressor, LOSS_FILE_NAME
from matplotlib import pyplot as plt
from tqdm import tqdm

from template_bounce import MyExpGroup

EARLY_STOP = .5

EXP_PATH = path.abspath(path.join('./experiments/', '''
2022_m12_d19@07_44_07_train_size_c_max_batch
'''.strip()))

def main():
    exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
        EXP_PATH, EXPERIMENT_PY_FILENAME, 
    ))
    groups: List[MyExpGroup]
    print(f'{exp_name = }')

    os.chdir(EXP_PATH)
    saveData(exp_name, n_rand_inits, groups)
    data = loadData()

    for early_stop in range(0.1, 1, .2):
        plot(data, early_stop)

def saveData(exp_name, n_rand_inits, groups):
    average_over = 100
    lossTypes = [
        LossType('train',    'loss_root.self_recon'), 
        LossType('validate', 'linear_proj_mse'), 
    ]
    experiment_py_path = path.join(EXP_PATH, EXPERIMENT_PY_FILENAME)

    data: List[Tuple[int, MyExpGroup, int, List[int], Dict[LossType, LossAcc]]] = []
    for (group_i, group), rand_init_i in tqdm([*itertools.product(
        enumerate(groups), range(n_rand_inits), 
    )]):
        lossAccs = {x: LossAcc(average_over) for x in lossTypes}
        with OnChangeOrEnd(*[
            x.endBatch for x in lossAccs.values()
        ]) as oCoE:
            for (
                epoch_i, batch_i, train_or_validate, _, entries, 
            ) in Decompressor(path.join(getTrainerPath(
                path.dirname(experiment_py_path), 
                group.pathName(), rand_init_i, 
            ), LOSS_FILE_NAME)):
                oCoE.eat(epoch_i)
                for loss_name, value in entries.items():
                    lossType = LossType(
                        'train' if train_or_validate else 'validate', 
                        loss_name, 
                    )
                    try:
                        lossAcc = lossAccs[lossType]
                    except KeyError:
                        pass
                    else:
                        lossAcc.eat(value)
        for lossType, lossAcc in lossAccs.items():
            if lossAcc.n_groups == 0:
                raise ValueError('Did not get any', lossType)
        epochs = [(i + 1) * average_over for i in range(
            next(iter(lossAccs.values())).n_groups, 
        )]
        data.append((
            group_i, group, rand_init_i, 
            epochs, lossAccs, 
        ))
    
    try:
        with open(__file__ + '.cache.data', 'wb') as f:
            pickle.dump(data, f)
    except:
        from console import console
        console({**globals(), **locals()})

def loadData():
    with open(__file__ + '.cache.data', 'rb') as f:
        return pickle.load(f)

def plot(data: List[Tuple[int, MyExpGroup, int, List[int], Dict[LossType, LossAcc]]], early_stop):
    X = []
    Y = []
    for (
        group_i, group, rand_init_i, 
        epochs, lossAccs, 
    ) in data:
        for lossType, lossAcc in lossAccs.items():
            loss = lossAcc.getHistory()
            if lossType.loss_name == 'loss_root.self_recon':
                i = torch.ceil(len(loss) * early_stop)
                print(early_stop, '~=', i / len(loss))
                X.append(loss[i])
            else:
                Y.append(loss[-1])

main()
