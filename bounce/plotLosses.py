import os
from os import path
import importlib.util

from matplotlib import pyplot as plt

from shared import *

EXP_PATH = 'C:/Users/iGlop/d/symmetry/danRepo/bounce/results/TRI_2'

AVERAGE_OVER = 100
START = 0

def getExp():
    spec = importlib.util.spec_from_file_location(
        "experiments", path.abspath('experiments.py'), 
    )
    experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiments)
    return experiments

def main():
    os.chdir(EXP_PATH)
    experiments = getExp()
    for i, (exp_name, config) in enumerate(
        experiments.EXPERIMENTS, 
    ):
        print(exp_name)
        print(config)
        for rand_init_i in range(experiments.RAND_INIT_TIMES):
            print('rand init', rand_init_i)
            exp_path = renderExperimentPath(rand_init_i, config)
            epochs, losses = extract(exp_path)
            plt.plot(
                epochs, losses, c='rgbky'[i], label=exp_name, 
            )
    plt.legend()
    plt.show()

PREFIX = '  validate_recon__loss = '
PREFIX_LEN = len(PREFIX)
def extract(exp_path):
    losses = []
    epochs = []
    group_sum = 0
    group_size = 0
    def pop():
        nonlocal group_sum, group_size
        losses.append(group_sum / group_size)
        epochs.append(len(losses) * AVERAGE_OVER)
        print(
            'epoch', epochs[-1], 
            end='\r', flush=True, 
        )
        group_sum = 0
        group_size = 0
    with open(path.join(exp_path, 'losses.log')) as f:
        for i, line in enumerate(f):
            if i % 10 == 2:
                if not line.startswith(PREFIX):
                    from console import console
                    console({**globals(), **locals()})
                group_sum += float(line[PREFIX_LEN:])
                group_size += 1
                if group_size == AVERAGE_OVER:
                    pop()
    return epochs[START:], losses[START:]

main()
