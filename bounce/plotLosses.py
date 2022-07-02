import enum
import os
from os import path
import importlib.util

from matplotlib import pyplot as plt

from shared import *

EXP_PATH = 'C:/Users/iGlop/d/symmetry/danRepo/bounce/results/residual_2'

AVERAGE_OVER = 100
START = 20
END = None
ALL_TYPES_OR_SINGLE = 1

train____recon__loss = 'train____recon__loss'
validate_recon__loss = 'validate_recon__loss'
validate_kld____loss = 'validate_kld____loss'
train____img_pred_loss = 'train____img_pred_loss'
validate_img_pred_loss = 'validate_img_pred_loss'
validate_z_pred_loss = 'validate_z_pred_loss'
MAP = [
    None, 
    train____recon__loss, 
    validate_recon__loss, 
    None, 
    validate_kld____loss, 
    train____img_pred_loss, 
    validate_img_pred_loss, 
    None, 
    validate_z_pred_loss, 
    None, 
]
MAP_LEN = len(MAP)

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
    for exp_i, (exp_name, config) in enumerate(
        experiments.EXPERIMENTS, 
    ):
        print(exp_name)
        print(config)
        for rand_init_i in range(experiments.RAND_INIT_TIMES):
            if ALL_TYPES_OR_SINGLE == 0:
                if not (rand_init_i == 0 and exp_i in (0, )):
                    break
            print('rand init', rand_init_i)
            exp_path = renderExperimentPath(rand_init_i, config)
            extract(exp_path, exp_name, exp_i)
    plt.legend()
    plt.show()

class LossAcc:
    def __init__(self, losses) -> None:
        self.group_sum = 0
        self.group_size = 0
        self.losses = losses
    
    def eat(self, x):
        self.group_sum += x
        self.group_size += 1
        if self.group_size == AVERAGE_OVER:
            self.pop()

    def pop(self):
        self.losses.append(self.group_sum / self.group_size)
        print(
            'epoch', len(self.losses) * AVERAGE_OVER, 
            end='\r', flush=True, 
        )
        self.group_sum = 0
        self.group_size = 0

def extract(exp_path, exp_name, exp_i):
    if ALL_TYPES_OR_SINGLE == 0:
        loss_types = [
            validate_recon__loss, 
            validate_kld____loss, 
            train____img_pred_loss, 
            validate_img_pred_loss, 
            validate_z_pred_loss, 
        ]
    else:
        loss_types = [
            # validate_recon__loss, 
            # train____recon__loss, 
            # validate_kld____loss, 
            # validate_img_pred_loss, 
            train____img_pred_loss, 
        ]
    epochs = []
    lossAccs = {x: LossAcc([]) for x in loss_types}
    with open(path.join(exp_path, 'losses.log')) as f:
        for i, line in enumerate(f):
            line_type = MAP[i % MAP_LEN]
            if line_type is None:
                continue
            try:
                lossAcc = lossAccs[line_type]
            except KeyError:
                continue
            else:
                prefix = '  %s = ' % line_type
                prefix_len = len(prefix)
                if not line.startswith(prefix):
                    from console import console
                    console({**globals(), **locals()})
                lossAcc.eat(float(line[prefix_len:]))
    epochs = [(i + 1) * AVERAGE_OVER for i, _ in enumerate(
        next(iter(lossAccs.values())).losses, 
    )]
    for loss_i, (loss_type, lossAcc) in enumerate(
        lossAccs.items(), 
    ):
        if ALL_TYPES_OR_SINGLE == 1:
            label = exp_name
            i = exp_i
        else:
            assert loss_type.startswith('validate_')
            label = loss_type[len('validate_'):]
            i = loss_i
        plt.plot(
            epochs[START:END], lossAcc.losses[START:END], 
            c='rgbky'[i], label=label, 
        )

main()
