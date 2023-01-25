#!/bin/python3

import os
from datetime import datetime
from subprocess import Popen

from arg_parser import ArgParser
from torchWork import loadExperiment

SBATCH_FILENAME = 'auto.sbatch'

def getExpName(experiment_py_path):
    exp_name, n_rand_inits, groups, experiment = loadExperiment(
        experiment_py_path, 
    )
    return experiment.EXP_NAME

def main():
    args = ArgParser()
    exp_name = getExpName(args.exp_py_path)

    t = datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')
    user = os.getenv('USER')

    os.chdir('./hpc')
    with open('template.sbatch', 'r') as fin:
        with open(SBATCH_FILENAME, 'w') as fout:
            for line in fin:
                fout.write(line.replace(
                    '{OUT_FILENAME}', f'{t}_%j_%x', 
                ).replace(
                    '{JOB_NAME}', exp_name, 
                ).replace(
                    '{ARGS}', f'"{args.exp_py_path}"', 
                ).replace(
                    '{USER}', user, 
                ))
    
    with Popen(['sbatch', SBATCH_FILENAME]) as p:
        p.wait()

main()
