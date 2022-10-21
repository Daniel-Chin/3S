#!/bin/python3

print('importing...')
import os
from datetime import datetime
import importlib.util
from subprocess import Popen

from arg_parser import ArgParser

SBATCH_FILENAME = 'auto.sbatch'

def loadExperiment(experiment_py_path):
    # this func already kinda exists in torchWork. 
    # no idea how to optimize tho
    spec = importlib.util.spec_from_file_location(
        "experiment", experiment_py_path, 
    )
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    return (
        experiment.EXP_NAME, 
        experiment.N_RAND_INITS, 
        experiment.GROUPS, 
        experiment, 
    )

def main():
    print('main...')
    args = ArgParser()
    exp_name, _, _, _ = loadExperiment(args.exp_py_path)

    t = datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')

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
                ))
    
    with Popen(['sbatch', SBATCH_FILENAME]) as p:
        p.wait()

main()
