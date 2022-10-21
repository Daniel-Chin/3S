#!/bin/python3

import os
from datetime import datetime
from subprocess import Popen

from arg_parser import ArgParser

SBATCH_FILENAME = 'auto.sbatch'

def getExpName(experiment_py_path):
    KEYWORD = 'EXP_NAME = '
    with open(experiment_py_path, 'r') as f:
        for line in f:
            if line.startswith(KEYWORD):
                return line.split(KEYWORD)[1]

def main():
    args = ArgParser()
    exp_name = getExpName(args.exp_py_path)

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
