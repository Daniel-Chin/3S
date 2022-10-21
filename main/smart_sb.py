#!/bin/python3

print('importing...')
import os
from datetime import datetime
import importlib.util
from subprocess import Popen

SBATCH_FILENAME = 'auto.sbatch'

def loadModule(module_path):
    # this func already kinda exists in torchWork. 
    # no idea how to optimize tho
    spec = importlib.util.spec_from_file_location(
        'what is this str', module_path, 
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    print('main...')
    os.chdir('..')
    from arg_parser import ArgParser
    args = ArgParser()
    exp = loadModule(args.exp_py_path)

    t = datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')

    os.chdir('./hpc')
    with open('template.sbatch', 'r') as fin:
        with open(SBATCH_FILENAME, 'w') as fout:
            for line in fin:
                fout.write(line.replace(
                    '{OUT_FILENAME}', f'{t}_%j_%x', 
                ).replace(
                    '{JOB_NAME}', exp.EXP_NAME, 
                ).replace(
                    '{ARGS}', f'"{args.exp_py_path}"', 
                ))
    
    with Popen(['sbatch', SBATCH_FILENAME]) as p:
        p.wait()

main()
