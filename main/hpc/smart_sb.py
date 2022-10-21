from subprocess import Popen
from datetime import datetime

from torchWork import loadExperiment

from ..arg_parser import ArgParser

def main():
    with Popen(['conda', 'deactivate']) as p:
        p.wait()

    args = ArgParser()
    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(args.exp_py_path)

    t = datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')

    sbatch_filename = f'auto_{t}.sbatch'

    with open('train.sbatch', 'r') as fin:
        with open(sbatch_filename, 'w') as fout:
            for line in fin:
                fout.write(line.replace(
                    '{OUT_FILENAME}', f'{t}_%j_%x', 
                ).replace(
                    '{JOB_NAME}', experiment_name, 
                ))

    with Popen(['sbatch', sbatch_filename, args.exp_py_path]) as p:
        p.wait()

main()
