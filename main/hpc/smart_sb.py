from datetime import datetime

from torchWork import loadExperiment

from ..arg_parser import ArgParser

SBATCH_FILENAME = 'auto.sbatch'

def main():
    args = ArgParser()
    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(args.exp_py_path)

    t = datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')

    with open('template.sbatch', 'r') as fin:
        with open(SBATCH_FILENAME, 'w') as fout:
            for line in fin:
                fout.write(line.replace(
                    '{OUT_FILENAME}', f'{t}_%j_%x', 
                ).replace(
                    '{JOB_NAME}', experiment_name, 
                )).replace(
                    '{ARGS}', f'"{args.exp_py_path}"', 
                )

main()
