#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:3090:1
#SBATCH --job-name={JOB_NAME}
#SBATCH --output={OUT_FILENAME}.out
#SBATCH --error={OUT_FILENAME}.err
#SBATCH --partition=aquila,gpu

module purge
module load miniconda
source activate danielTorch
cd ..
python loss_meta.py
python main.py {ARGS}
