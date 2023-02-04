#!/bin/bash

#SBATCH --mail-user={USER}@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
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
