#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:01:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=bounce
#SBATCH --output=slurm_train.out
#SBATCH --error=slurm_train.err
#SBATCH --partition=debug

module purge
module load miniconda
source activate danielTorch
python main.py
