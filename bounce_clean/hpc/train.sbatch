#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --job-name=bounce
#SBATCH --output=slurm_train.out
#SBATCH --error=slurm_train.err
#SBATCH --partition=aquila

module purge
module load miniconda
source activate danielTorch
cd ..
python loss_meta.py
python main.py
