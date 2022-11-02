#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=continue
#SBATCH --output=continue.out
#SBATCH --error=continue.err
#SBATCH --partition=aquila,gpu

module purge
module load miniconda
source activate danielTorch
cd ..
python loss_meta.py
python continue.py
