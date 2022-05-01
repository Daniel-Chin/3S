#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval-z
#SBATCH --output=slurm_eval_z.out

module purge

singularity exec --nv \
	    --overlay /scratch/nq285/myEnv/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python evalZ.py"
