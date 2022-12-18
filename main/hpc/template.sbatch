#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
#SBATCH --job-name={JOB_NAME}
#SBATCH --output={OUT_FILENAME}.out
#SBATCH --error={OUT_FILENAME}.err

module purge
source /scratch/$USER/py_env/masdar/select_images
singularity exec --nv --overlay /scratch/$USER/py_env/$ENV_NAME/$OVERLAY:ro /scratch/work/public/singularity/$SINGULARITY /bin/bash -c 'source /ext3/env.sh; PATH=$HOME/ffmpeg_4.2.2-1ubuntu1_amd64/usr/bin:$PATH; cd ..; python loss_meta.py; python main.py {ARGS}'
