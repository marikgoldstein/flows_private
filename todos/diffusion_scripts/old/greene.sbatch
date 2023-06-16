#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=25GB
#SBATCH --job-name=const
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14
#SBATCH --gres=gpu:1

SING_LOCATION="$SCRATCH/overlay_files/overlay-50G-10M.ext3"
SING_CUDA_LOCATION="/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif"
SING_ENV_FILE="/ext3/env.sh"
CONDA_ENV_NAME="new"
ROOT_DIR="/scratch/mg3479/variance_reduced_dgms"
RESUME=0
CKPT_NAME="const"
singularity exec --nv --overlay ${SING_LOCATION}:ro ${SING_CUDA_LOCATION} /bin/bash -c "
cd ${ROOT_DIR}
source ${SING_ENV_FILE}
conda activate ${CONDA_ENV_NAME}
python main.py --index ${SLURM_ARRAY_TASK_ID} --do_resume ${RESUME} --dgm_type diffusion --debug 0 --ckpt_name ${CKPT_NAME}
"
