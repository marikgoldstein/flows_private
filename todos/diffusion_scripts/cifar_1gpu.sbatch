#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=25GB
#SBATCH --job-name=cifar
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --gres=gpu:a100:1

SING_LOCATION="$SCRATCH/overlay_files/overlay-50G-10M.ext3"
SING_CUDA_LOCATION="/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif"
SING_ENV_FILE="/ext3/env.sh"
CONDA_ENV_NAME="new"
ROOT_DIR="/scratch/mg3479/variance_reduced_dgms"
RESUME=0
WANDB=1
DEBUG=0
singularity exec --nv --overlay ${SING_LOCATION}:ro ${SING_CUDA_LOCATION} /bin/bash -c "
cd ${ROOT_DIR}
source ${SING_ENV_FILE}
conda activate ${CONDA_ENV_NAME}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 train.py --index ${SLURM_ARRAY_TASK_ID} --do_resume ${RESUME} --debug ${DEBUG} --dataset cifar --use_wandb ${WANDB}
"
