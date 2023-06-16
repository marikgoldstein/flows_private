#!/bin/bash
#SBATCH --partition=gpu4_short,gpu4_medium,gpu4_long,gpu8_dev,gpu8_short,gpu8_medium,gpu8_long
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=25G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=cifar
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5

ROOT_DIR="/gpfs/data/ranganathlab/mark/variance_reduced_dgms"
RESUME=0
WAND=1
DEBUG=0
CKPT_NAME="cow"
ARCH="bigunet"

cd ${ROOT_DIR}

command="python main.py --index ${SLURM_ARRAY_TASK_ID} --arch ${ARCH} --dataset cifar --do_resume ${RESUME} --debug ${DEBUG} --save_ckpt_group_name ${CKPT_NAME} --use_wandb ${WAND}"

${command}



