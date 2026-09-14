#!/bin/bash

#SBATCH --job-name=cbramod_isruc_pt

#SBATCH --account=def-amiilab

#SBATCH --gpus-per-node=1

#SBATCH --time=03:00:00

#SBATCH --output=/scratch/nourmaj/cbramod_isruc_pretrained_%j.out

source ~/envs/eeg-gpu/bin/activate
export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR

cd /home/nourmaj/CBraMod

python finetune_main.py --downstream_dataset ISRUC --datasets_dir /scratch/nourmaj/isruc_processed_80h --num_of_classes 5 --lr 0.0001 --weight_decay 0.05 --epochs 25 --batch_size 64 --dropout 0.1 --model_dir /scratch/nourmaj/cbramod_isruc_pretrained --use_pretrained_weights True --foundation_dir pretrained_weights/pretrained_weights.pth

