#!/bin/bash
#SBATCH --account=def-amiilab
#SBATCH --time=0-00:30:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=csbrain_test
#SBATCH --output=csbrain_test_%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
source ~/envs/eeg-gpu/bin/activate

cd ~/CSBrain

python finetune_main.py \
    --downstream_dataset CHB-MIT \
    --seed 0 \
    --datasets_dir $SCRATCH/chb-mit-processed-full \
    --num_of_classes 2 \
    --model_dir $SCRATCH/csbrain_out/CHB-MIT/random_test \
    --foundation_dir ~/CSBrain/pth/CSBrain.pth \
    --model CSBrain \
    --dropout 0.1 \
    --weight_decay 0.05 \
    --lr 0.0001 \
    --epochs 1