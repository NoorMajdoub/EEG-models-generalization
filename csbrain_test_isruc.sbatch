#!/bin/bash
#SBATCH --account=def-amiilab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:30:00
#SBATCH --job-name=csbrain_isruc_pretrained
#SBATCH --output=/scratch/nourmaj/csbrain_isruc_pretrained_%j.log

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
source ~/envs/eeg-gpu/bin/activate
export MPLCONFIGDIR=/tmp/mplconfig
export NUMBA_CACHE_DIR=/tmp/numbacache
export XDG_CACHE_HOME=/tmp/xdgcache
mkdir -p $MPLCONFIGDIR $NUMBA_CACHE_DIR $XDG_CACHE_HOME

cd ~/CSBrain
python finetune_main.py --downstream_dataset ISRUC --num_of_classes 5 --epochs 1 --datasets_dir $SCRATCH/isruc_processed --model_dir $SCRATCH/csbrain_out/ISRUC/pretrained_seed0 --foundation_dir pth/CSBrain.pth --use_pretrained_weights