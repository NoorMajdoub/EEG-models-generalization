#!/bin/bash
#SBATCH --account=def-amiilab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:30:00
#SBATCH --job-name=csbrain_faced_pretrained
#SBATCH --output=/scratch/nourmaj/csbrain_faced_pretrained_%j.log

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
source ~/envs/eeg-gpu/bin/activate
export MPLCONFIGDIR=/tmp/mplconfig
export NUMBA_CACHE_DIR=/tmp/numbacache
export XDG_CACHE_HOME=/tmp/xdgcache
mkdir -p $MPLCONFIGDIR $NUMBA_CACHE_DIR $XDG_CACHE_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/CSBrain
python finetune_main.py --downstream_dataset FACED --num_of_classes 9 --epochs 25 --datasets_dir $SCRATCH/FACED/processed --model_dir $SCRATCH/csbrain_out/FACED/pretrained_seed0 --batch_size 8 --foundation_dir pth/CSBrain.pth --use_pretrained_weights