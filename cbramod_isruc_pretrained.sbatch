#!/bin/bash
#SBATCH --account=def-amiilab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --job-name=cbramod_isruc_pretrained
#SBATCH --output=/scratch/nourmaj/cbramod_isruc_pretrained_%j.log

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
source ~/envs/eeg-gpu/bin/activate
export MPLCONFIGDIR=/tmp/mplconfig
export NUMBA_CACHE_DIR=/tmp/numbacache
export XDG_CACHE_HOME=/tmp/xdgcache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $MPLCONFIGDIR $NUMBA_CACHE_DIR $XDG_CACHE_HOME

cd ~/CBraMod
python finetune_main.py --downstream_dataset ISRUC --num_of_classes 5 --epochs 25 --batch_size 8 --datasets_dir $SCRATCH/isruc_processed --model_dir $SCRATCH/cbramod_out/ISRUC/pretrained_seed0 --foundation_dir pretrained_weights/pretrained_weights.pth --use_pretrained_weights "True"