#!/bin/bash
#SBATCH --job-name=reve_dream_random
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/nourmaj/reve_dream_random_%j.out
#SBATCH --error=/scratch/nourmaj/reve_dream_random_%j.err
source ~/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/nourmaj/REVE/src
mkdir -p $SCRATCH/reve_outputs/dream_random
ln -sfn $SCRATCH/hf_cache $SCRATCH/reve_outputs/dream_random/.cache
python dt.py task=dream training_mode=ft task.data_loader.dataset.path=$SCRATCH/dream_processed task.fine_tuning.optimizer.lr=0.00003 seed=42 hydra.run.dir=$SCRATCH/reve_outputs/dream_random cache_dir=$SCRATCH/hf_cache
