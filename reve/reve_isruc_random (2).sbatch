#!/bin/bash
#SBATCH --job-name=reve_isruc_random
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/nourmaj/reve_isruc_random_%j.out
#SBATCH --error=/scratch/nourmaj/reve_isruc_random_%j.err

source ~/envs/reve-gpu/bin/activate

export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/REVE/src

mkdir -p $SCRATCH/reve_outputs/isruc_random
ln -sfn $SCRATCH/hf_cache $SCRATCH/reve_outputs/isruc_random/.cache
python dt.py \
  task=isruc \
  training_mode=ft \
  task.data_loader.dataset.path=$SCRATCH/isruc_processed_80h \
  task.fine_tuning.n_epochs=25 \
  seed=42 \
  hydra.run.dir=$SCRATCH/reve_outputs/isruc_random \
  cache_dir=$SCRATCH/hf_cache

