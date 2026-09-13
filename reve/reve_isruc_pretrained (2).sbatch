#!/bin/bash
#SBATCH --job-name=reve_isruc_pretrained
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/nourmaj/reve_isruc_pretrained_%j.out
#SBATCH --error=/scratch/nourmaj/reve_isruc_pretrained_%j.err

source ~/envs/reve-gpu/bin/activate

export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "DEBUG HF_HUB_CACHE=$HF_HUB_CACHE HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
cd ~/REVE/src

mkdir -p $SCRATCH/reve_outputs/isruc_pretrained
ln -sfn $SCRATCH/hf_cache $SCRATCH/reve_outputs/isruc_pretrained/.cache
python dt.py \
  task=isruc \
  training_mode=ft \
  pretrained_path=hf:brain-bzh/reve-base \
  task.data_loader.dataset.path=$SCRATCH/isruc_processed_80h \
  task.fine_tuning.n_epochs=25 \
  seed=42 \
  hydra.run.dir=$SCRATCH/reve_outputs/isruc_pretrained \
  cache_dir=$SCRATCH/hf_cache
