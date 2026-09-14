#!/bin/bash
#SBATCH --job-name=cbramod_tv_rand
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/nourmaj/cbramod_thoughtviz_random_%j.out
source ~/envs/eeg-gpu/bin/activate
export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR
cd /home/nourmaj/CBraMod
python finetune_main.py --downstream_dataset ThoughtViz --datasets_dir /scratch/nourmaj/thoughtviz_processed --num_of_classes 10 --model_dir /scratch/nourmaj/scratch_iteration2/cbramod_thoughtviz_random --batch_size 32 --seed 2 --use_pretrained_weights False
