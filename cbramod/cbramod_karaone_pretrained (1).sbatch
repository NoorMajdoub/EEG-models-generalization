#!/bin/bash
#SBATCH --job-name=cbramod_karaone_pt
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/nourmaj/cbramod_karaone_pretrained_%j.out
source ~/envs/eeg-gpu/bin/activate
export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR
cd /home/nourmaj/CBraMod
python finetune_main.py --downstream_dataset KaraOne --datasets_dir /scratch/nourmaj/karaone_processed --num_of_classes 2 --model_dir /scratch/nourmaj/scratch_iteration2/cbramod_karaone_pretrained --batch_size 32 --seed 42 --use_pretrained_weights True --lr 5e-5
