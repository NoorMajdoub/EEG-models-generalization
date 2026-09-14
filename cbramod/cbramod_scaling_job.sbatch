#!/bin/bash
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#
# One shared CBraMod scaling-run job script for every dataset/fraction/seed/
# arm combination -- do NOT sbatch this directly, use submit_cbramod_job.sh.
# Like REVE, CBraMod needed no repo code changes -- it already takes a
# clean --datasets_dir CLI flag.
#
# Only 4 datasets configured so far: chbmit, karaone, dream, thoughtviz.
# clinical (ds004504) is NOT included yet -- every script found so far
# claiming to be the Clinical CBraMod script turned out to actually be a
# CHB-MIT script in disguise (checked directly, not assumed), so the real
# one is still needed. isruc is parked too (same stale-path issue found in
# REVE's isruc script). Both pending, add later once resolved.
#
# Usage (via the wrapper, not directly):
#   bash submit_cbramod_job.sh --dataset=chbmit --fraction=50 --seed=1 --arm=pretrained

set -e

DATASET=""; FRACTION=""; SEED=""; ARM=""
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --fraction=*) FRACTION="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
for v in DATASET FRACTION SEED ARM; do
    if [ -z "${!v}" ]; then echo "Missing required --${v,,}=... argument"; exit 1; fi
done
if [ "$ARM" != "pretrained" ] && [ "$ARM" != "random" ]; then
    echo "--arm must be 'pretrained' or 'random', got: $ARM"; exit 1
fi
if [ "$FRACTION" != "100" ] && [ "$FRACTION" != "50" ] && [ "$FRACTION" != "25" ]; then
    echo "--fraction must be 100, 50, or 25, got: $FRACTION"; exit 1
fi

# Per-dataset config, copied EXACTLY from each dataset's own confirmed
# working pretrained/random scripts. COMMON_FLAGS applies to both arms;
# PRETRAINED_EXTRA / RANDOM_EXTRA hold what differs between them.
case "$DATASET" in
    chbmit)
        DS_NAME="CHB-MIT"; NUM_CLASSES=2
        ORIGINAL_PATH="/scratch/nourmaj/chb-mit-processed-80h"
        COMMON_FLAGS="--lr 0.0001 --weight_decay 0.05 --epochs 25 --batch_size 64 --dropout 0.1"
        PRETRAINED_EXTRA="--use_pretrained_weights True --foundation_dir pretrained_weights/pretrained_weights.pth"
        RANDOM_EXTRA="--use_pretrained_weights False"
        ;;
    karaone)
        DS_NAME="KaraOne"; NUM_CLASSES=2
        ORIGINAL_PATH="/scratch/nourmaj/karaone_processed"
        COMMON_FLAGS="--batch_size 32"
        PRETRAINED_EXTRA="--use_pretrained_weights True --lr 5e-5"
        RANDOM_EXTRA="--use_pretrained_weights False"
        ;;
    dream)
        DS_NAME="Dream"; NUM_CLASSES=2
        ORIGINAL_PATH="/scratch/nourmaj/dream_processed"
        COMMON_FLAGS="--batch_size 16"
        PRETRAINED_EXTRA="--use_pretrained_weights True --lr 5e-5"
        RANDOM_EXTRA="--use_pretrained_weights False"
        ;;
    thoughtviz)
        DS_NAME="ThoughtViz"; NUM_CLASSES=10
        ORIGINAL_PATH="/scratch/nourmaj/thoughtviz_processed"
        COMMON_FLAGS="--batch_size 32"
        PRETRAINED_EXTRA="--use_pretrained_weights True --lr 5e-5"
        RANDOM_EXTRA="--use_pretrained_weights False"
        ;;
    clinical)
        echo "clinical (ds004504) not configured yet -- the real script is still needed, every candidate so far turned out to be CHB-MIT in disguise."
        exit 1
        ;;
    isruc)
        echo "isruc not configured yet -- same stale-path issue found in REVE's isruc script. Pending."
        exit 1
        ;;
    *)
        echo "Unknown --dataset=$DATASET"; exit 1
        ;;
esac

if [ "$FRACTION" == "100" ]; then
    DATA_PATH="$ORIGINAL_PATH"
else
    DATA_PATH="${ORIGINAL_PATH}_${FRACTION}pct"
fi
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: expected data folder not found: $DATA_PATH"
    exit 1
fi

if [ "$ARM" == "pretrained" ]; then
    ARM_FLAGS="$PRETRAINED_EXTRA"
else
    ARM_FLAGS="$RANDOM_EXTRA"
fi

MODEL_DIR="/scratch/nourmaj/scratch_iteration2/cbramod_scaling_results/${DATASET}_${FRACTION}pct_s${SEED}_${ARM}"
mkdir -p "$MODEL_DIR"

echo "=== CBraMod scaling run ==="
echo "dataset=$DATASET fraction=$FRACTION seed=$SEED arm=$ARM"
echo "data_path=$DATA_PATH"
echo "model_dir=$MODEL_DIR"
echo "============================"

source ~/envs/eeg-gpu/bin/activate
export HOME=/scratch/nourmaj/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/scratch/nourmaj/numba_cache
mkdir -p $NUMBA_CACHE_DIR
cd /home/nourmaj/CBraMod

python finetune_main.py --downstream_dataset "$DS_NAME" --datasets_dir "$DATA_PATH" \
    --num_of_classes $NUM_CLASSES $COMMON_FLAGS --model_dir "$MODEL_DIR" \
    --seed $SEED $ARM_FLAGS