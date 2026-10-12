#!/bin/bash
# Wrapper: computes job-name/output for one (dataset, arm, fraction) BIOT
# Optuna study and submits it.
#
# Usage:
#   bash submit_biot_optuna_job.sh --dataset=chbmit --arm=pretrained --fraction=50 [--n_trials=20]
set -e

DATASET=""
ARM=""
FRACTION="100"
N_TRIALS="20"
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        --fraction=*) FRACTION="${arg#*=}" ;;
        --n_trials=*) N_TRIALS="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
if [ -z "$DATASET" ] || [ -z "$ARM" ]; then
    echo "Usage: bash submit_biot_optuna_job.sh --dataset=... --arm=pretrained|random --fraction=100|50|25 [--n_trials=20]"
    exit 1
fi

ARM_SHORT="${ARM:0:4}"  # pret / rand
JOB_NAME="biot_optuna_${DATASET}_${FRACTION}pct_${ARM_SHORT}"

sbatch --job-name="$JOB_NAME" \
       --output="/scratch/nourmaj/${JOB_NAME}.out" \
       --error="/scratch/nourmaj/${JOB_NAME}.err" \
       /home/nourmaj/biot_optuna_job.sbatch --dataset="$DATASET" --arm="$ARM" --fraction="$FRACTION" --n_trials="$N_TRIALS"