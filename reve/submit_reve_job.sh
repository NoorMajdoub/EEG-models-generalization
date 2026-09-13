#!/bin/bash
# Wrapper around `sbatch` for reve_scaling_job.sbatch -- computes the right
# --job-name / --output / --time for the requested dataset+arm (time limits
# vary by both, e.g. chbmit pretrained=6:30h vs chbmit random=5:30h, copied
# exactly from each dataset's existing working scripts) and submits the job.
#
# Usage:
#   bash submit_reve_job.sh --dataset=chbmit --fraction=50 --seed=1 --arm=pretrained
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET=""
FRACTION=""
SEED=""
ARM=""
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
    if [ -z "${!v}" ]; then
        echo "Missing required --${v,,}=... argument"; exit 1
    fi
done

# Per-dataset (and, for chbmit, per-arm) time limits, matching each
# dataset's existing working scripts exactly.
case "$DATASET" in
    chbmit)
        if [ "$ARM" == "pretrained" ]; then TIME="06:30:00"; else TIME="05:30:00"; fi
        ;;
    clinical|dream|karaone|thoughtviz) TIME="02:00:00" ;;
    yang2025) TIME="03:00:00" ;;
    japaneeg) TIME="04:00:00" ;;
    isruc) echo "isruc not configured for REVE scaling yet -- see reve_scaling_job.sbatch"; exit 1 ;;
    *) echo "Unknown --dataset=$DATASET"; exit 1 ;;
esac

ARM_SHORT="${ARM:0:4}"
JOB_NAME="reve_${DATASET}_${FRACTION}pct_s${SEED}_${ARM_SHORT}"
OUT_FILE="/scratch/nourmaj/${JOB_NAME}_%j.out"
ERR_FILE="/scratch/nourmaj/${JOB_NAME}_%j.err"

echo "Submitting: $JOB_NAME (time=$TIME)"
sbatch --job-name="$JOB_NAME" --output="$OUT_FILE" --error="$ERR_FILE" --time="$TIME" \
    "$SCRIPT_DIR/reve_scaling_job.sbatch" \
    --dataset="$DATASET" --fraction="$FRACTION" --seed="$SEED" --arm="$ARM"