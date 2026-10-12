#!/bin/bash
# Wrapper: computes job-name/output/time for one production BIOT run and
# submits it. Same reasoning as the other submit_*.sh wrappers in this
# project -- #SBATCH lines are read before dataset/fraction/seed/arm are
# known, so those get passed as sbatch flags instead.
#
# Usage:
#   bash submit_biot_production_job.sh --dataset=chbmit --arm=pretrained --fraction=50 --seed=1
set -e

DATASET=""
ARM=""
FRACTION=""
SEED=""
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        --fraction=*) FRACTION="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
if [ -z "$DATASET" ] || [ -z "$ARM" ] || [ -z "$FRACTION" ] || [ -z "$SEED" ]; then
    echo "Usage: bash submit_biot_production_job.sh --dataset=... --arm=pretrained|random --fraction=100|50|25 --seed=1|2|42"
    exit 1
fi

# Same per-dataset time budget as the original biot scaling jobs.
case "$DATASET" in
    yang2025) TIME="04:00:00" ;;
    *) TIME="03:00:00" ;;
esac

ARM_SHORT="${ARM:0:4}"  # pret / rand
JOB_NAME="biot_prod_${DATASET}_${FRACTION}pct_s${SEED}_${ARM_SHORT}"

sbatch --job-name="$JOB_NAME" \
       --time="$TIME" \
       --output="/scratch/nourmaj/${JOB_NAME}.out" \
       --error="/scratch/nourmaj/${JOB_NAME}.err" \
       /home/nourmaj/biot_production_job.sbatch --dataset="$DATASET" --arm="$ARM" --fraction="$FRACTION" --seed="$SEED"