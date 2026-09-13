#!/bin/bash
# Submits the full REVE scaling grid: every dataset (except isruc, pending)
# x every fraction x every seed x both arms, via submit_reve_job.sh.
#
# Usage:
#   bash submit_all_reve_scaling_jobs.sh
# Or just one dataset first (recommended before submitting all ~84):
#   bash submit_all_reve_scaling_jobs.sh chbmit
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(chbmit clinical dream karaone thoughtviz yang2025 japaneeg)
FRACTIONS=(50 25)
SEEDS=(1 2 42)
ARMS=(pretrained random)

if [ $# -ge 1 ]; then
    DATASETS=("$@")
fi

n=0
for ds in "${DATASETS[@]}"; do
    for pct in "${FRACTIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for arm in "${ARMS[@]}"; do
                bash "$SCRIPT_DIR/submit_reve_job.sh" \
                    --dataset="$ds" --fraction="$pct" --seed="$seed" --arm="$arm"
                n=$((n+1))
            done
        done
    done
done
echo "Submitted $n jobs total."