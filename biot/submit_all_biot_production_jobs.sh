#!/bin/bash
# Submits the full BIOT production grid: 6 datasets x 3 fractions x 3 seeds
# x 2 arms = 108 jobs, each using that (dataset, arm)'s winning
# hyperparameters from the Optuna search.
#
# Usage:
#   bash submit_all_biot_production_jobs.sh          # submits all 108
#   bash submit_all_biot_production_jobs.sh chbmit   # submits just chbmit's 18 (3 fractions x 3 seeds x 2 arms)
set -e

DATASETS=(chbmit clinical dream karaone yang2025 japaneeg)
FRACTIONS=(100 50 25)
SEEDS=(1 2 42)
ARMS=(pretrained random)

FILTER="${1:-}"
count=0
for ds in "${DATASETS[@]}"; do
    if [ -n "$FILTER" ] && [ "$ds" != "$FILTER" ]; then
        continue
    fi
    for frac in "${FRACTIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for arm in "${ARMS[@]}"; do
                bash submit_biot_production_job.sh --dataset="$ds" --arm="$arm" --fraction="$frac" --seed="$seed"
                count=$((count + 1))
            done
        done
    done
done
echo "Submitted $count production job(s) total."