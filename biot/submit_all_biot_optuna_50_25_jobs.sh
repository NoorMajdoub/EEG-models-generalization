#!/bin/bash
# Submits the FOLLOW-UP BIOT Optuna studies: 6 datasets x 2 arms x 2
# fractions (50, 25) = 24 studies. 100% was already tuned in the first
# round (chbmit/clinical/dream/karaone/yang2025/japaneeg x pretrained/random)
# -- this script deliberately does NOT redo those.
#
# Usage:
#   bash submit_all_biot_optuna_50_25_jobs.sh          # submits all 24
#   bash submit_all_biot_optuna_50_25_jobs.sh chbmit   # submits just chbmit's 4 (2 arms x 2 fractions)
set -e

DATASETS=(chbmit clinical dream karaone yang2025 japaneeg)
ARMS=(pretrained random)
FRACTIONS=(50 25)

FILTER="${1:-}"
count=0
for ds in "${DATASETS[@]}"; do
    if [ -n "$FILTER" ] && [ "$ds" != "$FILTER" ]; then
        continue
    fi
    for arm in "${ARMS[@]}"; do
        for frac in "${FRACTIONS[@]}"; do
            bash submit_biot_optuna_job.sh --dataset="$ds" --arm="$arm" --fraction="$frac"
            count=$((count + 1))
        done
    done
done
echo "Submitted $count follow-up Optuna study jobs total."