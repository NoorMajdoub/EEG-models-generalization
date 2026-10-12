#!/bin/bash
# Submits all 12 BIOT Optuna studies: 6 datasets x 2 arms.
# (isruc and thoughtviz are excluded -- they use run_multiclass_supervised.py,
# not yet added to biot_optuna_search.py.)
#
# Usage:
#   bash submit_all_biot_optuna_jobs.sh          # submits all 12
#   bash submit_all_biot_optuna_jobs.sh chbmit   # submits just chbmit's 2 (pretrained+random)
set -e

DATASETS=(chbmit clinical dream karaone yang2025 japaneeg)
ARMS=(pretrained random)

FILTER="${1:-}"
count=0
for ds in "${DATASETS[@]}"; do
    if [ -n "$FILTER" ] && [ "$ds" != "$FILTER" ]; then
        continue
    fi
    for arm in "${ARMS[@]}"; do
        bash submit_biot_optuna_job.sh --dataset="$ds" --arm="$arm"
        count=$((count + 1))
    done
done
echo "Submitted $count Optuna study jobs total."