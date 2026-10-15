/home/nourmaj/BIOT/biot_optuna_search.py
/home/nourmaj/BIOT/biot_production_finetune.py
/home/nourmaj/biot_optuna_job.sbatch
/home/nourmaj/biot_production_job.sbatch
/home/nourmaj/submit_all_biot_optuna_jobs.sh
/home/nourmaj/submit_all_biot_optuna_50_25_jobs.sh
/home/nourmaj/submit_biot_optuna_job.sh
/home/nourmaj/submit_all_biot_production_jobs.sh
/home/nourmaj/submit_biot_production_job.sh


--------------
### CAUEEG 
How to create the smaller datasets splits : 
cd /scratch/nourmaj/caueeg/caueeg-dataset

python3 cap_caueeg_train.py --target-hours 50 --seed 42 --out dementia_train_capped_50h.json --out-csv dementia_train_capped_50h_manifest.csv --split-dir splits

python3 cap_caueeg_train.py --target-hours 25 --seed 42 --out dementia_train_capped_25h.json --out-csv dementia_train_capped_25h_manifest.csv --split-dir splits_50pct

python3 cap_caueeg_train.py --target-hours 12.5 --seed 42 --out dementia_train_capped_12_5h.json --out-csv dementia_train_capped_12_5h_manifest.csv --split-dir splits_25pct

++ output to confirm you split the datasets correctly :
splits: train=229 val=82 test=90
splits_50pct: train=117 val=82 test=90
splits_25pct: train=58 val=82 test=90

