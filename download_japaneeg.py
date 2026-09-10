import boto3, os
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "openneuro.org"
local_root = os.path.join(os.environ["SCRATCH"], "raw_datasets", "japaneeg")

# 1. get the list of .edf keys to include, from your duration log
target_keys = []
with open("japaneeg_duration_log.txt") as f:
    for line in f:
        cum_hours = float(line.split("h cumulative")[0].strip())
        if cum_hours > 149.78:
            break
        key = line.strip().split("|")[-1].strip()
        target_keys.append(key)

print(f"{len(target_keys)} EDF files selected, ~149.78h")

# 2. for each .edf, also grab its sibling metadata files in the same folder
suffixes = ["_eeg.edf", "_eeg.json", "_channels.tsv", "_events.tsv"]
all_keys_to_download = set()
for edf_key in target_keys:
    base = edf_key.replace("_eeg.edf", "")
    for suf in suffixes:
        all_keys_to_download.add(base + suf)

# 3. also grab top-level dataset metadata (tiny files)
for meta_key in ["ds007808/dataset_description.json", "ds007808/participants.tsv", "ds007808/participants.json"]:
    all_keys_to_download.add(meta_key)

print(f"{len(all_keys_to_download)} total files to download (data + metadata)")

# 4. download, skipping anything missing (some siblings may not exist for every run)
for key in sorted(all_keys_to_download):
    local_path = os.path.join(local_root, key[len("ds007808/"):])
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3.download_file(bucket, key, local_path)
        print("downloaded", key)
    except Exception as e:
        print("skip (not found):", key)