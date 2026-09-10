import boto3, json
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "openneuro.org"
prefix = "ds005589/"

paginator = s3.get_paginator('list_objects_v2')
json_keys = []
data_keys = []
total_bytes = 0

for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']
        total_bytes += obj['Size']
        if key.endswith('.json') and 'eeg' in key.lower():
            json_keys.append(key)
        if '_eeg.' in key and not key.endswith('.json'):
            data_keys.append((key, obj['Size']))

print(f"Total dataset size: {total_bytes/1e9:.1f} GB")
print(f"Found {len(json_keys)} eeg.json sidecars, {len(data_keys)} data files")

print("\n--- sample data files ---")
for k, s in data_keys[:5]:
    print(f"{s/1e6:.1f} MB  {k}")

if json_keys:
    print("\n--- sample eeg.json content ---")
    body = s3.get_object(Bucket=bucket, Key=json_keys[0])['Body'].read()
    print(json_keys[0])
    print(json.dumps(json.loads(body), indent=2))