import boto3, json, io
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "openneuro.org"
prefix = "ds007808/"

paginator = s3.get_paginator('list_objects_v2')
total_seconds_by_subject = {}

for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('_eeg.json'):
            body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
            meta = json.loads(body)
            duration = meta.get('RecordingDuration')
            if duration is not None:
                subj = key.split('/')[1]  # e.g. "sub-01"
                total_seconds_by_subject[subj] = total_seconds_by_subject.get(subj, 0) + duration
                print(f"{key}: {duration/3600:.2f}h")

print("\n--- totals ---")
for subj, secs in total_seconds_by_subject.items():
    print(f"{subj}: {secs/3600:.2f}h")