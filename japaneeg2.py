import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "openneuro.org"
prefix = "ds007808/sub-01/"

paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if '_eeg.' in key and not key.endswith('.json'):
            print(f"{obj['Size']/1e6:.1f} MB  {key}")