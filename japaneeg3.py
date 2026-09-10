import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "openneuro.org"
prefix = "ds007808/"

BYTES_PER_HOUR = 134 * 2 * 1200 * 3600  # ~1158 MB

paginator = s3.get_paginator('list_objects_v2')
running_total_bytes = 0
rows = []

for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('_eeg.edf'):
            running_total_bytes += obj['Size']
            rows.append((key, obj['Size'], running_total_bytes / BYTES_PER_HOUR))

with open("japaneeg_duration_log.txt", "w") as f:
    for key, size, cum_hours in rows:
        line = f"{cum_hours:8.2f}h cumulative  |  {size/1e6:7.1f} MB  |  {key}\n"
        f.write(line)

print(f"Total files: {len(rows)}")
print(f"Grand total: {running_total_bytes / BYTES_PER_HOUR:.1f} hours")
print("Full breakdown written to japaneeg_duration_log.txt")