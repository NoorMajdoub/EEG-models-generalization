"""
Cap an already-built CHB-MIT train/ folder down to a target number of hours,
preserving the original positive/negative (seizure/non-seizure) ratio.

Val and test are NEVER touched by this script — it only reads from the full
train/ folder and writes symlinks into a new, separate output folder.

Env vars expected:
  CHB_TRAIN_SRC   - path to the FULL train folder (e.g. $SCRATCH/chb-mit-processed-full/train)
  CHB_TRAIN_OUT   - path to write the capped subset into (e.g. $SCRATCH/chb-mit-processed-100h/train)
  CHB_TARGET_HOURS - target hours for this subset (e.g. 100)
  CHB_SEED        - random seed, default 42
"""

import os
import pickle
import random

SRC = os.environ["CHB_TRAIN_SRC"]
OUT = os.environ["CHB_TRAIN_OUT"]
TARGET_HOURS = float(os.environ["CHB_TARGET_HOURS"])
SEED = int(os.environ.get("CHB_SEED", "42"))

SECONDS_PER_SEGMENT = 10
SEGMENTS_NEEDED = int(TARGET_HOURS * 3600 / SECONDS_PER_SEGMENT)

os.makedirs(OUT, exist_ok=True)

print(f"Scanning {SRC} ...")
all_files = [f for f in os.listdir(SRC) if f.endswith(".pkl")]
print(f"Found {len(all_files)} total segments in full train set")

# classify each file by its label without loading the full array into memory twice
pos_files, neg_files = [], []
for f in all_files:
    with open(os.path.join(SRC, f), "rb") as fh:
        record = pickle.load(fh)
    if record["y"] == 1:
        pos_files.append(f)
    else:
        neg_files.append(f)

pos_ratio = len(pos_files) / len(all_files)
print(f"Full train set: {len(pos_files)} positive, {len(neg_files)} negative "
      f"({pos_ratio*100:.3f}% positive)")

target_pos = round(SEGMENTS_NEEDED * pos_ratio)
target_neg = SEGMENTS_NEEDED - target_pos

if target_pos > len(pos_files):
    raise ValueError(
        f"Need {target_pos} positive segments to hit {TARGET_HOURS}h at the "
        f"original ratio, but only {len(pos_files)} exist in the full set. "
        f"Lower CHB_TARGET_HOURS or accept a different ratio."
    )
if target_neg > len(neg_files):
    raise ValueError(
        f"Need {target_neg} negative segments, but only {len(neg_files)} exist."
    )

random.seed(SEED)
selected_pos = random.sample(pos_files, target_pos)
selected_neg = random.sample(neg_files, target_neg)
selected = selected_pos + selected_neg
random.shuffle(selected)

print(f"Selecting {len(selected)} segments "
      f"({target_pos} positive, {target_neg} negative) "
      f"≈ {len(selected) * SECONDS_PER_SEGMENT / 3600:.2f}h")

for f in selected:
    src_path = os.path.abspath(os.path.join(SRC, f))
    dst_path = os.path.join(OUT, f)
    if not os.path.exists(dst_path):
        os.symlink(src_path, dst_path)

print(f"Done. {len(selected)} symlinks written to {OUT}")
print(f"Actual positive ratio in subset: {len(selected_pos)/len(selected)*100:.3f}%")