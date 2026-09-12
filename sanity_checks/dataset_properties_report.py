"""
Per-dataset properties report: label distribution (per split, per size tier)
and a computed scale-in-hours figure, derived directly from the actual data
rather than from memory -- useful for writing up the "80h-scale vs ~2h-scale"
domain grouping with real numbers instead of a rough label.

For each dataset:
  - Reads one sample .pkl from train to get (channels, n_samples) and
    combines with that dataset's known native sampling rate (confirmed
    earlier from each dataset's own build/preprocessing script) to compute
    per-trial duration in seconds.
  - Counts every .pkl's 'y' label across train (100%/50%/25%) and val/test
    (val/test are identical across tiers, so counted once).
  - Reports total dataset duration = (train_100% + val + test trial count)
    x per-trial duration, in hours.

Run directly on the cluster: python3 dataset_properties_report.py
"""
import glob
import os
import pickle
from collections import defaultdict

# dataset key -> (100% original root, native sampling rate in Hz).
# Sampling rates are the ones confirmed when each dataset was originally
# adapted (see each build_luna_*.py / BIOT sbatch script's own comments).
# yang2025 and thoughtviz rates are taken from BIOT's --sampling_rate flag,
# not independently re-derived -- flagged with a note in the output.
DATASETS = {
    "chbmit":     ("/scratch/nourmaj/chb-mit-processed-80h",   256, True),
    "ds004504":   ("/scratch/nourmaj/ds004504_processed",      500, True),
    "isruc":      ("/scratch/nourmaj/isruc_biot_processed",    200, True),
    "dream":      ("/scratch/nourmaj/dream_processed",         500, True),
    "karaone":    ("/scratch/nourmaj/karaone_processed",       1000, True),
    "japaneeg":   ("/scratch/nourmaj/japaneeg_processed_80h",  200, True),  # corrected base: original was ~117h, now trimmed to ~80h
    "yang2025":   ("/scratch/nourmaj/yang2025_processed",      200, False),
    "thoughtviz": ("/scratch/nourmaj/thoughtviz_processed",    200, False),
}


def label_counts(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
    counts = defaultdict(int)
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        counts[int(d["y"])] += 1
    return dict(sorted(counts.items())), len(files)


def sample_shape(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
    if not files:
        return None
    with open(files[0], "rb") as fh:
        d = pickle.load(fh)
    return d["X"].shape  # (channels, n_samples)


def main():
    for ds, (original_root, sr, sr_confirmed) in DATASETS.items():
        print(f"=== {ds} ===")
        train100 = os.path.join(original_root, "train")
        train50 = f"{original_root}_50pct/train"
        train25 = f"{original_root}_25pct/train"
        val = os.path.join(original_root, "val")
        test = os.path.join(original_root, "test")

        shape = sample_shape(train100)
        if shape is None:
            print("  (no train files found, skipping)")
            continue
        n_channels, n_samples = shape
        duration_sec = n_samples / sr
        sr_note = "" if sr_confirmed else "  (sampling rate assumed from BIOT config, not independently re-verified)"

        c100, n100 = label_counts(train100)
        c50, n50 = label_counts(train50)
        c25, n25 = label_counts(train25)
        cval, nval = label_counts(val)
        ctest, ntest = label_counts(test)

        total_trials = n100 + nval + ntest
        total_hours = total_trials * duration_sec / 3600
        train_hours = n100 * duration_sec / 3600

        print(f"  channels={n_channels}  samples/trial={n_samples}  sampling_rate={sr}Hz{sr_note}")
        print(f"  trial duration = {duration_sec:.3f}s")
        print(f"  train(100%) label counts: {c100}")
        print(f"  train(50%)  label counts: {c50}")
        print(f"  train(25%)  label counts: {c25}")
        print(f"  val         label counts: {cval}")
        print(f"  test        label counts: {ctest}")
        print(f"  num_classes (from train) = {len(c100)}")
        print(f"  TOTAL dataset duration (train100%+val+test): {total_hours:.2f} hours")
        print(f"  train-only(100%) duration: {train_hours:.2f} hours")
        print()


if __name__ == "__main__":
    main()