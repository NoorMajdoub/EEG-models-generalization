"""
CHB-MIT raw -> cleaned -> 10s segmented pkl dataset.
Combines the original cell-1 (channel cleaning) and cell-2 (segmenting/labeling)
into one script, with paths driven by env vars so it works on Trillium without editing.

Env vars expected (set these before running, or export in the sbatch script):
  CHB_RAW      - path to the raw dataset root, i.e. the folder that directly
                 contains chb01/, chb02/, ... chb01-summary.txt etc.
                 e.g. $SCRATCH/chb-mit-raw/chb-mit-scalp-eeg-database-1.0.0
  CHB_CLEAN    - scratch/disposable intermediate folder for cleaned per-file pkls
                 e.g. $SCRATCH/chb-mit-clean
  CHB_OUT      - final persistent output folder (train/val/test segments)
                 e.g. $PROJECT/chb-mit-processed-full
  CHB_NPROC    - number of worker processes to use (should match --cpus-per-task)
"""

import os
from collections import defaultdict
import pyedflib.highlevel as hl
import numpy as np
import pickle
import multiprocessing as mp
from tqdm import tqdm

signals_path = os.environ["CHB_RAW"]
clean_path = os.environ["CHB_CLEAN"]
out_path = os.environ["CHB_OUT"]
NPROC = int(os.environ.get("CHB_NPROC", mp.cpu_count()))

os.makedirs(clean_path, exist_ok=True)
os.makedirs(out_path, exist_ok=True)


# ---------- shared helpers ----------

def compressed_pickle(title, data):
    pickle.dump(data, open(title, "wb"))


def process_metadata(summary, filename):
    f = open(summary, "r")
    metadata = {}
    lines = f.readlines()
    times = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2] == filename:
            j = i + 1
            processed = False
            while not processed:
                if lines[j].split()[0] == "Number":
                    seizures = int(lines[j].split()[-1])
                    processed = True
                j = j + 1

            if seizures > 0:
                j = i + 1
                for s in range(seizures):
                    processed = False
                    while not processed:
                        l = lines[j].split()
                        if l[0] == "Seizure" and "Start" in l:
                            start = int(l[-2]) * 256 - 1
                            end = int(lines[j + 1].split()[-2]) * 256 - 1
                            processed = True
                        j = j + 1
                    times.append((start, end))

            metadata["seizures"] = seizures
            metadata["times"] = times

    return metadata


def drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None):
    signals, signal_headers, header = hl.read_edf(edf_source, ch_nrs=to_keep, digital=False)
    clean_file = {}
    for signal, header in zip(signals, signal_headers):
        channel = header.get("label")
        if channel in clean_file.keys():
            channel = channel + "-2"
        clean_file[channel] = signal
    return clean_file


def move_channels(clean_dict, channels, target):
    keys_to_delete = []
    for key in clean_dict:
        if key != "metadata" and key not in channels.keys():
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del clean_dict[key]

    size = 0
    for item in clean_dict.keys():
        if item != "metadata":
            size = len(clean_dict.get(item))
            break

    for k in channels.keys():
        if k not in clean_dict.keys():
            clean_dict[k] = np.zeros(size, dtype=float)

    compressed_pickle(target + ".pkl", clean_dict)


def process_files(pacient, valid_channels, channels, start, end):
    for num in range(start, end + 1):
        to_keep = []
        num = ("0" + str(num))[-2:]
        filename = "{path}/chb{p}/chb{p}_{n}.edf".format(path=signals_path, p=pacient, n=num)

        try:
            signals, signal_headers, header = hl.read_edf(filename, digital=False)
            n = 0
            for h in signal_headers:
                if h.get("label") in valid_channels:
                    if n not in to_keep:
                        to_keep.append(n)
                n = n + 1
        except OSError:
            print(f"WARNING: file {filename} does not exist, skipping.")
            continue

        if len(to_keep) > 0:
            try:
                print("Removing", len(signal_headers) - len(to_keep), "channels from",
                      f"chb{pacient}_{num}.edf")
                clean_dict = drop_channels(filename, to_keep=to_keep)
                print("Processing file", filename)
            except AssertionError:
                print(f"WARNING: file {filename} does not exist, skipping.")
                continue

        metadata = process_metadata(
            "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
            "chb{p}_{n}.edf".format(p=pacient, n=num),
        )
        metadata["channels"] = valid_channels
        clean_dict["metadata"] = metadata
        target = "{path}/chb{p}/chb{p}_{n}.edf".format(path=clean_path, p=pacient, n=num)
        move_channels(clean_dict, channels, target)


def start_process(pacient, num, start, end, sum_ind):
    f = open("{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient), "r")

    channels = defaultdict(list)
    valid_channels = []
    to_keep = []

    channel_index = 1
    summary_index = 0

    for line in f:
        line = line.split()
        if len(line) == 0:
            continue

        if line[0] == "Channels" and line[1] == "changed:":
            summary_index += 1

        if (line[0] == "Channel" and summary_index == sum_ind
                and (line[2] != "-" and line[2] != ".")):
            if line[2] in channels.keys():
                name = line[2] + "-2"
            else:
                name = line[2]

            channels[name].append(str(channel_index))
            channel_index += 1
            valid_channels.append(name)
            to_keep.append(int(line[1][:-1]) - 1)

    filename = "{path}/chb{p}/chb{p}_{n}.edf".format(path=signals_path, p=pacient, n=num)
    target = "{path}/chb{p}/chb{p}_{n}.edf".format(path=clean_path, p=pacient, n=num)

    os.makedirs("{path}/chb{p}".format(p=pacient, path=clean_path), exist_ok=True)

    clean_dict = drop_channels(filename, to_keep=to_keep)

    metadata = process_metadata(
        "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
        "chb{p}_{n}.edf".format(p=pacient, n=num),
    )

    metadata["channels"] = valid_channels
    clean_dict["metadata"] = metadata

    compressed_pickle(target + ".pkl", clean_dict)

    process_files(pacient, valid_channels, channels, start, end)


# ---------- stage 1: clean ----------

PARAMETERS = [
    ("01", "01", 2, 46, 0),
    ("02", "01", 2, 35, 0),
    ("03", "01", 2, 38, 0),
    ("05", "01", 2, 39, 0),
    ("06", "01", 2, 24, 0),
    ("07", "01", 2, 19, 0),
    ("08", "02", 3, 29, 0),
    ("10", "01", 2, 89, 0),
    ("11", "01", 2, 99, 0),
    ("14", "01", 2, 42, 0),
    ("20", "01", 2, 68, 0),
    ("21", "01", 2, 33, 0),
    ("22", "01", 2, 77, 0),
    ("23", "06", 7, 20, 0),
    ("24", "01", 3, 21, 0),
    ("04", "07", 1, 43, 1),
    ("09", "02", 1, 19, 1),
    ("15", "02", 1, 63, 1),
    ("16", "01", 2, 19, 0),
    ("18", "02", 1, 36, 1),
    ("19", "02", 1, 30, 1),
]


def run_stage1():
    print(f"=== Stage 1: cleaning {len(PARAMETERS)} patients with {NPROC} workers ===")
    with mp.Pool(NPROC) as pool:
        pool.starmap(start_process, PARAMETERS)
    print("=== Stage 1 done ===")


# ---------- stage 2: segment ----------

TEST_PATS = ["chb23", "chb24"]
VAL_PATS = ["chb21", "chb22"]

CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]
SAMPLING_RATE = 256


def sub_to_segments(folder, out_folder):
    print(f"Processing {folder}...")
    folder_path = os.path.join(clean_path, folder)
    for f in tqdm(os.listdir(folder_path)):
        record = pickle.load(open(os.path.join(folder_path, f), "rb"))

        signal = []
        for channel in CHANNELS:
            if channel in record:
                signal.append(record[channel])
            else:
                raise ValueError(f"Channel {channel} not found in record {f}")
        signal = np.array(signal)

        seizure_times = record["metadata"].get("times", [])

        for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
            segment = signal[:, i:i + 10 * SAMPLING_RATE]
            if segment.shape[1] == 10 * SAMPLING_RATE:
                label = 0
                for seizure_time in seizure_times:
                    if (i < seizure_time[0] < i + 10 * SAMPLING_RATE
                            or i < seizure_time[1] < i + 10 * SAMPLING_RATE):
                        label = 1
                        break
                pickle.dump(
                    {"X": segment, "y": label},
                    open(os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"), "wb"),
                )

        for idx, seizure_time in enumerate(seizure_times):
            for i in range(
                max(0, seizure_time[0] - SAMPLING_RATE),
                min(seizure_time[1] + SAMPLING_RATE, signal.shape[1]),
                5 * SAMPLING_RATE,
            ):
                segment = signal[:, i:i + 10 * SAMPLING_RATE]
                pickle.dump(
                    {"X": segment, "y": 1},
                    open(os.path.join(out_folder, f"{f.split('.')[0]}-s-{idx}-add-{i}.pkl"), "wb"),
                )


def run_stage2():
    folders = [f for f in os.listdir(clean_path) if f.startswith("chb")]
    print(f"=== Stage 2: segmenting {len(folders)} patient folders found in {clean_path} ===")
    out_folders = []
    for folder in folders:
        if folder in TEST_PATS:
            of = os.path.join(out_path, "test")
        elif folder in VAL_PATS:
            of = os.path.join(out_path, "val")
        else:
            of = os.path.join(out_path, "train")
        os.makedirs(of, exist_ok=True)
        out_folders.append(of)

    with mp.Pool(NPROC) as pool:
        pool.starmap(sub_to_segments, zip(folders, out_folders))
    print("=== Stage 2 done ===")


if __name__ == "__main__":
    run_stage1()
    run_stage2()
    print("\nAll done. Final segmented dataset is in:", out_path)