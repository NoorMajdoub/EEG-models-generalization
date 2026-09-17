import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import pickle
from scipy import signal

# CBraMod's patch_size is hard-baked to 200 (see cbramod.py's spectral_proj,
# tied to 200/2+1=101 FFT bins -- same hard constraint as REVE). Yang2025
# (WBCIC-SHU, Yang et al. 2025, Scientific Data, DOI 10.1038/s41597-025-04826-y)
# trials are on disk as (59, 1500) float32 @ 250Hz (6s trials, 2-class
# left/right-hand motor imagery subset only). 250Hz doesn't divide evenly
# into 200-sample patches, so we resample to 200Hz first (6s -> 1200 samples
# = 6 whole 200-sample patches), matching CBraMod's own convention of
# resampling to a rate that tiles evenly rather than padding.
#
# NOTE: this is a fresh loader reading Yang2025's real per-trial .pkl format
# directly -- NOT a reuse of shu_dataset.py, which was written by the
# original CBraMod authors against an LMDB-packed version of this same
# dataset (SHU-MI) and is incompatible with our on-disk format.
TARGET_RATE = 200
N_PATCHES = 6
N_SAMPLES = TARGET_RATE * N_PATCHES  # 1200


class CustomDataset(Dataset):
    """
    Yang2025 / WBCIC-SHU binary (left-hand vs right-hand) motor imagery.
    Raw files: 'X' (59, 1500) float32 @ 250Hz (6s trials), 'y' (0/1).
    Resamples to 200Hz (1200 samples), z-scores per-channel (real
    z-scoring, not chb_dataset.py's borrowed /100 divisor -- see project
    convention established for Clinical/KaraOne/DREAM), reshapes to
    (59, 6, 200).
    """
    def __init__(self, data_dir, mode='train'):
        super(CustomDataset, self).__init__()
        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']  # (59, 1500) @ 250Hz
        label = data_dict['y']
        data = signal.resample(data, N_SAMPLES, axis=1)
        mu = data.mean(axis=1, keepdims=True)
        sigma = data.std(axis=1, keepdims=True)
        data = (data - mu) / (sigma + 1e-8)
        n_chans = data.shape[0]
        data = data.reshape(n_chans, N_PATCHES, TARGET_RATE).astype(np.float32)
        return data, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        # Binary task -> BCEWithLogitsLoss -> Float target (matches
        # CHB-MIT/Clinical/KaraOne/DREAM convention in finetune_trainer.py).
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader