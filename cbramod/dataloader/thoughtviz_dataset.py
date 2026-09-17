import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import pickle
from scipy import signal

# CBraMod's patch_size is hard-baked to 200 (same constraint as REVE), and
# every trial must reshape into a WHOLE number of 200-sample patches.
# ThoughtViz's real recorded signal is only 0.25s (32 samples @ 128Hz) --
# far short of even one full patch. Same disclosed treatment as the REVE
# build: resample the real 0.25s to 200Hz (32 -> 50 samples), z-score using
# only the real samples' own statistics, then zero-pad the remaining 150
# samples to fill out exactly one 200-sample patch. Padding (not
# repeating/tiling) was chosen so we never place real recorded values into
# time slots where they never occurred.
N_REAL = 50  # 0.25s @ 200Hz
N_PATCH = 200


class CustomDataset(Dataset):
    """
    ThoughtViz 10-class object-category classification. Raw files: 'X'
    (14, 32) float32 @ 128Hz (0.25s window -- known unfixable train/test
    overlap-window leakage per the ThoughtViz paper, same caveat as the
    REVE/BIOT builds), 'y' (int, 0-9). Reshapes to (14, 1, 200): a single
    patch, only 25% of which is real signal (see module docstring).
    """
    def __init__(self, data_dir, mode='train'):
        super(CustomDataset, self).__init__()
        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']  # (14, 32), 0.25s @ 128Hz
        label = data_dict['y']
        data = signal.resample(data, N_REAL, axis=1)
        mu = data.mean(axis=1, keepdims=True)
        sigma = data.std(axis=1, keepdims=True)
        data = (data - mu) / (sigma + 1e-8)
        n_chans = data.shape[0]
        padded = np.zeros((n_chans, N_PATCH), dtype=np.float32)
        padded[:, :N_REAL] = data
        padded = padded.reshape(n_chans, 1, N_PATCH)
        return padded, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        # CrossEntropyLoss (used for this 10-class multiclass dataset) requires
        # a Long target, unlike BCEWithLogitsLoss (used by the binary datasets)
        # which wants Float -- to_tensor() casts to float by default, so cast
        # explicitly here instead.
        return to_tensor(x_data), torch.LongTensor(y_label)


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