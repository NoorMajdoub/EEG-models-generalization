import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import pickle
from scipy import signal


class CustomDataset(Dataset):
    """
    KaraOne imagined-speech binary classification. Raw files: 'X'
    (60, ~4960-4970) float32 @ 1000Hz (trial length varies slightly,
    ~5s), 'y' (int, 0/1). 60 real 10-10 system channels.

    Resamples to a fixed 5s target (1000 samples @ 200Hz = 5 one-second
    patches) regardless of the exact native length, then reshapes to
    (60, 5, 200) to satisfy CBraMod's fixed patch_size=200.

    Normalization: real per-channel z-scoring, not a borrowed magic-number
    divisor.
    """
    def __init__(self, data_dir, mode='train'):
        super(CustomDataset, self).__init__()
        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']  # (60, ~4960-4970), ~5s @ 1000Hz
        label = data_dict['y']
        data = signal.resample(data, 1000, axis=1)
        mu = data.mean(axis=1, keepdims=True)
        sigma = data.std(axis=1, keepdims=True)
        data = (data - mu) / (sigma + 1e-8)
        data = data.reshape(60, 5, 200)
        return data, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
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