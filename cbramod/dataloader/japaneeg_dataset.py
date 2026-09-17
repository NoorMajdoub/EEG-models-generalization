
import torch

from torch.utils.data import Dataset, DataLoader

import numpy as np

from utils.util import to_tensor

import os

import pickle

from scipy import signal





class CustomDataset(Dataset):

    def __init__(

            self,

            data_dir,

            mode='train',

    ):

        super(CustomDataset, self).__init__()

        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]



    def __len__(self):

        return len((self.files))



    def __getitem__(self, idx):

        file = self.files[idx]

        data_dict = pickle.load(open(file, 'rb'))

        data = data_dict['X']

        label = data_dict['y']

        data = signal.resample(data, 2000, axis=1)

        # Per-channel z-score normalization: JapanEEG's raw amplitude scale (volts)

        # is ~200,000x smaller than CHB-MIT's pre-scaled units, so CBraMod's fixed

        # /100 divisor (tuned for CHB-MIT) is not appropriate here. Z-scoring is

        # scale-invariant regardless of the source dataset's absolute units.

        mu = data.mean(axis=1, keepdims=True)

        sigma = data.std(axis=1, keepdims=True)

        data = (data - mu) / (sigma + 1e-8)

        data = data.reshape(128, 10, 200)

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

