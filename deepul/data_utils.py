# Some credits: https://github.com/yulun-rayn/CS294-158/blob/main/deepul/data/utils.py
import random
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

__all__ = [
    "IntDataLoader",
    "ImageDataLoader",
    "NCHWDataLoader",
    "LabeledDataset",
]

class IntDataLoader:
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.int64)

    def __getitem__(self, index):
        return self.data[index]
                
    def __len__(self):
        return len(self.data)


class ImageDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, device, to_nchw=True):
        # data: numpy array in NHWC or NCHW
        self.to_nchw = to_nchw
        self.data = torch.tensor(data, dtype=torch.int64).to(device)

    def __getitem__(self, index):
        x = self.data[index]

        # x is NHWC by default (H W C)
        if self.to_nchw:
            # NHWC → NCHW
            x = x.permute(2, 0, 1)

        return x

    def __len__(self):
        return len(self.data)    


class NCHWDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)

    def __getitem__(self, idx):
        x = self.data[idx]          # NHWC
        x = x.permute(2, 0, 1)      # C,H,W
        return x

    def __len__(self):
        return len(self.data)
    
    
class LabeledDataset:
    def __init__(self, data, labels, dropout=0.):
        self.data = torch.FloatTensor(data)
        self.labels = labels
        self.dropout = dropout

    def __getitem__(self, index):
        return (self.data[index], self.labels[index]
            if self.dropout <= 0 or random.random() > self.dropout else -1)

    def __len__(self):
        return len(self.labels)