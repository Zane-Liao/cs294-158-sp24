# Some credits: https://github.com/yulun-rayn/CS294-158/blob/main/deepul/data/utils.py
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

__all__ = [
    "IntDataLoader",
]

# LongTensor Wrapper
class IntDataLoader:
    def __init__(self, data):
        self.data = torch.LongTensor(data)

    def __getitem__(self, index):
        return self.data[index]
                
    def __len__(self):
        return len(self.data)
    
