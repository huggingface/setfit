
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, get_idx):
        return self.data[get_idx]