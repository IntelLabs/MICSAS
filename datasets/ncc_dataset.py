from typing import *
import numpy as np
from torch.nn.utils.rnn import pack_sequence
import torch
import os
import pickle

from .dataset import Dataset


class NCCDataset(Dataset):
    def __init__(self, dataset_dir):
        with open(os.path.join(dataset_dir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        
        with open(os.path.join(dataset_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)

    def get_dataset(self):
        return self.dataset

    def collate(self, batch):
        seqs = [torch.from_numpy(x) for x in batch]
        seqs = pack_sequence(seqs, enforce_sorted=False)
        return seqs.long(),
