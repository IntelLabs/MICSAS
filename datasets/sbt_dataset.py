from typing import *
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
import os
import pickle

from .dataset import Dataset


class SBTDataset(Dataset):    
    def __init__(self, dataset_dir):
        with open(os.path.join(dataset_dir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        
        with open(os.path.join(dataset_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)

    def get_dataset(self):
        return self.dataset

    def collate(self, batch):
        seqs = []
        indices = []
        for i, sol_seqs in enumerate(batch):
            seqs += [torch.from_numpy(seq) for seq in sol_seqs]
            indices.append(torch.full((len(sol_seqs),), i, dtype=torch.long))
        seqs = pack_sequence(seqs, enforce_sorted=False)
        indices = torch.cat(indices)
        return seqs.long(), indices
