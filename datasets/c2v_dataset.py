from typing import *
import numpy as np
import torch
import os
import pickle

from .dataset import Dataset


class C2VDataset(Dataset):
    def __init__(self, dataset_dir):
        with open(os.path.join(dataset_dir, 'vocab.pkl'), 'rb') as f:
            self.leaf_vocab, self.path_vocab = pickle.load(f)
        
        with open(os.path.join(dataset_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)

    def get_dataset(self):
        return self.dataset

    def collate(self, batch):
        contexts = []
        indices = []
        for i, context in enumerate(batch):
            contexts.append(torch.from_numpy(context))
            indices.append(
                torch.full((context.shape[0],), i, dtype=torch.long))
        contexts = torch.cat(contexts, dim=0)
        indices = torch.cat(indices)
        return contexts, indices
