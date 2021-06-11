from typing import *
import numpy as np
import torch
import os
import pickle

from .dataset import Dataset


class BoFDataset(Dataset):
    def __init__(self, dataset_dir):
        with open(os.path.join(dataset_dir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        
        with open(os.path.join(dataset_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)

    def get_dataset(self):
        return self.dataset

    def collate(self, batch):
        features = []
        indices = []
        for i, (ft, _) in enumerate(batch):
            ft = torch.from_numpy(ft)
            features.append(ft)
            indices.append(torch.full_like(ft, i))
        features = torch.cat(features)
        indices = torch.cat(indices)
        return features, indices
