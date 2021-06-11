from typing import *
import numpy as np
import torch
import os
import pickle

from .dataset import Dataset


class GNNDataset(Dataset):    
    def __init__(self, dataset_dir):
        with open(os.path.join(dataset_dir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        
        with open(os.path.join(dataset_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)

    def get_dataset(self):
        return self.dataset

    def collate(self, batch):
        num_prev_nodes = 0
        nodes_batch = []
        edges_batch = []
        indices = []
        for i, (nodes, edges) in enumerate(batch):
            num_nodes = nodes.shape[0]
            nodes_batch.append(torch.from_numpy(nodes))
            edges_batch.append(torch.from_numpy(edges) + num_prev_nodes)
            num_prev_nodes += num_nodes
            indices.append(torch.full((num_nodes,), i, dtype=torch.long))
        return torch.cat(nodes_batch), torch.cat(edges_batch, dim=1), torch.cat(indices)
