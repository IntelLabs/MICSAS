'''
MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
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
