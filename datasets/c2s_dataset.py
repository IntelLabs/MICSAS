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
from os import replace
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import os
import pickle
import numpy as np
import itertools

from .dataset import Dataset


class C2SDataset(Dataset):
    def __init__(self, dataset_dir, max_contexts=200):
        with open(os.path.join(dataset_dir, 'vocab.pkl'), 'rb') as f:
            self.subtoken_vocab, self.node_vocab = pickle.load(f)

        with open(os.path.join(dataset_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)

        self.max_contexts = max_contexts

    def get_dataset(self):
        return self.dataset

    def collate(self, batch):
        batch_ll_subtokens = []
        batch_ll_indices = []
        batch_rl_subtokens = []
        batch_rl_indices = []
        batch_paths = []
        batch_path_lengths = []
        batch_indices = []
        j = 0  # context id

        for i, (subtokens, subtoken_ranges, padded_paths, path_lengths) in enumerate(batch):
            if len(path_lengths) > self.max_contexts:
                sample_ids = np.random.choice(
                    len(path_lengths), self.max_contexts, replace=False)
                subtoken_ranges = subtoken_ranges[sample_ids]
                padded_paths = padded_paths[sample_ids]
                path_lengths = path_lengths[sample_ids]

            batch_ll_subtokens.append(
                torch.from_numpy(
                    subtokens[
                        np.r_[tuple(map(lambda t: slice(t[0], t[1]), subtoken_ranges))]
                    ]
                )
            )
            batch_rl_subtokens.append(
                torch.from_numpy(
                    subtokens[
                        np.r_[tuple(map(lambda t: slice(t[1], t[2]), subtoken_ranges))]
                    ]
                )
            )

            batch_ll_indices.append(torch.tensor(
                list(
                    itertools.chain(*[[ti] * tn for ti, tn in enumerate(subtoken_ranges[:, 1] - subtoken_ranges[:, 0])])
                ),
                dtype=torch.long
            ) + j)
            batch_rl_indices.append(torch.tensor(
                list(
                    itertools.chain(*[[ti] * tn for ti, tn in enumerate(subtoken_ranges[:, 2] - subtoken_ranges[:, 1])])
                ),
                dtype=torch.long
            ) + j)
            j += len(subtoken_ranges)

            batch_paths.append(torch.from_numpy(padded_paths))
            batch_path_lengths.append(torch.from_numpy(path_lengths))
            batch_indices.append(
                torch.full((len(path_lengths),), i, dtype=torch.long))

        batch_ll_subtokens = torch.cat(batch_ll_subtokens)
        batch_ll_indices = torch.cat(batch_ll_indices)
        batch_rl_subtokens = torch.cat(batch_rl_subtokens)
        batch_rl_indices = torch.cat(batch_rl_indices)
        batch_path_lengths = torch.cat(batch_path_lengths)
        batch_paths = torch.cat(batch_paths)
        batch_paths = pack_padded_sequence(batch_paths, batch_path_lengths,
                                           batch_first=True, enforce_sorted=False)
        batch_indices = torch.cat(batch_indices)

        return batch_ll_subtokens, batch_ll_indices, batch_rl_subtokens, batch_rl_indices, batch_paths, batch_indices
