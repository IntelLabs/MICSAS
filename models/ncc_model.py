import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class NCCModel(nn.Module):
    def __init__(self, inst2vec_emb, rnn_size, dense_size, output_size, use_i2v_emb):
        super().__init__()
        if use_i2v_emb:
            self.emb = nn.Embedding.from_pretrained(inst2vec_emb, freeze=True)
        else:
            self.emb = nn.Embedding(inst2vec_emb.size(0), inst2vec_emb.size(1))
        self.rnn = nn.LSTM(self.emb.embedding_dim, rnn_size, num_layers=2)
        self.batch_norm = nn.BatchNorm1d(rnn_size)
        self.out = nn.Sequential(
            nn.Linear(rnn_size, dense_size),
            nn.ReLU(),
            nn.Linear(dense_size, output_size)
        )

    def forward(self, seqs):
        seqs = PackedSequence(
            self.emb(seqs.data),
            seqs.batch_sizes,
            seqs.sorted_indices,
            seqs.unsorted_indices
        )

        _, (hn, _) = self.rnn(seqs)
        x = hn[-1]

        x = self.batch_norm(x)

        return self.out(x)
