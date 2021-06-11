import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class BagOfFeaturesModel(nn.Module):
    def __init__(self, feature_emb_size, feature_vocab_size, output_size):
        super().__init__()
        self.feature_emb = nn.Sequential(
            nn.Embedding(feature_vocab_size, feature_emb_size),
            nn.Dropout(0.5)
        )
        self.out = nn.Sequential(
            nn.Linear(feature_emb_size, output_size)
        )

    def forward(self, features, indices):
        feature_emb = self.feature_emb(features)
        return self.out(scatter_mean(feature_emb, indices, dim=0))
