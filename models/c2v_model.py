import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax


class C2VModel(nn.Module):
    def __init__(self, leaf_emb_size, leaf_vocab_size, path_emb_size, path_vocab_size, code_vec_size, output_size):
        super().__init__()
        self.leaf_emb = nn.Embedding(leaf_vocab_size, leaf_emb_size)
        self.path_emb = nn.Embedding(path_vocab_size, path_emb_size)
        self.emb_dropout = nn.Dropout(0.25)

        self.fc = nn.Sequential(
            nn.Linear(leaf_emb_size * 2 + path_emb_size,
                      code_vec_size, bias=False),
            nn.Tanh()
        )

        self.a = nn.Parameter(torch.empty(code_vec_size, dtype=torch.float))
        nn.init.uniform_(self.a)

        self.out = nn.Linear(code_vec_size, output_size)

    def forward(self, contexts, indices):
        context_emb = torch.cat(
            (self.leaf_emb(contexts[:, 0]), self.path_emb(contexts[:, 1]), self.leaf_emb(contexts[:, 2])),
            dim=1
        )
        context_emb = self.emb_dropout(context_emb)
        context_emb = self.fc(context_emb)

        attn_score = torch.matmul(context_emb, self.a)
        attn_weight = scatter_softmax(attn_score, indices, dim=0)
        weighted_context = context_emb * attn_weight.unsqueeze(1)
        v = scatter_add(weighted_context, indices, dim=0)

        return self.out(v)
