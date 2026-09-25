"""travelset models. prefix -> scores over all items.

Neither a type head nor candidate masking is used. That the ground truth is a travel check-in
is a consequence of how the data was constructed, not information available at inference time.
Cutting candidates by the ground-truth type would be leakage of knowing the answer in advance.

The prefix's is_daily / prior_visit describe the nature of past visits, so they are safe as inputs.
"""

import numpy as np
import torch
import torch.nn as nn

NEG = -1e9      # excludes padding. Using -inf makes log_softmax backprop produce NaN.


class TravelSASRec(nn.Module):
    def __init__(self, n_items, max_len, hidden=64, n_layers=2, n_heads=2,
                 dropout=0.2, n_prior=4):
        super().__init__()
        self.n_items, self.max_len, self.hidden = n_items, max_len, hidden
        self.item_emb = nn.Embedding(n_items, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.type_emb = nn.Embedding(2, hidden)
        self.prior_emb = nn.Embedding(n_prior, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, n_heads, dim_feedforward=hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, items, types, priors, lengths):
        B, L = items.shape
        pos = torch.arange(L, device=items.device).unsqueeze(0).expand(B, L)
        x = (self.item_emb(items) + self.pos_emb(pos)
             + self.type_emb(types) + self.prior_emb(priors))
        x = self.drop(self.ln(x))
        #  key_padding_mask is not used. Combined with the causal mask, the leading positions
        #  become all -inf and softmax produces NaN. Since padding is on the right, the causal
        #  mask alone blocks the trailing padding.
        causal = torch.triu(torch.ones(L, L, device=items.device, dtype=torch.bool), 1)
        h = self.encoder(x, mask=causal)
        return h[torch.arange(B, device=items.device), (lengths - 1).clamp(min=0)]

    def scores(self, h):
        out = h @ self.item_emb.weight.t()
        out[:, 0] = NEG
        return out


class TravelLightGCN(nn.Module):
    def __init__(self, n_users, n_items, edges, hidden=64, n_layers=2, device="cpu"):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.emb = nn.Embedding(n_users + n_items, hidden)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.adj = self._norm_adj(*edges, device)

    def _norm_adj(self, u, i, device):
        n = self.n_users + self.n_items
        r = np.concatenate([u, i + self.n_users])
        c = np.concatenate([i + self.n_users, u])
        deg = np.bincount(r, minlength=n).astype(np.float64)
        deg[deg == 0] = 1.0
        val = 1.0 / np.sqrt(deg[r] * deg[c])
        idx = torch.as_tensor(np.stack([r, c]), dtype=torch.long)
        return torch.sparse_coo_tensor(
            idx, torch.as_tensor(val, dtype=torch.float32), (n, n)).coalesce().to(device)

    def propagate(self):
        x = self.emb.weight
        out = x
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            out = out + x
        out = out / (self.n_layers + 1)
        return out[:self.n_users], out[self.n_users:]

    def scores(self, uids):
        eu, ei = self.propagate()
        out = eu[uids] @ ei.t()
        out[:, 0] = NEG
        return out
