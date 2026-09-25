"""travelset data loading.

Task
    Given the prefix (routine and travel mixed), predict **the set of venues to be visited on the next trip**.
    The ground truth is a set (2.8 items on average), not a single item, so RecBole's evaluation path
    cannot be used as-is.

split codes
    0 = train (prefix)   1 = valid (1 travel)   2 = test (last 3 travel)

Leakage review
    The prefix contains only the **chronologically earlier part** up to just before valid. valid/test check-ins
    do not enter the input.
    is_daily is a category-based item attribute, independent of time and user (no leakage).
    prior_visit is the cumulative revisit count within the prefix, so it looks only at the past.
    Stored global aggregate features such as is_habitual are not used.
"""

import json
import os

import numpy as np
import pandas as pd
import torch

TRAIN, VALID, TEST = 0, 1, 2
PRIOR_BINS = [1, 3, 10]          # 0 times / 1-2 / 3-9 / 10+
N_PRIOR = len(PRIOR_BINS) + 1


class TravelData:
    def __init__(self, name, root=None, max_len=200, device="cpu"):
        root = root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
        d = os.path.join(root, name)
        self.name, self.dir, self.max_len, self.device = name, d, max_len, device
        self.meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))

        inter = pd.read_csv(os.path.join(d, "inter.tsv"), sep="\t")
        items = pd.read_csv(os.path.join(d, "items.tsv"), sep="\t")
        users = pd.read_csv(os.path.join(d, "users.tsv"), sep="\t")
        self.n_users = int(users["uid"].max()) + 1      # 0 is padding
        self.n_items = int(items["iid"].max()) + 1

        self.item_is_daily = torch.zeros(self.n_items, dtype=torch.bool)
        self.item_is_daily[torch.as_tensor(items["iid"].values)] = \
            torch.as_tensor(items["is_daily"].values, dtype=torch.bool)
        self.item_coord = np.zeros((self.n_items, 2), dtype=np.float64)
        self.item_coord[items["iid"].values] = items[["latitude", "longitude"]].values

        inter = inter.sort_values(["uid", "timestamp"], kind="mergesort")
        self._build(inter)

    # ------------------------------------------------------------------
    def _build(self, inter):
        tr = inter[inter["split"] == TRAIN].copy()
        tr["prior"] = np.digitize(
            tr.groupby(["uid", "iid"]).cumcount().to_numpy(), PRIOR_BINS)
        self.seq_item, self.seq_type, self.seq_prior = {}, {}, {}
        fu, fp = [], []
        for uid, g in tr.groupby("uid", sort=True):
            self.seq_item[uid] = g["iid"].to_numpy(np.int64)
            self.seq_type[uid] = g["is_daily"].to_numpy(np.int64)
            self.seq_prior[uid] = g["prior"].to_numpy(np.int64)
            n = len(self.seq_item[uid])
            fu.append(np.full(n - 1, uid, dtype=np.int64))
            fp.append(np.arange(1, n, dtype=np.int64))
        self.train_u = np.concatenate(fu)
        self.train_p = np.concatenate(fp)
        self.n_train_inst = len(self.train_u)
        self.train_items = {u: set(v.tolist()) for u, v in self.seq_item.items()}

        #  Evaluation is per user. valid has 1 ground-truth item, test has a ground-truth set.
        self.eval_users = np.array(sorted(self.seq_item.keys()), dtype=np.int64)
        self.answers = {}
        for phase, code in (("valid", VALID), ("test", TEST)):
            g = inter[inter["split"] == code].groupby("uid")["iid"]
            self.answers[phase] = {u: np.unique(v.to_numpy()) for u, v in g}

    # ------------------------------------------------------------------
    def prefix_batch(self, uids):
        """Right-padded prefix. Uses the most recent max_len of the whole train sequence."""
        B, L = len(uids), self.max_len
        it = np.zeros((B, L), np.int64)
        ty = np.zeros((B, L), np.int64)
        pv = np.zeros((B, L), np.int64)
        ln = np.ones(B, np.int64)
        for b, u in enumerate(uids):
            s = self.seq_item[u]
            n = min(len(s), L)
            it[b, :n] = s[-n:]
            ty[b, :n] = self.seq_type[u][-n:]
            pv[b, :n] = self.seq_prior[u][-n:]
            ln[b] = n
        t = torch.as_tensor
        return (t(it).to(self.device), t(ty).to(self.device),
                t(pv).to(self.device), t(ln).to(self.device))

    def _window(self, uids, cuts):
        B, L = len(uids), self.max_len
        it = np.zeros((B, L), np.int64); ty = np.zeros((B, L), np.int64)
        pv = np.zeros((B, L), np.int64); ln = np.ones(B, np.int64)
        for b, (u, c) in enumerate(zip(uids, cuts)):
            s = max(0, c - L); n = c - s
            if n <= 0:
                continue
            it[b, :n] = self.seq_item[u][s:c]
            ty[b, :n] = self.seq_type[u][s:c]
            pv[b, :n] = self.seq_prior[u][s:c]
            ln[b] = n
        t = torch.as_tensor
        return (t(it).to(self.device), t(ty).to(self.device),
                t(pv).to(self.device), t(ln).to(self.device))

    def train_batches(self, batch_size, rng, need_prefix=True):
        order = rng.permutation(self.n_train_inst)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            u, p = self.train_u[idx], self.train_p[idx]
            tgt = np.array([self.seq_item[a][b] for a, b in zip(u, p)], np.int64)
            if need_prefix:
                it, ty, pv, ln = self._window(u, p)
            else:
                it = ty = pv = ln = None
            yield (torch.as_tensor(u).to(self.device), it, ty, pv, ln,
                   torch.as_tensor(tgt).to(self.device))

    def graph_edges(self):
        """Edges for LightGCN. train portion only."""
        us, its = [], []
        for u, s in self.seq_item.items():
            us.append(np.full(len(s), u, np.int64)); its.append(s)
        return np.concatenate(us), np.concatenate(its)
