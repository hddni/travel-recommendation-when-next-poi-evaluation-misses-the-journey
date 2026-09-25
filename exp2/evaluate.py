"""EXP2 (travelset) accuracy evaluation. The ground truth is a **set**.

Accuracy   T = ground-truth set, R_k = top-k recommendations, H_u = set of venues in the user's prefix
    recall@k = |T ∩ R_k| / |T|
    ndcg@k   = DCG / IDCG,  IDCG is the case where min(|T|, k) items are all hits

Contribution decomposition — splits hits into "places already visited" / "places visited for the first time".
    recall_revisit@k + recall_novel@k = recall@k
    ndcg_revisit@k   + ndcg_novel@k   = ndcg@k

    Looking only at the total score does not reveal that revisits are propping up the score.
    The decomposition looks at the same hits twice: without position weighting (recall) and with it (ndcg).
    The ndcg side captures that revisits at top ranks take more of the score.

Aggregation is macro — compute the metric per user, then average across users.

Revisit handling
    Candidates are not restricted to new venues. Seen or unseen, they are recommended as-is.
    Restricting them would make a substantial part of the ground truth structurally unreachable,
    and the experiment measuring revisit contribution and RevisitAmp would not hold in the first place.
"""

import numpy as np
import torch
from tqdm import tqdm

TOPK = (5, 10, 20)
ACC = ("recall", "ndcg", "recall_revisit", "recall_novel",
       "ndcg_revisit", "ndcg_novel")
NEG = -1e30


# ──────────────────────────────────────────────────────────────────────
#  Scorer — the junction that makes models and rule baselines go through the same evaluation path
# ──────────────────────────────────────────────────────────────────────
class Scorer:
    """uids -> (B, n_items) scores. Larger value ranks higher."""

    name = "scorer"

    def scores(self, uids):
        raise NotImplementedError


class ModelScorer(Scorer):
    def __init__(self, model, data, kind="sasrec"):
        self.model, self.data, self.kind = model, data, kind
        self.name = kind

    def scores(self, uids):
        if self.kind == "sasrec":
            it, ty, pv, ln = self.data.prefix_batch(uids)
            h = self.model.encode(it, ty, pv, ln)
            if not torch.isfinite(h).all():
                raise RuntimeError("Encoder output contains NaN/Inf.")
            return self.model.scores(h)
        ut = torch.as_tensor(np.asarray(uids)).to(self.data.device)
        return self.model.scores(ut)


def topk_indices(scorer, data, uids, kmax):
    #  Rule baselines have no reason to build a (B, n_items) matrix. In the ALL scope,
    #  batch 256 x 480k items = 500MB would be wasted, so they provide top-k directly.
    if hasattr(scorer, "topk"):
        return scorer.topk(uids, kmax)
    sc = scorer.scores(uids).clone()
    sc[:, 0] = NEG                                   # 0 is padding
    return sc.topk(kmax, dim=1).indices.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────
def _agg(recs):
    """macro average. One row per user."""
    if not recs:
        return {"n_users": 0}
    out = {}
    for k in recs[0]:
        v = np.array([r[k] for r in recs], dtype=float)
        m = np.isfinite(v)
        out[k] = float(v[m].mean()) if m.any() else float("nan")
    out["n_users"] = len(recs)
    return out


#  Position discount for DCG. Built in advance and sliced for use.
_DISCOUNT = 1.0 / np.log2(np.arange(2, max(TOPK) + 2))


def _acc(row, gs, hist, n_gt, k):
    """One user's recall / ndcg and their revisit / novel decomposition."""
    tk = [int(x) for x in row[:k]]
    hit = np.fromiter((x in gs for x in tk), bool, k)
    rev = np.fromiter((x in hist for x in tk), bool, k)

    disc = _DISCOUNT[:k]
    cap = min(n_gt, k)
    idcg = float(_DISCOUNT[:cap].sum())

    n_hit = int(hit.sum())
    dcg = float((hit * disc).sum())
    dcg_rev = float((hit & rev) @ disc)
    return {
        f"recall@{k}": n_hit / n_gt,
        f"recall_revisit@{k}": int((hit & rev).sum()) / n_gt,
        f"recall_novel@{k}": int((hit & ~rev).sum()) / n_gt,
        f"ndcg@{k}": dcg / idcg if idcg > 0 else 0.0,
        f"ndcg_revisit@{k}": dcg_rev / idcg if idcg > 0 else 0.0,
        f"ndcg_novel@{k}": (dcg - dcg_rev) / idcg if idcg > 0 else 0.0,
    }


@torch.no_grad()
def evaluate(scorer, data, phase="test", batch_size=256, ks=TOPK, beyond=None,
             progress=""):
    """beyond: if a BeyondAccuracy instance is given, metrics beyond accuracy are reported too.

    Not given for the valid evaluation used for early stopping (slow if run every epoch). Test only.
    """
    if hasattr(scorer, "model"):
        scorer.model.eval()
    ans = data.answers[phase]
    users = [u for u in data.eval_users if u in ans and len(ans[u])]
    kmax = max(ks)
    all_r = []
    tops, top_users = [], []

    starts = range(0, len(users), batch_size)
    if progress:
        starts = tqdm(starts, desc=progress, unit="batch",
                      dynamic_ncols=True, leave=False)
    for s in starts:
        ub = users[s:s + batch_size]
        top = topk_indices(scorer, data, ub, kmax)
        if beyond is not None:
            tops.append(top); top_users.extend(ub)

        for row, u in zip(top, ub):
            gs = set(ans[u].tolist())
            hist = data.train_items.get(u, set())
            rec = {}
            for k in ks:
                rec.update(_acc(row, gs, hist, len(gs), k))
            all_r.append(rec)

    res = {"accuracy": _agg(all_r), "n_candidates": int(data.n_items - 1)}
    if beyond is not None and tops:
        top = np.concatenate(tops, axis=0)
        res["beyond"] = beyond.rec_metrics(top, top_users)
        res["beyond"].update(beyond.revisit_amp(top, top_users))
    return res


def valid_score(res):
    return res["accuracy"]["ndcg@10"]


def fmt(res):
    a = res["accuracy"]
    t = (f"R@10 {a['recall@10']:.4f} (rev {a['recall_revisit@10']:.4f}"
         f" + new {a['recall_novel@10']:.4f}) "
         f"N@10 {a['ndcg@10']:.4f} (rev {a['ndcg_revisit@10']:.4f}"
         f" + new {a['ndcg_novel@10']:.4f}) (n={a['n_users']})")
    b = res.get("beyond") or {}
    if "revisitamp@10" in b:
        t += f" | amp {b['revisitamp@10']:.2f}x"
    return t
