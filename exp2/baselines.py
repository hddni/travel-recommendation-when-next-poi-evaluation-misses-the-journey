"""EXP2 rule baselines. Build top-k without training.

Why this is needed
    Coverage, unseen-category and location metrics cannot be called high or low from
    their absolute values alone. Model numbers only become readable next to "the value
    a rule that learned nothing produces". Likewise for accuracy: we need the score
    obtained by merely listing the user's history to see what sequence modeling did.

4 rules  (if k items cannot be filled, fill with the global most-popular items)
    userfreq      venues in the user's prefix, **by visit count, descending**
    userrandom    venues in the user's prefix, **random order**
    random        **random** over all items
    popularity    **most popular** aggregated over all prefixes in that dataset (user-independent)

Settings follow configs/exp2.yaml; command-line arguments win when given.

Usage:
    python exp2/baselines.py -d foursquare-kr-travelset
    python exp2/baselines.py -d foursquare-kr-travelset --rules userfreq,popularity
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from data import TravelData                                    # noqa: E402
from evaluate import Scorer, evaluate, fmt                     # noqa: E402
from metrics import BeyondAccuracy                             # noqa: E402
from train import load_config                                  # noqa: E402
from runtime import mean_std, parse_seeds                      # noqa: E402

RESULT_ROOT = os.path.join(ROOT, "results", "exp2")
RULES = ("userfreq", "userrandom", "random", "popularity")
#  Rules that use no randomness. Results are the same across seeds.
DET_RULES = ("userfreq", "popularity")


class RuleScorer(Scorer):
    """Uses evaluate()'s topk delegation hook. Does not build a score matrix."""

    def __init__(self, rule, data, beyond, rng):
        self.rule, self.data, self.rng = rule, data, rng
        self.name = rule
        self.pop_rank = np.asarray([i for i in beyond.pop_rank if i > 0], np.int64)
        self.n_items = data.n_items

    def topk(self, uids, kmax):
        #  For user-independent rules, build a single row and copy it.
        if self.rule == "popularity":
            row = self._fill(list(self.pop_rank[:kmax]), kmax)
            return np.broadcast_to(row, (len(uids), kmax)).copy()
        out = np.zeros((len(uids), kmax), np.int64)
        for b, u in enumerate(uids):
            out[b] = self._fill(self._pick(u, kmax), kmax)
        return out

    def _pick(self, u, kmax):
        r = self.rule
        if r == "userfreq":
            c = Counter(self.data.seq_item[u].tolist())
            return [i for i, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))]
        if r == "userrandom":
            a = np.array(sorted(self.data.train_items.get(u, set())), np.int64)
            return list(self.rng.permutation(a)) if len(a) else []
        if r == "random":
            return list(self.rng.choice(np.arange(1, self.n_items), size=kmax,
                                        replace=False))
        raise ValueError(r)

    def _fill(self, picked, kmax):
        """Filter out duplicates / padding and fill kmax items. Fall back to global popularity if short."""
        out, got = [], set()
        for it in picked:
            it = int(it)
            if it <= 0 or it in got:
                continue
            out.append(it); got.add(it)
            if len(out) == kmax:
                return np.array(out, np.int64)
        for it in self.pop_rank:
            it = int(it)
            if it in got:
                continue
            out.append(it); got.add(it)
            if len(out) == kmax:
                break
        while len(out) < kmax:                       # unreachable in theory
            out.append(0)
        return np.array(out, np.int64)


# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dataset", required=True,
                   help="travelset dataset name (e.g. foursquare-kr-travelset)")
    p.add_argument("--rules", type=str, default=",".join(RULES))
    p.add_argument("--data_root", type=str, default=None,
                   help="Directory containing the travelset folders. If split by threshold combination,\n"
                        "pass dataset/<combo name> (default dataset/)")
    p.add_argument("--tourist", action="store_true",
                   help="Also report travel purity (tourist_purity)")
    p.add_argument("--topk", type=str, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--seeds", type=str, default=None,
                   help="e.g. 556-565. Empty means 556~565. Only random rules are averaged over seeds")
    a = p.parse_args()
    #  Uses the same config file as train.py. Training settings unused by the rules are ignored.
    for key in ("epochs", "stopping_step", "train_batch_size", "learning_rate",
                "hidden_size", "n_layers", "n_heads", "dropout", "max_len"):
        setattr(a, key, None)
    cfg = load_config(a)
    ks = tuple(cfg["topk"])
    seeds = parse_seeds(a.seeds)

    name = a.dataset
    data = TravelData(name, root=a.data_root, max_len=cfg["max_len"],
                      device="cpu")
    #  Put the folder name into the file path so results of different threshold combinations do not mix.
    thr = os.path.basename(os.path.abspath(
        a.data_root or os.path.join(ROOT, "dataset")))
    out_dir = os.path.join(RESULT_ROOT, "" if thr == "dataset" else thr)
    beyond = BeyondAccuracy(data, ks=ks, tourist=a.tourist)
    rules = [r.strip() for r in a.rules.split(",") if r.strip()]
    for r in rules:
        if r not in RULES:
            raise SystemExit(f"Unknown rule: {r} (available: {', '.join(RULES)})")
    print(f"[{name}] users {data.n_users-1:,} / items {data.n_items-1:,}", flush=True)

    #  Metrics of the ground-truth set itself (GT baseline). Once, independent of the rules.
    ref = beyond.reference(data, "test")
    print(f"  [GT] revisit {ref['test_revisit_rate']:.3f}"
          f" | unseen_cat {ref['unseen_cat_rate@gt']}"
          f" | inside_radius {ref.get('locinside@gt', float('nan')):.3f}"
          f" | compactness {ref.get('loccompact@gt', float('nan')):.2f}km", flush=True)

    def run_rule(rule, seed):
        sc = RuleScorer(rule, data, beyond, np.random.default_rng(seed))
        res = evaluate(sc, data, "test", int(cfg["eval_batch_size"]), ks,
                       beyond=beyond, progress=f"  {rule} seed {seed}")
        return {**res["accuracy"], **(res.get("beyond") or {})}

    out, out_std = {}, {}
    for rule in rules:
        #  Deterministic rules do not depend on the seed, so compute them only once.
        det = rule in DET_RULES
        mean, std = mean_std([run_rule(rule, s) for s in (seeds[:1] if det else seeds)])
        print(f"  [{rule}] {fmt({'accuracy': mean, 'beyond': mean})}", flush=True)
        out[rule] = {"deterministic": det, **mean}
        out_std[rule] = std

    payload = {
        "meta": {"experiment": "exp2", "dataset": name, "scope": data.meta["scope"],
                 "thresholds": thr,
                 "model": "rule-baselines", "rules": list(out), "seeds": seeds,
                 "topk": list(ks), "tourist": bool(a.tourist),
                 "timestamp": datetime.now().strftime("%b-%d-%Y_%H-%M-%S")},
        "config": {k: cfg[k] for k in ("eval_batch_size", "topk", "max_len")},
        "data": {k: data.meta[k] for k in
                 ("n_users", "n_items", "n_interactions", "n_train",
                  "answer_items_mean", "test_seen_rate", "daily_ratio")
                 if k in data.meta},
        "ground_truth": ref,
        "metrics": {"test_revisit_rate": ref["test_revisit_rate"], "by_rule": out},
        "metrics_std": {"by_rule": out_std},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"{name}-baselines-{payload['meta']['timestamp']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {path}", flush=True)


if __name__ == "__main__":
    main()
