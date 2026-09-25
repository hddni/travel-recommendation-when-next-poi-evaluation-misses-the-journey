"""EXP2 runner — travelset (set of venues visited on the next trip) prediction.

A standalone pipeline that does not use RecBole's data loaders.

Why RecBole cannot be used
    (1) Different file format — inter.tsv / items.tsv / users.tsv
    (2) The split is given explicitly in a split column (not reproducible with eval_args)
    (3) The ground truth is a **set**, not a single item, so it does not fit
        FullSortEvalDataLoader's "one user = one positive" assumption

Training
    Standard sequence training that predicts the next item from a prefix (SASRec: CE, LightGCN: BPR).
    Only evaluation is set prediction. The ground-truth set is not trained on directly because
    changing the training signal would mix "the effect of the split scheme" with
    "the effect of the training objective".

Early stopping uses valid ndcg@10.

Config precedence
    command-line arguments > configs/exp2.yaml
    If an argument is not given, the yaml value is used as-is. run.sh does not pass
    an argument at all when its variable is empty.

Runs seeds 556~565 (change with --seeds) in turn and saves the mean into one JSON.
The GPU is picked automatically (runtime.py).

Usage:
    python exp2/train.py -d foursquare-us-travelset -m sasrec
    python exp2/train.py -d foursquare-us-travelset -m lightgcn --epochs 50
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, ROOT)

from data import N_PRIOR, TravelData                          # noqa: E402
from evaluate import ModelScorer, evaluate, fmt, valid_score   # noqa: E402
from metrics import BeyondAccuracy                            # noqa: E402
from models import TravelLightGCN, TravelSASRec               # noqa: E402
from runtime import mean_std, parse_seeds, use_device         # noqa: E402

CONFIG = os.path.join(ROOT, "configs", "exp2.yaml")
#  Results, checkpoints and logs are split per threshold combination, mirroring dataset/.
#      results/exp2/<combination>/   saved/exp2/<combination>/   logs/exp2/<combination>/
#  So that aggregating per-combination results into tables/figures later only needs one folder.
RESULT_ROOT = os.path.join(ROOT, "results", "exp2")
SAVED_ROOT = os.path.join(ROOT, "saved", "exp2")

#  Settings that can be overridden from the command line, and their types.
OVERRIDABLE = {
    "epochs": int, "stopping_step": int, "train_batch_size": int,
    "eval_batch_size": int, "learning_rate": float, "hidden_size": int,
    "n_layers": int, "n_heads": int, "dropout": float, "max_len": int,
}


def load_config(args):
    """Read configs/exp2.yaml and override only what was given on the command line."""
    if not os.path.exists(CONFIG):
        raise SystemExit(f"Config file not found: {CONFIG}")
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key in OVERRIDABLE:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.topk:
        cfg["topk"] = [int(x) for x in args.topk.split(",") if x.strip()]
    missing = [k for k in (*OVERRIDABLE, "topk") if cfg.get(k) is None]
    if missing:
        raise SystemExit(f"Missing settings: {', '.join(missing)}\n"
                         f"  Set them in {CONFIG} or pass them as arguments.")
    return cfg


def run_seed(a, cfg, seed, data, beyond, dev, ckpt_dir, ks):
    """Train (or reuse checkpoint) + test evaluation for one seed."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if a.model == "sasrec":
        model = TravelSASRec(data.n_items, cfg["max_len"], cfg["hidden_size"],
                             cfg["n_layers"], cfg["n_heads"], cfg["dropout"],
                             N_PRIOR).to(dev)
    else:
        model = TravelLightGCN(data.n_users, data.n_items, data.graph_edges(),
                               cfg["hidden_size"], cfg["n_layers"], dev).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    need_prefix = a.model == "sasrec"

    #  Skip training if a checkpoint exists, so we do not burn hours again
    #  every time a metric is added.
    ckpt = os.path.join(ckpt_dir, f"seed_{seed}.pt")
    t0 = time.time()
    if a.reuse and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location=dev)
        model.load_state_dict(blob["state_dict"])
        print(f"  [reused] {ckpt} (epoch {blob.get('epochs_run')}, "
              f"valid {blob.get('best_valid'):.4f})", flush=True)
        best, hist, reused = blob.get("best_valid"), blob.get("history", []), True
    else:
        print("  [training] no checkpoint - training from scratch", flush=True)
        best, best_state, bad, hist = -1.0, None, 0, []
        for ep in range(1, int(cfg["epochs"]) + 1):
            model.train()
            tot, nb = 0.0, 0
            n_batch = -(-data.n_train_inst // int(cfg["train_batch_size"]))
            bar = tqdm(data.train_batches(int(cfg["train_batch_size"]), rng,
                                          need_prefix),
                       total=n_batch, desc=f"  ep{ep:>3}", unit="batch",
                       dynamic_ncols=True, leave=False)
            for uid, it, ty, pv, ln, tgt in bar:
                opt.zero_grad()
                if a.model == "sasrec":
                    loss = F.cross_entropy(model.scores(model.encode(it, ty, pv, ln)), tgt)
                else:
                    eu, ei = model.propagate()
                    z = eu[uid]
                    neg = torch.as_tensor(rng.integers(1, model.n_items, uid.shape[0]),
                                          device=z.device)
                    loss = (F.softplus((z * ei[neg]).sum(1) - (z * ei[tgt]).sum(1)).mean()
                            + 1e-5 * model.emb.weight.pow(2).sum() / uid.shape[0])
                loss.backward(); opt.step()
                tot += loss.detach().item(); nb += 1
                bar.set_postfix(loss=f"{tot/nb:.4f}")
            bar.close()
            vr = evaluate(ModelScorer(model, data, a.model), data, "valid",
                          int(cfg["eval_batch_size"]), ks,
                          progress=f"  ep{ep:>3} valid")
            vs = valid_score(vr)
            hist.append({"epoch": ep, "loss": tot / max(nb, 1), "valid_ndcg@10": vs})
            print(f"  ep{ep:>3} loss {tot/max(nb,1):9.4f} | {fmt(vr)}", flush=True)
            if vs > best:
                best, bad = vs, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(cfg["stopping_step"]):
                    print(f"  early stopping (epoch {ep}, best {best:.4f})", flush=True)
                    break
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "best_valid": best,
                    "epochs_run": len(hist), "history": hist, "config": cfg}, ckpt)
        print(f"  [saved] {ckpt}", flush=True)
        reused = False

    test = evaluate(ModelScorer(model, data, a.model), data, "test",
                    int(cfg["eval_batch_size"]), ks, beyond=beyond,
                    progress="  test")
    print(f"  [test] {fmt(test)}", flush=True)
    run = {"seed": seed, "checkpoint": ckpt, "reused": reused,
           "epochs_run": len(hist), "best_valid_ndcg@10": best,
           "seconds": round(time.time() - t0, 1),
           "metrics": {**test["accuracy"], **(test.get("beyond") or {}),
                       "n_candidates": test["n_candidates"]},
           "history": hist}
    del model, opt
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dataset", required=True,
                   help="travelset dataset name (e.g. foursquare-us-travelset)")
    p.add_argument("-m", "--model", default="sasrec", choices=["sasrec", "lightgcn"])
    p.add_argument("--data_root", type=str, default=None,
                   help="Directory containing the travelset folders. If split by threshold combination,\n"
                        "pass dataset/<combo name> (default dataset/)")
    p.add_argument("--seeds", type=str, default=None,
                   help="e.g. 556-565 or 556,557. Empty means 556~565")
    p.add_argument("--tourist", action="store_true",
                   help="Also report travel purity (tourist_purity)")
    p.add_argument("--reuse", action="store_true",
                   help="If a checkpoint exists, skip training and only evaluate")
    p.add_argument("--topk", type=str, default=None,
                   help="Comma-separated. Empty uses the configs/exp2.yaml value")
    #  All of the following default to None. If not given, configs/exp2.yaml values are used.
    for key, cast in OVERRIDABLE.items():
        p.add_argument(f"--{key}", type=cast, default=None)
    a = p.parse_args()

    cfg = load_config(a)
    ks = tuple(cfg["topk"])
    seeds = parse_seeds(a.seeds)
    dev = use_device()
    name = a.dataset
    data = TravelData(name, root=a.data_root, max_len=cfg["max_len"],
                      device=dev)
    #  If DATA_ROOT is dataset/ itself, no combination folder is used.
    thr = os.path.basename(os.path.abspath(
        a.data_root or os.path.join(ROOT, "dataset")))
    combo = "" if thr == "dataset" else thr
    out_dir = os.path.join(RESULT_ROOT, combo)
    tag = f"{name}-{a.model}"
    print(f"[{tag}] device={dev} seeds={seeds} users={data.n_users-1:,} "
          f"items={data.n_items-1:,} train_inst={data.n_train_inst:,} "
          f"mean_answer_size={data.meta['answer_items_mean']:.2f}", flush=True)

    #  The metric calculator is built only once per dataset (loads original category / coordinate tables).
    beyond = BeyondAccuracy(data, ks=ks, tourist=a.tourist)
    #  Metrics of the ground-truth set itself. A model-independent baseline, so compute once.
    ref = beyond.reference(data, "test")
    print(f"  [GT] revisit {ref['test_revisit_rate']:.3f}"
          f" | unseen_cat {ref['unseen_cat_rate@gt']}"
          f" | inside_radius {ref.get('locinside@gt', float('nan')):.3f}"
          f" | compactness {ref.get('loccompact@gt', float('nan')):.2f}km", flush=True)

    ckpt_dir = os.path.join(SAVED_ROOT, combo, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    runs = []
    for seed in seeds:
        print(f"\n===== seed {seed} ({len(runs) + 1}/{len(seeds)}) =====", flush=True)
        runs.append(run_seed(a, cfg, seed, data, beyond, dev, ckpt_dir, ks))

    mean, std = mean_std([r["metrics"] for r in runs])
    payload = {
        "meta": {"experiment": "exp2", "dataset": name, "scope": data.meta["scope"],
                 "thresholds": thr,
                 "model": a.model, "seeds": seeds, "topk": list(ks),
                 "task": "travel_set_prediction", "tourist": bool(a.tourist),
                 "timestamp": datetime.now().strftime("%b-%d-%Y_%H-%M-%S")},
        "config": cfg,
        "data": {k: data.meta[k] for k in
                 ("n_users", "n_items", "n_interactions", "n_train",
                  "answer_items_mean", "test_seen_rate", "daily_ratio")
                 if k in data.meta},
        "ground_truth": ref,
        "metrics": {"test_revisit_rate": ref["test_revisit_rate"], **mean},
        "metrics_std": std,
        "per_seed": runs,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}-{payload['meta']['timestamp']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[mean over {len(seeds)} seeds] {fmt({'accuracy': mean, 'beyond': mean})}")
    print(f"saved: {path}", flush=True)


if __name__ == "__main__":
    main()
