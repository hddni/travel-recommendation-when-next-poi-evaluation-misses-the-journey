"""EXP1 rule baselines — build top-k without training.

Why this is needed
    On routine data, SASRec reached Recall@10 0.90. So we must check whether sequence
    modeling actually did anything, or whether just listing "places this person often
    goes" or "simply popular places" yields the same score. The most direct way to
    check that is training-free, rule-based recommendation.

4 rules (if top-k falls short, fill with the global most-popular items)
    userfreq      venues in the user's history, **by visit count, descending**
    userrandom    venues in the user's history, **random order**
    random        **random** over all items
    popularity    **most popular** in that dataset's training split (user-independent, same for everyone)

Evaluation path
    Uses RecBole's Collector + Evaluator as-is. We build only the score matrix; the
    rest (item id mapping, training-split counts, metric computation) is all handled
    by the same code as exp1/run.py, so the numbers are directly comparable.

Caveat — only meaningful with the sequential setup
    The general family (LightGCN) uses repeatable=False in eval, so already-visited
    venues are masked out of the candidates. userfreq / userrandom, which recommend
    history, are wiped out entirely under that setting and score 0. Therefore we run
    with configs/sequential.yaml and compare under the same conditions as SASRec.

Usage:
    python exp1/baselines.py -d foursquare-de-routine \
        --config_files="configs/common.yaml configs/sequential.yaml"
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if sys.path[0] != ROOT:
    sys.path.insert(0, ROOT)

from recbole.config import Config                                    # noqa: E402
from recbole.data import create_dataset, data_preparation            # noqa: E402
from recbole.evaluator import Collector, Evaluator                   # noqa: E402
from recbole.utils import init_seed, init_logger                     # noqa: E402

from runtime import mean_std, parse_seeds, pick_gpu                  # noqa: E402
from exp1.run import (CATEGORY_METRICS, DATA_DIR, METRICS,            # noqa: E402
                     TOURIST_METRIC, has_category, inject_reference_data,
                     rename_metrics, test_revisit_rate, user_history)

RESULT_ROOT = os.path.join(ROOT, "results", "exp1")
RULES = ("userfreq", "userrandom", "random", "popularity")
#  Rules that use no randomness. Results are the same across seeds.
DET_RULES = ("userfreq", "popularity")


def build_scores(rule, users, hist_csr, pop_rank, n_items, kmax, rng):
    """(B, n_items) score matrix. Only the top kmax entries need to be meaningful.

    Scores are assigned in descending order, so the argsort result is exactly our intended order.
    Global popular venues used for filling are skipped if they overlap with those already added.
    """
    scores = np.zeros((len(users), n_items), dtype=np.float32)
    #  For user-independent rules, build the row once and copy it.
    if rule == "popularity":
        row = np.zeros(n_items, dtype=np.float32)
        for rank, it in enumerate(pop_rank[:kmax]):
            row[it] = float(kmax - rank)
        return torch.from_numpy(np.broadcast_to(row, scores.shape).copy())

    for r, u in enumerate(users):
        if rule == "random":
            picked = list(rng.choice(np.arange(1, n_items), size=kmax,
                                     replace=False))
        else:
            row = hist_csr.getrow(u)
            items, cnts = row.indices, row.data
            if rule == "userfreq":
                order = items[np.argsort(-cnts, kind="stable")]
            else:                                   # userrandom
                order = rng.permutation(items)
            picked = list(order[:kmax])
        if len(picked) < kmax:
            seen = set(int(x) for x in picked)
            for it in pop_rank:
                if it not in seen:
                    picked.append(it); seen.add(it)
                    if len(picked) == kmax:
                        break
        for rank, it in enumerate(picked):
            scores[r, int(it)] = float(kmax - rank)
    return torch.from_numpy(scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dataset", required=True)
    p.add_argument("--config_files", type=str, required=True)
    p.add_argument("--data_path", type=str, default=DATA_DIR,
                   help="Directory containing the dataset folders (default dataset/)")
    p.add_argument("--seeds", type=str, default=None,
                   help="e.g. 556-565. Empty means 556~565. Only random rules are averaged over seeds")
    p.add_argument("--rules", type=str, default=",".join(RULES))
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--topk", type=str, default=None,
                   help="Comma-separated. Empty uses the config file value")
    p.add_argument("--tourist", action="store_true",
                   help="Also report travel purity (touristpurity)")
    p.add_argument("--cpu", action="store_true", help="Do not use the GPU")
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)

    data_dir = os.path.abspath(a.data_path)
    thr = os.path.basename(data_dir)
    out_dir = os.path.join(RESULT_ROOT, "" if thr == "dataset" else thr)
    want_category = has_category(a.dataset, data_dir)
    metrics = [m for m in METRICS if m not in CATEGORY_METRICS or want_category]
    if a.tourist:
        metrics.append(TOURIST_METRIC)
    cfg = {"gpu_id": "" if a.cpu else pick_gpu(), "seed": seeds[0], "metrics": metrics,
           "metric_decimal_place": 6,
           "data_path": data_dir}
    if a.cpu:
        cfg["use_gpu"] = False
    if a.eval_batch_size is not None:
        cfg["eval_batch_size"] = a.eval_batch_size
    if a.topk:
        cfg["topk"] = [int(x) for x in a.topk.split(",") if x.strip()]

    config = Config(model="SASRec", dataset=a.dataset,
                    config_file_list=a.config_files.split(), config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    inject_reference_data(config, dataset, a.dataset, want_tourist=a.tourist,
                          data_dir=data_dir)
    hist, test_pair = user_history(dataset, config)
    config["user_hist_csr"] = hist
    revisit_rate, n_test = test_revisit_rate(hist, test_pair)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    #  The global popularity ranking used for filling comes from the same source RecBole uses for metrics.
    counter = train_data._dataset.item_counter
    pop_rank = [i for i, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
                if i > 0]

    n_items = dataset.item_num
    ks = list(config["topk"])
    kmax = max(ks)
    uid_field = config["USER_ID_FIELD"]
    device = config["device"]

    n_short = int((np.diff(hist.indptr)[1:] < kmax).sum())
    print(f"[{a.dataset}] users {dataset.user_num-1:,} / items {n_items-1:,} / "
          f"users with fewer than {kmax} history venues {n_short:,} "
          f"({100*n_short/(dataset.user_num-1):.1f}%) -> filled with global popularity", flush=True)
    print(f"[revisit] previously visited venues among {n_test:,} test targets "
          f"{revisit_rate * 100:.2f}%", flush=True)

    def run_rule(rule, seed):
        rng = np.random.default_rng(seed)
        collector = Collector(config)
        collector.data_collect(train_data)
        evaluator = Evaluator(config)
        for batched in tqdm(test_data, desc=f"  {rule} seed {seed}", unit="batch",
                            dynamic_ncols=True, leave=False):
            interaction, history_index, positive_u, positive_i = batched
            users = interaction[uid_field].numpy()
            scores = build_scores(rule, users, hist, pop_rank,
                                  n_items, kmax, rng).to(device)
            scores[:, 0] = -np.inf          # 0 is padding
            #  No history_index masking. The sequential setup has repeatable=True, so it is
            #  None to begin with, and masking would make history-based rules meaningless.
            collector.eval_batch_collect(scores, interaction.to(device),
                                         positive_u.to(device), positive_i.to(device))
        res = {k: float(v) for k, v in
               evaluator.evaluate(collector.get_data_struct()).items()}
        return rename_metrics(res, ks, want_category, a.tourist)

    out, out_std = {}, {}
    for rule in [r.strip() for r in a.rules.split(",") if r.strip()]:
        if rule not in RULES:
            raise SystemExit(f"Unknown rule: {rule} (available: {', '.join(RULES)})")
        #  Deterministic rules do not depend on the seed, so compute them only once.
        det = rule in DET_RULES
        mean, std = mean_std([run_rule(rule, s) for s in (seeds[:1] if det else seeds)])
        out[rule] = {"deterministic": det, **mean}
        out_std[rule] = std
        print(f"\n=== {rule} ===")
        for k, v in mean.items():
            print(f"    {k:<34}{v}")

    payload = {
        "meta": {"experiment": "exp1", "dataset": a.dataset, "model": "rule-baselines",
                 "thresholds": thr, "data_path": data_dir,
                 "seeds": seeds, "topk": ks, "rules": list(out),
                 "tourist": bool(a.tourist),
                 "timestamp": datetime.now().strftime("%b-%d-%Y_%H-%M-%S")},
        "data": {"n_users": dataset.user_num - 1, "n_items": dataset.item_num - 1,
                 "n_interactions": dataset.inter_num,
                 "n_test_targets": n_test,
                 "orig_cat_size": int(config["orig_cat_size"] or 0) or None,
                 "users_short_of_k": n_short},
        "metrics": {"test_revisit_rate": revisit_rate, "by_rule": out},
        "metrics_std": {"by_rule": out_std},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"{a.dataset}-baselines-{payload['meta']['timestamp']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {path}", flush=True)


if __name__ == "__main__":
    main()
