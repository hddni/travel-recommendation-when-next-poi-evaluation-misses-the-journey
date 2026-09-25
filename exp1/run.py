"""EXP1 runner — performance comparison across travel / routine / raw datasets.

One training run + one inference pass produce the groups below.

    Accuracy                recall@k, ndcg@k
    Revisit composition     test_revisit_rate
                            Share of test targets that are venues the user has
                            **already visited**. A property of the dataset, not
                            the model. Reported to check whether accuracy gaps
                            between arms come from differences in this value
                            rather than from model ability.
    Revisit amplification   revisitamp@k
                            Share of already-visited venues among recommendations
                            / the user's own original revisit rate.
                            1 means the tendency is preserved; above 1 means the
                            model pushed the habit further.
    Coverage                category_coverage@k   (null for corpora without categories)
    Exploration             unseen_cat_rate@k
                            Share of the k recommendations that are venue kinds
                            the user has never visited. Regardless of hits, it
                            only checks whether the model goes outside the
                            user's existing categories. (null for corpora
                            without categories)
    Popularity bias         popularity_internal@k       relative to that dataset's training split

Uses the same definitions as exp2, because the values of the two experiments must
be readable side by side.

The three arms were cut with different filters, so measuring only against internal
popularity / catalog makes cross-arm comparison invalid. Hence we also report metrics
that use the original corpus as a common yardstick. The internal basis is kept as
well because when the two yardsticks disagree, that itself is information.

With --tourist, tourist purity (touristpurity) is additionally reported. It needs
the label table, so it is only meaningful on the arm closest to the original (raw).

Runs seeds 556~565 (change with --seeds) in turn and saves the mean into one JSON.
Reuses checkpoints if present (--reuse), otherwise trains. The GPU is picked automatically.

Usage:
    python exp1/run.py -m SASRec -d foursquare-kr-routine \
        --config_files="configs/common.yaml configs/sequential.yaml" --reuse
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datetime import datetime

from scipy.sparse import coo_matrix

#  This repo's recbole/ must be picked up before the pip-installed one, because the
#  evaluation metrics (RevisitAmp / CategoryCoverage / TouristPurity, etc.) and the
#  Collector's rec.users collection exist only here.
#  The script lives in a subdirectory so sys.path[0] is exp1/; insert ROOT manually.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if sys.path[0] != ROOT:
    sys.path.insert(0, ROOT)

from recbole.config import Config                                    # noqa: E402
from recbole.data import create_dataset, data_preparation            # noqa: E402
from recbole.utils import (get_model, get_trainer, init_seed,        # noqa: E402
                           init_logger)
from runtime import mean_std, parse_seeds, pick_gpu                  # noqa: E402

#  Metrics requested from RecBole. Names are not used as-is; they are renamed by
#  RENAME below before writing to JSON (so keys match the requested metric list 1:1).
METRICS = ["Recall", "NDCG", "RevisitAmp", "CategoryCoverage",
           "UnseenCategoryRate", "AveragePopularity"]
#  Metrics dropped for corpora without categories (gowalla).
CATEGORY_METRICS = ("CategoryCoverage", "UnseenCategoryRate")
TOURIST_METRIC = "TouristPurity"

#  Datasets can be relocated with --data_path (a subfolder per threshold combination).
#  The 3 reference tables below do not depend on thresholds, so they always live at the dataset/ root.
DATA_DIR = os.path.join(ROOT, "dataset")
CAT_DIR = os.path.join(DATA_DIR, "venue_category_stats")
TOURIST_DIR = os.path.join(DATA_DIR, "venue_tourist_label")
#  Results, checkpoints and logs are split per threshold combination, mirroring dataset/.
#      results/exp1/<combination>/   saved/exp1/<combination>/   logs/exp1/<combination>/
#  So that aggregating per-combination results into tables/figures later only needs one folder.
RESULT_ROOT = os.path.join(ROOT, "results", "exp1")
SAVED_ROOT = os.path.join(ROOT, "saved", "exp1")


def resolve_scope(dataset_name):
    """Dataset name -> original corpus scope key (venue_category_stats / venue_tourist_label).

    foursquare is <corpus>-<scope>-<arm>, so the second piece is the scope.
    gowalla has no country information, so the whole corpus is a single scope.
    """
    parts = dataset_name.split("-")
    if parts[0] == "gowalla":
        return "GOWALLA"
    return parts[1].upper()


def has_category(dataset_name, data_dir=DATA_DIR):
    """Whether .item has actual category values. If not, category_coverage is reported as null.

    gowalla has no categories, so they are all empty strings. Measuring as-is gives 0,
    which is misread as "diversity is 0".
    """
    path = os.path.join(data_dir, dataset_name, f"{dataset_name}.item")
    if not os.path.exists(path):
        return True
    col = pd.read_csv(path, sep="\t", dtype=str, usecols=[1]).iloc[:, 0]
    return bool((col.fillna("").astype(str).str.strip() != "").any())


def user_history(dataset, config):
    """(n_user x n_item) CSR of which venues each user visited how many times before test time.

    Must not use what the model actually saw. A sequential model's prefix is cut
    at MAX_ITEM_LIST_LENGTH (default 50), so the same user's history has different
    sizes in SASRec and LightGCN, and then the very basis of the question "has the
    user already been there?" differs per model. Here it is built directly from the split rule.

    Evaluation is LS (leave-one-out) + TO (temporal order), so each user's last row is the test target.
    Dropping just that one row leaves "everything before test" (= train + valid).
    """
    #  Before data_preparation, inter_feat is still a pandas DataFrame;
    #  afterwards it becomes an Interaction (tensors). Accept both.
    inter = dataset.inter_feat
    col = lambda f: np.asarray(inter[f])
    uid = col(config["USER_ID_FIELD"])
    iid = col(config["ITEM_ID_FIELD"])
    tf = config["TIME_FIELD"]
    if tf in inter:
        order = np.lexsort((col(tf), uid))
        uid, iid = uid[order], iid[order]
    is_last = np.ones(len(uid), dtype=bool)
    is_last[:-1] = uid[:-1] != uid[1:]      # last row of the user block = test target
    test_u, test_i = uid[is_last], iid[is_last]
    uid, iid = uid[~is_last], iid[~is_last]
    keep = iid > 0                          # 0 is padding
    uid, iid = uid[keep], iid[keep]
    hist = coo_matrix((np.ones(len(uid), dtype=np.float32), (uid, iid)),
                      shape=(dataset.user_num, dataset.item_num)).tocsr()
    return hist, (test_u, test_i)


def test_revisit_rate(hist, test_pair):
    """Share of test targets that are revisit venues for that user. A dataset property independent of the model."""
    u, i = test_pair
    keep = i > 0
    u, i = u[keep], i[keep]
    if not len(u):
        return float("nan"), 0
    seen = np.asarray(hist[u, i]).ravel() > 0
    return float(seen.mean()), int(len(u))


def inject_reference_data(config, dataset, dataset_name, want_tourist=False,
                          data_dir=DATA_DIR):
    """Load into config the reference data used by metrics measured against the original corpus.

    CategoryCoverage / TouristPurity use the **original corpus before filtering** as
    the yardstick, not this dataset's internals. Filter strength differs per arm, so
    measuring against each arm's own catalog makes cross-arm comparison invalid.
    Built once here so that exp1/run.py and exp1/baselines.py use the same yardstick.
    """
    scope = resolve_scope(dataset_name)
    tokens = dataset.field2id_token[config["ITEM_ID_FIELD"]]
    n_item = len(tokens) - 1

    #  ── item -> category ──────────────────────────────────────────────
    #  Touching load_col changes the dataset filtering path, so read .item directly,
    #  build an array in internal item id order, and inject it.
    item_path = os.path.join(data_dir, dataset_name, f"{dataset_name}.item")
    item_df = pd.read_csv(item_path, sep="\t", dtype=str)
    icol, ccol = item_df.columns[0], item_df.columns[1]   # item_id, venue_category_id
    cat_of = dict(zip(item_df[icol], item_df[ccol]))
    codes = {c: i for i, c in enumerate(sorted(set(cat_of.values())))}
    item_cat = np.array([-1] + [codes.get(cat_of.get(t), -1) for t in tokens[1:]],
                        dtype=np.int64)
    config["item_cat_array"] = item_cat

    if any(m in (config["metrics"] or []) for m in CATEGORY_METRICS):
        cat_path = os.path.join(CAT_DIR, f"{scope}.csv")
        if not os.path.exists(cat_path):
            raise SystemExit(f"Original category statistics not found: {cat_path}")
        config["orig_cat_size"] = len(pd.read_csv(cat_path, encoding="utf-8-sig"))
        no_cat = int((item_cat[1:] < 0).sum())
        if no_cat:
            print(f"[warning] items with no category found: {no_cat:,}/{n_item:,}",
                  flush=True)
        print(f"[denominator] original categories "
              f"{config['orig_cat_size']:,} | dataset venues {n_item:,} / categories "
              f"{len(np.unique(item_cat[item_cat >= 0])):,}", flush=True)
    else:
        config["orig_cat_size"] = 0
        print(f"[denominator] dataset venues "
              f"{n_item:,}  (no categories - category_coverage is null)", flush=True)

    if want_tourist:
        inject_tourist_labels(config, dataset, dataset_name, scope, tokens, item_df)


def inject_tourist_labels(config, dataset, dataset_name, scope, tokens, item_df):
    """3-tier labels for tourist purity. Population P = original check-ins - valid/test."""
    vpath = os.path.join(TOURIST_DIR, f"{scope}.venue.tsv")
    ppath = os.path.join(TOURIST_DIR, f"{scope}.pair.tsv")
    if not os.path.exists(vpath):
        raise SystemExit(f"Travel label table not found: {vpath}\n"
                         f"  python dataset/build_tourist_label.py --scopes {scope}")

    #  Tier 2: user-level majority-vote ratio. Unobserved is left as NaN to distinguish it from tier 3.
    vdf = pd.read_csv(vpath, sep="\t", dtype={"venue_id": str})
    frac = dict(zip(vdf["venue_id"], vdf["frac_tourist"]))
    config["item_tourist_frac_array"] = np.array(
        [np.nan] + [frac.get(t, np.nan) for t in tokens[1:]], dtype=np.float64)

    #  Category-level routine decision. Use .item's is_daily, else fall back to the label table.
    icol = item_df.columns[0]
    if "is_daily" in item_df.columns:
        daily_of = dict(zip(item_df[icol], item_df["is_daily"].astype(str)))
        cat_daily = np.array(
            [False] + [str(daily_of.get(t, "0")).strip() in ("1", "True", "true")
                       for t in tokens[1:]], dtype=bool)
    else:
        chk = dict(zip(vdf["venue_id"], vdf["is_cat_daily"]))
        cat_daily = np.array([False] + [bool(int(chk.get(t, 0))) for t in tokens[1:]],
                             dtype=bool)
    config["item_cat_daily_array"] = cat_daily

    #  Tier 1 exceptions: (u, v) pairs whose category is travel but which are routine for that user.
    uid_tokens = dataset.field2id_token[config["USER_ID_FIELD"]]
    u2i = {t: i for i, t in enumerate(uid_tokens)}
    v2i = {t: i for i, t in enumerate(tokens)}
    pair = set()
    if os.path.exists(ppath):
        pdf = pd.read_csv(ppath, sep="\t",
                          dtype={"user_token": str, "venue_id": str})
        for a, b in zip(pdf["user_token"], pdf["venue_id"]):
            if a in u2i and b in v2i:
                pair.add((u2i[a], v2i[b]))
    config["pair_daily_set"] = pair
    n_lab = int(np.isfinite(config["item_tourist_frac_array"][1:]).sum())
    print(f"[travel labels] venues labeled {n_lab:,}/{len(tokens)-1:,} | routine categories "
          f"{int(cat_daily[1:].sum()):,} | per-user exception pairs {len(pair):,}", flush=True)


#  Original keys emitted by RecBole -> names written to JSON.
#  Renamed once here so keys match the requested metric list 1:1.
def rename_metrics(result, ks, want_category, want_tourist):
    out = {}
    for k in ks:
        out[f"recall@{k}"] = result.get(f"recall@{k}")
        out[f"ndcg@{k}"] = result.get(f"ndcg@{k}")
    for k in ks:
        out[f"revisitamp@{k}"] = result.get(f"revisitamp@{k}")
        out[f"revisit_rec@{k}"] = result.get(f"revisitrec@{k}")
        out[f"revisit_hist@{k}"] = result.get(f"revisithist@{k}")
    #  For corpora without categories (gowalla), do not fabricate values; leave them null.
    for k in ks:
        out[f"category_coverage@{k}"] = (
            result.get(f"categorycoverage@{k}") if want_category else None)
        out[f"unseen_cat_rate@{k}"] = (
            result.get(f"unseencategoryrate@{k}") if want_category else None)
    for k in ks:
        out[f"popularity_internal@{k}"] = result.get(f"averagepopularity@{k}")
    if want_tourist:
        for k in ks:
            out[f"tourist_purity@{k}"] = result.get(f"touristpurity@{k}")
            out[f"tourist_purity_soft@{k}"] = result.get(f"touristpuritysoft@{k}")
            out[f"tourist_undecided@{k}"] = result.get(f"touristundef@{k}")
    return out


def run_seed(a, seed, gpu_id, data_dir, ckpt_root, metrics, want_category):
    """Train (or reuse checkpoint) + test evaluation for one seed."""
    ckpt_dir = os.path.join(ckpt_root, f"seed_{seed}")
    #  ShannonEntropy-like values are around 1e-4, so the default 4 decimals leave no significant digits.
    cfg = {"gpu_id": gpu_id, "seed": seed, "metrics": metrics,
           "checkpoint_dir": ckpt_dir,
           "metric_decimal_place": 6, "data_path": data_dir}
    #  Arguments that are None are not passed at all, so config file values are used as-is.
    for key, val in (("eval_batch_size", a.eval_batch_size),
                     ("epochs", a.epochs),
                     ("stopping_step", a.stopping_step),
                     ("train_batch_size", a.train_batch_size),
                     ("learning_rate", a.learning_rate)):
        if val is not None:
            cfg[key] = val
    if a.topk:
        cfg["topk"] = [int(x) for x in a.topk.split(",") if x.strip()]

    config = Config(model=a.model, dataset=a.dataset,
                    config_file_list=a.config_files.split(), config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    inject_reference_data(config, dataset, a.dataset, want_tourist=a.tourist,
                          data_dir=data_dir)

    #  Built before data_preparation. Once build() augments the sequential dataset,
    #  the original (user, item) pairs can no longer be recovered.
    hist, test_pair = user_history(dataset, config)
    config["user_hist_csr"] = hist
    revisit_rate, n_test = test_revisit_rate(hist, test_pair)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    found = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    if a.reuse and found:
        ckpt = found[-1]
        print(f"[reused] {ckpt}", flush=True)
        trainer.eval_collector.data_collect(train_data)
        result = trainer.evaluate(test_data, model_file=ckpt,
                                  show_progress=config["show_progress"])
    else:
        print("[training] no checkpoint - training from scratch", flush=True)
        trainer.fit(train_data, valid_data, saved=True,
                    show_progress=config["show_progress"])
        ckpt = trainer.saved_model_file
        result = trainer.evaluate(test_data, load_best_model=True,
                                  show_progress=config["show_progress"])

    ks = list(config["topk"])
    metrics_out = rename_metrics({k: float(v) for k, v in result.items()},
                                 ks, want_category, a.tourist)
    info = {
        "topk": ks,
        "config": {"epochs": config["epochs"],
                   "stopping_step": config["stopping_step"],
                   "train_batch_size": config["train_batch_size"],
                   "eval_batch_size": config["eval_batch_size"],
                   "learning_rate": config["learning_rate"]},
        "data": {"n_users": dataset.user_num - 1, "n_items": dataset.item_num - 1,
                 "n_interactions": dataset.inter_num,
                 "n_test_targets": n_test,
                 "orig_cat_size": int(config["orig_cat_size"] or 0) or None,
                 "ds_cat_size": int(len(np.unique(
                     config["item_cat_array"][config["item_cat_array"] >= 0])))},
    }
    run = {"seed": seed, "checkpoint": ckpt, "reused": bool(a.reuse and found),
           "metrics": {"test_revisit_rate": revisit_rate, **metrics_out}}
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-d", "--dataset", required=True)
    p.add_argument("--config_files", type=str, required=True)
    p.add_argument("--data_path", type=str, default=DATA_DIR,
                   help="Directory containing the dataset folders. If split by threshold combination,\n"
                        "pass dataset/<combo name> (default dataset/)")
    p.add_argument("--seeds", type=str, default=None,
                   help="e.g. 556-565 or 556,557. Empty means 556~565")
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--stopping_step", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--topk", type=str, default=None,
                   help="Comma-separated. Empty uses the config file value")
    p.add_argument("--tourist", action="store_true",
                   help="Also report travel purity (touristpurity)")
    p.add_argument("--reuse", action="store_true",
                   help="Skip training if a checkpoint exists")
    a = p.parse_args()

    seeds = parse_seeds(a.seeds)
    gpu_id = pick_gpu()
    print(f"[run] GPU {gpu_id or 'none (CPU)'} / seeds {seeds}", flush=True)

    data_dir = os.path.abspath(a.data_path)
    #  Use the folder name as a tag so results/checkpoints of different combinations do not mix.
    #  If DATA_ROOT is dataset/ itself, no combination folder is used.
    thr = os.path.basename(data_dir)
    combo = "" if thr == "dataset" else thr
    out_dir = os.path.join(RESULT_ROOT, combo)
    ckpt_root = os.path.join(SAVED_ROOT, combo, f"{a.dataset}-{a.model}")
    want_category = has_category(a.dataset, data_dir)
    metrics = [m for m in METRICS
               if m not in CATEGORY_METRICS or want_category]
    if a.tourist:
        metrics.append(TOURIST_METRIC)

    runs = []
    for seed in seeds:
        print(f"\n===== seed {seed} ({len(runs) + 1}/{len(seeds)}) =====", flush=True)
        run, info = run_seed(a, seed, gpu_id, data_dir, ckpt_root, metrics,
                             want_category)
        runs.append(run)

    mean, std = mean_std([r["metrics"] for r in runs])
    payload = {
        "meta": {"experiment": "exp1", "dataset": a.dataset, "model": a.model,
                 "thresholds": thr, "data_path": data_dir,
                 "seeds": seeds, "topk": info["topk"], "tourist": bool(a.tourist),
                 "timestamp": datetime.now().strftime("%b-%d-%Y_%H-%M-%S")},
        "config": info["config"],
        "data": info["data"],
        "metrics": mean,
        "metrics_std": std,
        "per_seed": runs,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"{a.dataset}-{a.model}-{payload['meta']['timestamp']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[mean ± std over {len(seeds)} seeds]")
    for k, v in mean.items():
        print(f"    {k:<34}{v}  ± {std.get(k, 0)}")
    print(f"saved: {path}", flush=True)


if __name__ == "__main__":
    main()
