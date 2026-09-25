"""Build the travel recommendation dataset (travelset).

In the conventional leave-one-out the answer is "the single next check-in", so routine
revisits make up most of the ground truth and performance effectively becomes the
difficulty of revisit prediction. This dataset avoids that trap by changing the ground
truth to **the set of the last travel visits**.

Split
    test   = the user's last N_TEST (3) travel check-ins.
             Routine check-ins in between are discarded (treated as one continuous trip).
    valid  = the 1 travel check-in right before that
    train  = all check-ins up to just before valid (routine and travel mixed as-is)
    condition = travel check-ins >= N_TEST+1,  train >= MIN_PREFIX

    travel = is_daily == 0.  No home anchor or distance condition is used. This is to
    keep users who have only travel check-ins (10-27% of all users).

k-core and the split break each other
    Since the split discards routine check-ins after valid, some venues drop below k
    occurrences, and when k-core removes travel check-ins some users fail the condition.
    Both are monotonically decreasing, so **alternating repeatedly** converges to a fixed
    point. Running only once breaks the constraints.

Routine labels are computed only once on the raw data before filtering (dataset/daily_label.py).
Recomputing each round would change the stage-2 comparison group (per-venue visitor
distribution), so the meaning of "routine" would change from round to round.

Output: dataset/foursquare-<scope>-travelset/
    inter.tsv   uid iid timestamp local_timestamp is_daily split(0=train,1=valid,2=test)
    items.tsv   iid venue_id category latitude longitude is_daily
    users.tsv   uid user_token n_train n_travel n_answer
    meta.json

Usage:
    python dataset/build_travelset.py
    python dataset/build_travelset.py --scopes US,JP --min_prefix 20
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset")
sys.path.insert(0, HERE)

from atomic_common import drop_invalid_coords  # noqa: E402
from build_travel import ALL_COUNTRIES, load_source, scope_subset  # noqa: E402
from daily_label import (  # noqa: E402
    MIN_VENUES_USED, REL_TAU, RVR_THRESHOLD, UVR_THRESHOLD,
)
import daily_label  # noqa: E402

SCOPES = ["ALL", "DE", "SG", "KR", "JP", "US"]
KCORE = 5
MIN_PREFIX = 10
N_TEST = 3
MAX_ROUND = 40


def split_mask(u, daily, mask, nu):
    """New mask obtained by applying the split rule to the current mask, plus auxiliary arrays."""
    idx = np.flatnonzero(mask)
    ku, kd = u[idx], daily[idx]
    starts = np.searchsorted(ku, np.arange(nu))
    pos = np.arange(len(ku)) - starts[ku]
    tv = ~kd
    tu = ku[tv]
    tcnt = np.bincount(tu, minlength=nu)
    tst = np.searchsorted(tu, np.arange(nu))
    trank = np.full(len(ku), -1, np.int64)
    trank[tv] = tcnt[tu] - 1 - (np.arange(len(tu)) - tst[tu])
    vpos = np.full(nu, -1, np.int64)
    sel = trank == N_TEST
    vpos[ku[sel]] = pos[sel]
    ok = (tcnt >= N_TEST + 1) & (vpos >= MIN_PREFIX)
    row = ok[ku] & ((pos < vpos[ku]) | ((trank >= 0) & (trank <= N_TEST)))
    new = np.zeros(len(u), bool)
    new[idx[row]] = True
    return new, trank, ku, idx


def fit(u, i, daily, nu, ni, k):
    """Alternately apply k-core and the split until both hold."""
    mask = np.ones(len(u), bool)
    for rnd in range(1, MAX_ROUND + 1):
        prev = int(mask.sum())
        while True:
            uc = np.bincount(u[mask], minlength=nu)
            ic = np.bincount(i[mask], minlength=ni)
            keep = mask & (uc[u] >= k) & (ic[i] >= k)
            if int(keep.sum()) == int(mask.sum()):
                break
            mask = keep
        if not mask.any():
            return mask, rnd
        mask = split_mask(u, daily, mask, nu)[0]
        if int(mask.sum()) == prev or not mask.any():
            return mask, rnd
    return mask, MAX_ROUND


def main():
    global MIN_PREFIX, N_TEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", type=str, default=",".join(SCOPES))
    ap.add_argument("--kcore", type=int, default=KCORE)
    ap.add_argument("--min_prefix", type=int, default=MIN_PREFIX)
    ap.add_argument("--n_test", type=int, default=N_TEST)
    ap.add_argument("--rvr", type=float, default=RVR_THRESHOLD)
    ap.add_argument("--uvr", type=float, default=UVR_THRESHOLD)
    ap.add_argument("--min_venues_used", type=int, default=MIN_VENUES_USED)
    ap.add_argument("--rel_tau", type=float, default=REL_TAU)
    ap.add_argument("--out_root", type=str, default=DATA,
                    help="Parent directory in which to create dataset folders. To split by\n"
                         "threshold combination, pass dataset/<combination name>")
    args = ap.parse_args()
    MIN_PREFIX, N_TEST = args.min_prefix, args.n_test
    scopes = [s.strip().upper() for s in args.scopes.split(",") if s.strip()]

    poi, chk = load_source()
    pidx = poi.set_index("venue")
    coords = pidx[["latitude", "longitude"]]
    v2cat = pidx["cat"]
    del poi

    summary = []
    for scope in scopes:
        name = f"foursquare-{scope.lower()}-travelset"
        sub = scope_subset(chk, scope, ALL_COUNTRIES if scope == "ALL" else None)
        c = sub["item_id"].map(v2cat)
        sub = sub[c.notna() & (c != "")].copy()
        sub = drop_invalid_coords(sub, "item_id", coords)

        # Routine labeling. Computed only once on the raw data before filtering.
        params = daily_label.category_params(scope, args.rvr, args.uvr,
                                             args.min_venues_used, verbose=False)
        is_daily, st = daily_label.label(sub, v2cat, params, rel_tau=args.rel_tau,
                                         uvr_th=args.uvr)
        sub["daily"] = is_daily
        drop = set(params.index[params["quadrant"] == "A"])   # category-confirmed routine
        raw_u, raw_n = sub["user_id"].nunique(), len(sub)
        print(f"\n=== {name}  ({args.kcore}-core, prefix>={MIN_PREFIX}) ===")
        print(f"    original: users {raw_u:,} / check-ins {raw_n:,}", flush=True)

        uc_, uu = pd.factorize(sub["user_id"])
        ic_, iu = pd.factorize(sub["item_id"])
        srt = np.lexsort((sub["timestamp"].to_numpy(), uc_))
        u = uc_[srt].astype(np.int64)
        i = ic_[srt].astype(np.int64)
        daily = sub["daily"].to_numpy()[srt]
        ts = sub["timestamp"].to_numpy()[srt]
        lts = sub["local_timestamp"].to_numpy()[srt]
        nu, ni = len(uu), len(iu)

        mask, rnd = fit(u, i, daily, nu, ni, args.kcore)
        if not mask.any():
            print("    No data left.", flush=True)
            continue
        _, trank, ku, idx = split_mask(u, daily, mask, nu)
        ki = i[idx]
        #  trank is the reverse rank among the user's **travel check-ins** (0=last). Routine is -1.
        #  Travel check-ins inside the prefix have trank >= N_TEST+1, so they must be train.
        split = np.where((trank >= 0) & (trank <= N_TEST - 1), 2,
                         np.where(trank == N_TEST, 1, 0))

        # id remapping (0 is padding)
        fu = np.unique(ku)
        fi = np.unique(ki)
        umap = np.full(nu, 0, np.int64); umap[fu] = np.arange(1, len(fu) + 1)
        imap = np.full(ni, 0, np.int64); imap[fi] = np.arange(1, len(fi) + 1)

        inter = pd.DataFrame({
            "uid": umap[ku], "iid": imap[ki],
            "timestamp": ts[idx], "local_timestamp": lts[idx],
            "is_daily": daily[idx].astype(np.int8), "split": split.astype(np.int8),
        }).sort_values(["uid", "timestamp"], kind="mergesort")

        venue = pd.Index(iu[fi])
        ic2 = coords.reindex(venue)
        items = pd.DataFrame({
            "iid": np.arange(1, len(fi) + 1), "venue_id": venue,
            "category": v2cat.reindex(venue).values,
            "latitude": ic2["latitude"].to_numpy(),
            "longitude": ic2["longitude"].to_numpy(),
        })
        #  is_daily in items marks only **category-confirmed routine (A)**.
        #  The stage-2 labeling, which differs per (u,v) pair, is in is_daily of inter.tsv.
        items["is_daily"] = items["category"].isin(drop).astype(np.int8)

        g = inter.groupby("uid")
        n_train = g.apply(lambda x: int((x.split == 0).sum()))
        n_travel = g.apply(lambda x: int((~x.is_daily.astype(bool)).sum()))
        n_answer = inter[inter.split == 2].groupby("uid")["iid"].nunique()
        users = pd.DataFrame({
            "uid": np.arange(1, len(fu) + 1), "user_token": uu[fu],
            "n_train": n_train.reindex(np.arange(1, len(fu) + 1)).values,
            "n_travel": n_travel.reindex(np.arange(1, len(fu) + 1)).values,
            "n_answer": n_answer.reindex(np.arange(1, len(fu) + 1)).values,
        })

        # ---- Validation ----
        assert (users["n_train"] >= MIN_PREFIX).all(), "prefix condition violated"
        assert (inter.groupby("uid")["split"].apply(lambda s: (s == 1).sum()) == 1).all()
        assert (inter.groupby("uid")["split"].apply(lambda s: (s == 2).sum()) == N_TEST).all()
        vc_i = inter["iid"].value_counts()
        vc_u = inter["uid"].value_counts()
        assert not inter[inter.split == 2]["is_daily"].astype(bool).any(), "routine check-ins mixed into answers"

        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        inter.to_csv(os.path.join(out_dir, "inter.tsv"), sep="\t", index=False)
        items.to_csv(os.path.join(out_dir, "items.tsv"), sep="\t", index=False)
        users.to_csv(os.path.join(out_dir, "users.tsv"), sep="\t", index=False)

        # Fraction of ground-truth items already present in that user's training span
        tr = inter[inter.split != 2]
        te = inter[inter.split == 2]
        kt = set(map(tuple, tr[["uid", "iid"]].to_numpy()))
        seen = float(np.mean([tuple(x) in kt for x in te[["uid", "iid"]].to_numpy()]))

        meta = dict(
            name=name, scope=scope, kcore=args.kcore, min_prefix=MIN_PREFIX,
            n_test_per_user=N_TEST, rounds=int(rnd),
            raw_users=int(raw_u), raw_inter=int(raw_n),
            n_users=int(len(fu)), n_items=int(len(fi)), n_interactions=int(len(inter)),
            n_train=int((inter.split == 0).sum()), n_valid=int((inter.split == 1).sum()),
            n_test=int((inter.split == 2).sum()),
            answer_items_mean=round(float(users["n_answer"].mean()), 4),
            prefix_mean=round(float(users["n_train"].mean()), 2),
            prefix_median=float(users["n_train"].median()),
            daily_ratio=round(float(inter["is_daily"].mean()), 6),
            test_seen_rate=round(seen, 6),
            item_min=int(vc_i.min()), user_min=int(vc_u.min()),
        )
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"    converged after round {rnd} | users {meta['n_users']:,} / venues {meta['n_items']:,} "
              f"/ check-ins {meta['n_interactions']:,}")
        print(f"    answer items {meta['answer_items_mean']:.2f} / revisit "
              f"{seen*100:.1f}% / prefix median {meta['prefix_median']:.0f} "
              f"/ item_min {meta['item_min']}", flush=True)
        summary.append(meta)

    if summary:
        print("\n" + "=" * 104)
        print(f"  {'dataset':<34}{'users':>9}{'items':>10}{'inter':>13}"
              f"{'answer':>7}{'revisit':>8}{'retained':>8}")
        for m in summary:
            print(f"  {m['name']:<34}{m['n_users']:>9,}{m['n_items']:>10,}"
                  f"{m['n_interactions']:>13,}{m['answer_items_mean']:>7.2f}"
                  f"{m['test_seen_rate']*100:>7.1f}%{m['n_users']/m['raw_users']*100:>7.1f}%")


if __name__ == "__main__":
    main()
