"""Build venue/pair label tables for metric 2 (travel purity).

Why build it separately
    is_daily in inter.tsv was assigned from rvr(u,v) over **the whole period before the split**.
    Using that label as-is for evaluation leaks visit counts from the test/valid
    periods into the metric. Here the test/valid check-ins are removed from the
    labeling population and counts are recomputed.

Population P
    Raw check-ins (category/coordinate filters only, before k-core)
      - check-ins with split ∈ {1,2} in the travelset

    The reason for "raw" is that if the majority-vote denominator differed per filter,
    comparisons across scopes would break.

Labels (rules of dataset/daily_label.py re-applied on population P)
    Uses stage-1 category labeling (A/B/C/E) + stage-2 relative repetition labeling as-is.
    The stage-2 comparison group (per-venue visitor distribution) is computed within P,
    so valid/test does not leak into the labels.

The majority vote is **per user**. Counting per check-in would let a single user who
visited a venue 500 times decide that venue's label alone. Since "repeat visits =
routine" is already the labeling rule, check-in weighting would amount to applying
the same rule twice.

Output: dataset/venue_tourist_label/
    <SCOPE>.venue.tsv   venue_id n_users n_users_tourist frac_tourist is_cat_daily
    <SCOPE>.pair.tsv    user_token venue_id
        Only the (user, venue) pairs in the travelset that **are not category-routine
        but became routine via rvr_P**. Other pairs are determined by is_cat_daily, so
        they are not written.

Usage:
    python dataset/build_tourist_label.py
    python dataset/build_tourist_label.py --scopes US,JP
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from atomic_common import drop_invalid_coords  # noqa: E402
from build_travel import ALL_COUNTRIES, load_source, scope_subset  # noqa: E402
from daily_label import (  # noqa: E402
    MIN_VENUES_USED, REL_TAU, RVR_THRESHOLD, UVR_THRESHOLD,
)
import daily_label  # noqa: E402

SCOPES = ["ALL", "US", "BR", "JP", "KR", "SG"]
DATA = os.path.join(os.path.dirname(HERE), "dataset")
OUT = os.path.join(DATA, "venue_tourist_label")


def travelset_dir(scope):
    return os.path.join(DATA, f"foursquare-{scope.lower()}-travelset")


def holdout_keys(d):
    """travelset valid/test check-ins as raw keys (user_id, item_id, timestamp)."""
    inter = pd.read_csv(os.path.join(d, "inter.tsv"), sep="\t")
    items = pd.read_csv(os.path.join(d, "items.tsv"), sep="\t", dtype={"venue_id": str})
    users = pd.read_csv(os.path.join(d, "users.tsv"), sep="\t", dtype={"user_token": str})
    ho = inter[inter["split"].isin((1, 2))]
    return pd.DataFrame({
        "user_id": ho["uid"].map(dict(zip(users["uid"], users["user_token"]))).values,
        "item_id": ho["iid"].map(dict(zip(items["iid"], items["venue_id"]))).values,
        "timestamp": ho["timestamp"].values,
    }), users, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", type=str, default=",".join(SCOPES))
    ap.add_argument("--rvr", type=float, default=RVR_THRESHOLD)
    ap.add_argument("--uvr", type=float, default=UVR_THRESHOLD)
    ap.add_argument("--min_venues_used", type=int, default=MIN_VENUES_USED)
    ap.add_argument("--rel_tau", type=float, default=REL_TAU)
    args = ap.parse_args()
    scopes = [s.strip().upper() for s in args.scopes.split(",") if s.strip()]

    poi, chk = load_source()
    pidx = poi.set_index("venue")
    coords = pidx[["latitude", "longitude"]]
    v2cat = pidx["cat"]
    del poi
    os.makedirs(OUT, exist_ok=True)

    for scope in scopes:
        d = travelset_dir(scope)
        if not os.path.isdir(d):
            print(f"[skip] travelset not found: {d}", flush=True)
            continue
        print(f"\n=== {scope} ===", flush=True)

        sub = scope_subset(chk, scope, ALL_COUNTRIES if scope == "ALL" else None)
        c = sub["item_id"].map(v2cat)
        sub = sub[c.notna() & (c != "")]
        sub = drop_invalid_coords(sub, "item_id", coords)
        n_raw = len(sub)

        #  --- Build P: remove valid/test check-ins from the raw data ---
        ho, users, items = holdout_keys(d)
        n_dup = int(ho.duplicated().sum())
        ho = ho.drop_duplicates()
        merged = sub.merge(ho.assign(_ho=1), on=["user_id", "item_id", "timestamp"],
                           how="left")
        merged.index = sub.index
        removed = int(merged["_ho"].notna().sum())
        P = sub[merged["_ho"].isna()]
        print(f"    original {n_raw:,} - holdout {removed:,} = P {len(P):,}"
              f"  (holdout keys {len(ho):,}, duplicate keys {n_dup:,})", flush=True)
        if removed > len(ho):
            print(f"    [warn] a single key matched multiple original rows "
                  f"({removed - len(ho):,} extra rows removed). Duplicate rows with identical (user, venue, time).",
                  flush=True)

        #  --- Labels: apply daily_label as-is on population P ---
        params = daily_label.category_params(scope, args.rvr, args.uvr,
                                             args.min_venues_used, verbose=False)
        is_daily, st = daily_label.label(P, v2cat, params, rel_tau=args.rel_tau,
                                         uvr_th=args.uvr)
        daily_label.report(st)
        cat_daily_set = set(params.index[params["quadrant"] == "A"])
        uc, uu = pd.factorize(P["user_id"])
        ic, iu = pd.factorize(P["item_id"])
        cat_daily = np.array([str(v2cat.get(v, "")) in cat_daily_set for v in iu])
        pair = pd.DataFrame({"u": uc, "i": ic, "daily": is_daily})
        rvr = pair.groupby(["u", "i"], sort=False)["daily"].max().reset_index()

        #  --- Venue table: per-user majority vote ---
        g = rvr.groupby("i", sort=True)
        n_users = g.size()
        n_tour = g["daily"].apply(lambda s: int((~s).sum()))
        venue = pd.DataFrame({
            "venue_id": iu[n_users.index.to_numpy()],
            "n_users": n_users.to_numpy(),
            "n_users_tourist": n_tour.to_numpy(),
            "is_cat_daily": cat_daily[n_users.index.to_numpy()].astype(np.int8),
        })
        venue["frac_tourist"] = venue["n_users_tourist"] / venue["n_users"]
        venue = venue[["venue_id", "n_users", "n_users_tourist", "frac_tourist",
                       "is_cat_daily"]]
        vpath = os.path.join(OUT, f"{scope}.venue.tsv")
        venue.to_csv(vpath, sep="\t", index=False)

        #  --- Pair table: pairs in the travelset that are "category travel but routine via rvr" ---
        #  metrics.py looks up this table only when v is in u's prefix. Any user and venue
        #  that are each in the travelset could be looked up, so we store that superset.
        ts_users = set(users["user_token"])
        ts_items = set(items["venue_id"])
        sel = rvr[(~cat_daily[rvr["i"].to_numpy()]) & rvr["daily"].to_numpy()]
        pu, pv = uu[sel["u"].to_numpy()], iu[sel["i"].to_numpy()]
        keep = np.array([a in ts_users and b in ts_items for a, b in zip(pu, pv)], bool) \
            if len(sel) else np.zeros(0, bool)
        pair_df = pd.DataFrame({"user_token": pu[keep], "venue_id": pv[keep]})
        ppath = os.path.join(OUT, f"{scope}.pair.tsv")
        pair_df.to_csv(ppath, sep="\t", index=False)

        #  --- travelset items with no observation in P (tier 3) ---
        obs = set(venue["venue_id"])
        undef = int(sum(1 for v in items["venue_id"] if v not in obs))
        print(f"    venues {len(venue):,} (median travel ratio {venue['frac_tourist'].median():.3f}, "
              f"category-routine {int(venue['is_cat_daily'].sum()):,})", flush=True)
        print(f"    pair exceptions {len(pair_df):,} | travelset items unobserved in P "
              f"{undef:,}/{len(items):,} ({100*undef/len(items):.3f}%)", flush=True)
        print(f"    -> {vpath}\n    -> {ppath}", flush=True)


if __name__ == "__main__":
    main()
