"""Split Gowalla into routine/travel and build the EXP1 datasets.

Gowalla has no venue category, so stage 1 uses a **spatial grid** (GRID_DEG) as the
pseudo-category. Everything else is the same rule as Foursquare (daily_label.py):

    stage 1   per grid bin, mean UVR / RVR over venues with >= MIN_CLUSTER_CHECKINS
              check-ins (the same aggregation as venue_category_stats.py)
                  A  RVR >= t_R  AND  UVR <= t_U   -> routine
                  B  RVR <  t_R  AND  UVR >  t_U   -> travel
                  C  otherwise, E  bins with fewer than MIN_VENUES_USED usable venues
                                                   -> stage 2
    stage 2   daily_label.label(): Ratio(u,v) >= REL_TAU for |U_v| >= 2,
              n_uv >= ceil(1/t_U) for |U_v| = 1

Grid resolution
    GRID_DEG=0.5 (about 56km, metropolitan-area scale). Changing the resolution moves the uvr-rvr independence
    and the decidable coverage together. Observed (decidable check-in ratio / corr(uvr,1/rvr)):
        0.05deg  58.0% / 0.859      0.5deg   90.1% / 0.795   <- closest to Foursquare 0.80
        0.10deg  70.8% / 0.829      1.0deg   94.5% / 0.784
        0.20deg  81.4% / 0.816      2.0deg   97.2% / 0.772
    Splitting too finely makes clusters close to a single venue; grouping too coarsely mixes
    routine and travel within one cluster, and discriminative power disappears.

    TOURISM_WHITELIST is based on category names, so it cannot be applied to Gowalla.

The whole corpus is labeled. The continental-US restriction is applied only when the
data are analyzed (figures), not here.

Output
    dataset/gowalla-{travel,routine,raw}/
    dataset/venue_category_stats/GOWALLA.csv   grid-bin stats
    recbole/properties/dataset/gowalla-*.yaml

Usage:
    python preprocess/build_gowalla.py
    python preprocess/build_gowalla.py --grid_deg 0.2 --kcore 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from atomic_common import (  # noqa: E402
    build_item_frame, drop_invalid_coords, iterative_kcore, parse_iso_time, report,
    write_dataset,
)
from build_travel import (  # noqa: E402
    INTER_HEADER, KCORE, MIN_VENUES_USED, PROPS_DIR, RVR_THRESHOLD, UVR_THRESHOLD,
    dataset_yaml,
)
from daily_label import REL_TAU  # noqa: E402
import daily_label  # noqa: E402

DATA = os.path.join(ROOT, "dataset")
SRC = os.path.join(DATA, "loc-gowalla_totalCheckins.txt.gz")
CAT_DIR = os.path.join(DATA, "venue_category_stats")
SCOPE = "GOWALLA"

GRID_DEG = 0.5              # side length (degrees) of the spatial grid used as a pseudo-category
MIN_CLUSTER_CHECKINS = 10   # minimum check-ins for a venue to be included in the cluster uvr mean
                            # (default of --min_checkins in venue_category_stats.py)
INTER_COLS = ["user_id", "item_id", "timestamp", "local_timestamp"]


def load_source():
    """SNAP Gowalla: user / UTC time / lat / lon / location_id, no header."""
    t0 = time.time()
    df = pd.read_csv(SRC, sep="\t", header=None,
                     names=["user_id", "time", "latitude", "longitude", "item_id"],
                     dtype={"user_id": np.int64, "item_id": np.int64})
    df["timestamp"] = parse_iso_time(df["time"])
    #  Gowalla provides only UTC. Foursquare has offsets, so local time could be
    #  constructed, but that is impossible here. Neither RQ uses local time
    #  (labeling is uvr/rvr, splitting is chronological), so we fill in the same value just to match columns.
    df["local_timestamp"] = df["timestamp"]
    n0 = len(df)
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(["user_id", "item_id", "timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    #  Some venues are recorded with multiple coordinates (297). Fix them to the median.
    coords = df.groupby("item_id")[["latitude", "longitude"]].median()
    df = drop_invalid_coords(df, "item_id", coords)
    coords = coords.loc[coords.index.isin(df["item_id"].unique())]
    print(f"  check-ins {len(df):,} (original {n0:,})  users {df.user_id.nunique():,}  "
          f"venues {df.item_id.nunique():,}  ({time.time()-t0:.0f}s)", flush=True)
    return df.reset_index(drop=True), coords


def venue_table(df, theta=2):
    """Per-venue check-in count / unique user count / revisitor count / uvr / rvr."""
    pair = df.groupby(["user_id", "item_id"]).size()
    ck = pair.groupby("item_id").sum()
    us = pair.groupby("item_id").size()
    rp = pair[pair >= theta].groupby("item_id").size()
    v = pd.DataFrame({"ck": ck, "us": us})
    v["rp"] = rp.reindex(v.index).fillna(0).astype(np.int64)
    v["uvr"] = v["us"] / v["ck"]
    v["rvr"] = v["rp"] / v["us"]        # revisitor ratio (same definition as daily_label.py)
    return v, pair


def cluster_key(coords, deg):
    return (np.floor(coords["latitude"] / deg).astype(np.int64).astype(str)
            + "_" + np.floor(coords["longitude"] / deg).astype(np.int64).astype(str))


def cluster_table(v, ckey, min_checkins):
    """Per-cluster uvr/rvr. Same method as the category aggregation in venue_category_stats.py.

    uvr(c) = mean uvr over venues with check-ins >= min_checkins
    rvr(c) = mean rvr over the same venue set   (rvr(v) = revisitor ratio)

    The definitions and filters must be identical to Foursquare's category aggregation for the
    thresholds 0.5/0.25 to carry over as-is. rvr_all is the old definition (mean visits per pair), for comparison.
    """
    d = v.join(ckey.rename("c"))
    g = d.groupby("c")
    out = pd.DataFrame({"n_venues": g.size(), "checkins": g["ck"].sum(),
                        "n_pairs": g["us"].sum()})
    out["rvr_all"] = out["checkins"] / out["n_pairs"]
    sel = d[d["ck"] >= min_checkins].groupby("c")
    out["uvr"] = sel["uvr"].mean()
    out["rvr"] = sel["rvr"].mean()
    out["n_venues_used"] = sel.size().reindex(out.index).fillna(0).astype(np.int64)
    out["n_pairs_rvr"] = sel["rp"].sum().reindex(out.index).fillna(0).astype(np.int64)
    return out.reset_index().rename(columns={"c": "venueCategory"})


def cluster_params(ctab, rvr_th, uvr_th, min_used):
    """Grid-bin quadrants in the form daily_label.label() expects (index = bin)."""
    t = ctab.set_index("venueCategory")
    judged = t["uvr"].notna() & t["rvr"].notna() & (t["n_venues_used"] >= min_used)
    a = judged & (t["rvr"] >= rvr_th) & (t["uvr"] <= uvr_th)
    b = judged & (t["rvr"] < rvr_th) & (t["uvr"] > uvr_th)
    quad = np.where(~judged, "E", np.where(a, "A", np.where(b, "B", "C")))
    return pd.DataFrame({"uvr": t["uvr"], "rvr": t["rvr"], "quadrant": quad},
                        index=t.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_deg", type=float, default=GRID_DEG)
    ap.add_argument("--kcore", type=int, default=KCORE)
    ap.add_argument("--uvr", type=float, default=UVR_THRESHOLD)
    ap.add_argument("--rvr", type=float, default=RVR_THRESHOLD)
    ap.add_argument("--rel_tau", type=float, default=REL_TAU,
                    help="stage 2: relative multiple vs. other users at the same venue")
    ap.add_argument("--min_venues_used", type=int, default=MIN_VENUES_USED)
    a = ap.parse_args()

    print("Reading Gowalla...", flush=True)
    df, coords = load_source()
    v, _ = venue_table(df)

    # ---------------- spatial clusters ----------------
    ckey = cluster_key(coords, a.grid_deg)
    ctab = cluster_table(v, ckey, MIN_CLUSTER_CHECKINS)
    os.makedirs(CAT_DIR, exist_ok=True)
    ctab.sort_values("uvr", ascending=False, na_position="last").to_csv(
        os.path.join(CAT_DIR, f"{SCOPE}.csv"), index=False)
    params = cluster_params(ctab, a.rvr, a.uvr, a.min_venues_used)
    q = params["quadrant"].value_counts()
    print(f"  clusters {len(params):,} ({a.grid_deg} deg)  "
          f"A {q.get('A', 0):,} / B {q.get('B', 0):,} / C {q.get('C', 0):,} / "
          f"E {q.get('E', 0):,}", flush=True)

    #  Same two-stage rule as Foursquare, with the grid bin as the category.
    is_daily, st = daily_label.label(df, ckey, params, rel_tau=a.rel_tau,
                                     uvr_th=a.uvr)
    daily_label.report(st)
    note1 = (f"stage 1: {a.grid_deg}-deg grid bins rvr>={a.rvr} AND uvr<={a.uvr}; "
             f"stage 2: relative multiple>={a.rel_tau}")

    slug = ckey.map(lambda x: f"grid_{x}")
    cats = pd.DataFrame({"venue_category_id": slug, "venue_category": slug})
    work = df.copy()
    work["is_daily"] = is_daily
    target_p = float(is_daily.mean())
    summary = []
    #  All three arms apply only k-core. raw is the judgment-independent all-data reference point.
    arms = {"gowalla-travel": work[~is_daily],
            "gowalla-routine": work[is_daily],
            "gowalla-raw": work}
    for name, sub in arms.items():
        out = iterative_kcore(sub.copy(), "user_id", "item_id", a.kcore)
        _write(name, out, coords, cats, a, note1, summary, target_p)

    print(f"\n{'='*92}\nSummary")
    print(pd.DataFrame(summary).to_string(index=False))
    with open(os.path.join(DATA, "gowalla_build_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _write(name, out, coords, cats, a, note1, summary, target_p):
    out = out.sort_values(["user_id", "timestamp"], kind="mergesort")
    item = build_item_frame(out["item_id"].to_numpy(), coords, cats)
    out_dir = os.path.join(DATA, name)
    write_dataset(out_dir, name, out[INTER_COLS], item, INTER_HEADER)
    note = (f"Gowalla - {'routine removed -> travel only' if 'travel' in name else ''}"
            f"{'travel removed -> routine only' if 'routine' in name else ''}"
            f"{'all data, k-core only, no labeling' if 'raw' in name else ''}"
            f" ({note1})")
    os.makedirs(PROPS_DIR, exist_ok=True)
    with open(os.path.join(PROPS_DIR, f"{name}.yaml"), "w", encoding="utf-8") as f:
        f.write(dataset_yaml(name, a.kcore, note).replace(
            "dataset/build_travel.py", "preprocess/build_gowalla.py"))
    report(name, out, item, kcore=a.kcore)
    summary.append({"dataset": name,
                    "users": int(out.user_id.nunique()),
                    "items": int(out.item_id.nunique()),
                    "inter": int(len(out)),
                    "daily_ratio": round(float(out.is_daily.mean()), 6),
                    "target_daily_ratio": round(target_p, 6)})


if __name__ == "__main__":
    main()
