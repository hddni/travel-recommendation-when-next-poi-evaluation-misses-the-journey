"""Split Gowalla into routine/travel and build the 6 EXP1 datasets.

Why Gowalla
    Foursquare's stage 1 rule is **category-level** (rvr(c)>=2.5 AND uvr(c)<=0.25).
    Gowalla has no venueCategory. The purpose of this script is to see whether uvr/rvr
    still work on a corpus without categories.

Two labeling variants
    direct      Defines stage 1 solely as **venue-level uvr <= UVR_THRESHOLD**.
                At the venue level rvr(v) = check-ins/unique users = 1/uvr(v), so
                the two metrics are exact reciprocals (observed Pearson 1.000000). The AND condition
                does nothing and reduces to the stricter side, so we state explicitly that
                it has collapsed to one dimension and use uvr only.
                Venues with check-ins < MIN_VENUE_CHECKINS automatically get uvr 1.0 and
                cannot be judged. Such venues are not removed (they remain as travel).

    cluster     Groups venues into a **spatial grid** and uses it as a pseudo-category. uvr(c) is
                the mean of per-venue uvr and rvr(c) is the mean of per-pair rvr, so the aggregation
                units differ and the reciprocal relation between the two metrics breaks (0.5-degree grid observed 0.795,
                Foursquare categories 0.80). So the AND condition regains its meaning and
                the Foursquare thresholds can be applied **as-is**.

    The stage 2 rule (rvr(u,v) >= PAIR_RVR_THRESHOLD) is identical in both variants.

Grid resolution
    GRID_DEG=0.5 (about 56km, metropolitan-area scale). Changing the resolution moves the uvr-rvr independence
    and the decidable coverage together. Observed (decidable check-in ratio / corr(uvr,1/rvr)):
        0.05deg  58.0% / 0.859      0.5deg   90.1% / 0.795   <- closest to Foursquare 0.80
        0.10deg  70.8% / 0.829      1.0deg   94.5% / 0.784
        0.20deg  81.4% / 0.816      2.0deg   97.2% / 0.772
    Splitting too finely makes clusters close to a single venue, producing the same degeneration as direct;
    grouping too coarsely mixes routine and travel within one cluster, and discriminative power disappears.

    TOURISM_WHITELIST is based on category names, so it cannot be applied to Gowalla.

Output (6 datasets)
    dataset/gowalla-{travel,routine,raw}/                   direct variant
    dataset/gowalla-{travel,routine,raw}-clustering/        cluster variant
    raw applies only k-core without judgment, so its content is the same in both variants.
    dataset/venue_category_stats/GOWALLA.csv   cluster stats (CategoryCoverage denominator)
    recbole/properties/dataset/gowalla-*.yaml

Usage:
    python dataset/build_gowalla.py
    python dataset/build_gowalla.py --grid_deg 0.2 --kcore 5
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
    INTER_HEADER, KCORE, MIN_VENUES_USED, PAIR_RVR_THRESHOLD, PROPS_DIR,
    RVR_THRESHOLD, UVR_THRESHOLD, dataset_yaml,
)

DATA = os.path.join(ROOT, "dataset")
SRC = os.path.join(DATA, "loc-gowalla_totalCheckins.txt.gz")
CAT_DIR = os.path.join(DATA, "venue_category_stats")
SCOPE = "GOWALLA"

GRID_DEG = 0.5              # side length (degrees) of the spatial grid used as a pseudo-category
MIN_VENUE_CHECKINS = 5      # minimum check-ins to trust uvr in direct stage 1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_deg", type=float, default=GRID_DEG)
    ap.add_argument("--kcore", type=int, default=KCORE)
    ap.add_argument("--uvr", type=float, default=UVR_THRESHOLD)
    ap.add_argument("--rvr", type=float, default=RVR_THRESHOLD)
    ap.add_argument("--pair_rvr", type=int, default=PAIR_RVR_THRESHOLD)
    ap.add_argument("--min_venues_used", type=int, default=MIN_VENUES_USED)
    ap.add_argument("--variants", type=str, default="direct,cluster")
    a = ap.parse_args()

    print("Reading Gowalla...", flush=True)
    df, coords = load_source()
    v, pair = venue_table(df)

    # ---------------- spatial clusters ----------------
    ckey = cluster_key(coords, a.grid_deg)
    ctab = cluster_table(v, ckey, MIN_CLUSTER_CHECKINS)
    os.makedirs(CAT_DIR, exist_ok=True)
    ctab.sort_values("uvr", ascending=False, na_position="last").to_csv(
        os.path.join(CAT_DIR, f"{SCOPE}.csv"), index=False)
    judged = ctab[(ctab["uvr"].notna()) & (ctab["n_venues_used"] >= a.min_venues_used)]
    corr = float(np.corrcoef(judged["uvr"], 1 / judged["rvr"])[0, 1])
    print(f"  clusters {len(ctab):,} ({a.grid_deg} deg)  labelable {len(judged):,}  "
          f"corr(uvr,1/rvr) {corr:.4f}", flush=True)

    pair_hi = set(pair.index[pair >= a.pair_rvr])
    mi = pd.MultiIndex.from_arrays([df["user_id"], df["item_id"]])
    summary = []

    for variant in [s.strip() for s in a.variants.split(",") if s.strip()]:
        if variant == "direct":
            s1_venues = set(v.index[(v["ck"] >= MIN_VENUE_CHECKINS)
                                    & (v["uvr"] <= a.uvr)])
            cats, suffix = None, ""
            note1 = f"stage 1: venue uvr<={a.uvr} (check-ins>={MIN_VENUE_CHECKINS})"
        else:
            drop = set(judged["venueCategory"][(judged["rvr"] >= a.rvr)
                                               & (judged["uvr"] <= a.uvr)])
            s1_venues = set(ckey.index[ckey.isin(drop)])
            slug = ckey.map(lambda x: f"grid_{x}")
            cats = pd.DataFrame({"venue_category_id": slug, "venue_category": slug})
            suffix = "-clustering"
            note1 = (f"stage 1: {a.grid_deg}-deg grid clusters rvr>={a.rvr} AND "
                     f"uvr<={a.uvr} ({len(drop):,}/{len(judged):,} clusters)")

        d1 = df["item_id"].isin(s1_venues).to_numpy()
        d2 = mi.isin(pair_hi) & ~d1
        daily = d1 | d2
        target_p = float(daily.mean())
        print(f"\n{'='*92}\n[{variant}] {note1}\n"
              f"  stage 1 {int(d1.sum()):,} ({100*d1.mean():.2f}%)  "
              f"stage 2 {int(d2.sum()):,} ({100*d2.mean():.2f}%)  "
              f"-> routine {int(daily.sum()):,} ({100*target_p:.2f}%)", flush=True)

        work = df.copy()
        work["is_daily"] = daily
        #  All three arms apply only k-core. raw is the judgment-independent all-data reference point.
        arms = {f"gowalla-travel{suffix}": work[~daily],
                f"gowalla-routine{suffix}": work[daily],
                f"gowalla-raw{suffix}": work}
        for name, sub in arms.items():
            out = iterative_kcore(sub.copy(), "user_id", "item_id", a.kcore)
            _write(name, out, coords, cats, a, note1, variant, summary, target_p)

    print(f"\n{'='*92}\nSummary")
    print(pd.DataFrame(summary).to_string(index=False))
    with open(os.path.join(DATA, "gowalla_build_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _write(name, out, coords, cats, a, note1, variant, summary, target_p, extra=""):
    out = out.sort_values(["user_id", "timestamp"], kind="mergesort")
    item = build_item_frame(out["item_id"].to_numpy(), coords, cats)
    out_dir = os.path.join(DATA, name)
    write_dataset(out_dir, name, out[INTER_COLS], item, INTER_HEADER)
    note = (f"Gowalla - {'routine removed -> travel only' if 'travel' in name else ''}"
            f"{'travel removed -> routine only' if 'routine' in name else ''}"
            f"{'all data, k-core only, no labeling' if 'raw' in name else ''}"
            f" ({variant} labeling, {note1})")
    os.makedirs(PROPS_DIR, exist_ok=True)
    with open(os.path.join(PROPS_DIR, f"{name}.yaml"), "w", encoding="utf-8") as f:
        f.write(dataset_yaml(name, a.kcore, note).replace(
            "dataset/build_travel.py", "dataset/build_gowalla.py"))
    report(name, out, item, kcore=a.kcore)
    if extra:
        print(f"    {extra}", flush=True)
    summary.append({"dataset": name, "variant": variant,
                    "users": int(out.user_id.nunique()),
                    "items": int(out.item_id.nunique()),
                    "inter": int(len(out)),
                    "daily_ratio": round(float(out.is_daily.mean()), 6),
                    "target_daily_ratio": round(target_p, 6)})


if __name__ == "__main__":
    main()
