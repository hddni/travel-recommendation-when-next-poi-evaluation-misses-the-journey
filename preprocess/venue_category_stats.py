"""Compute uvr / rvr per venueCategory and save them as CSV.

Computes, at the category level, the two core metrics that separate travel
venues from routine venues. They point in opposite directions, so looking at
them together gives high discriminative power.

    uvr (unique visitor ratio)  higher = more travel
        uvr(v) = (number of distinct users who visited venue v) / (total check-ins at venue v)
        uvr(c) = mean uvr over venues belonging to category c
                 Venues with fewer than min_checkins check-ins are excluded.
                 Over 60% of all venues have only 1-2 check-ins and automatically get uvr=1.0,
                 so without filtering every category collapses to 0.8-0.9.

    rvr (repetitive visit ratio)  higher = more routine
        rvr(v) = |{u in U_v : count(u,v) >= THETA}| / |U_v|      (THETA default 2)
                 = fraction of the venue's visitors who visited it THETA or more times
        rvr(c) = mean rvr over venues belonging to category c
                 As with uvr, venues with fewer than min_checkins check-ins are excluded.
                 (With few check-ins no revisit can occur, so rvr automatically becomes 0.
                  This is the exact mirror image of uvr automatically becoming 1.0.)

        The key is that the denominator is all visitors. A venue where 3 of 100
        visitors revisited and one where 3 of 5 revisited are entirely different in nature.

    Difference from the old definition (rvr_all) - still reported for reference
        rvr_all(c) = checkins(c) / n_pairs(c) = mean number of visits per user.
        At the venue level rvr_all(v) = ck/|U_v| = 1/uvr(v), i.e. the reciprocal of uvr,
        and the extreme repetition of a single user (500 visits) dominated the value.
        The new rvr is bounded in [0,1] and is not affected by that.

    THETA extension
        --rvr_theta changes the main definition, and --rvr_extra_thetas also emits
        ablation columns (rvr_t3, rvr_t5, ...). The computational cost is negligible.

Input (raw, before k-core and before country split)
    foursquare/raw_POIs.txt                        venueId, lat, lon, category, country
    foursquare/dataset_WWW_Checkins_anonymized.txt  userId, venueId, utcTime, tzOffset

Output
    dataset/venue_category_stats/ALL.csv    worldwide
    dataset/venue_category_stats/<CC>.csv   per country

    Columns: venueCategory, uvr, rvr, rvr_all, n_venues, n_venues_used,
             checkins, n_pairs, n_pairs_rvr   (+ rvr_t* with --rvr_extra_thetas)
    Sorting: uvr descending (categories with empty uvr go last)

Usage:
    python dataset/venue_category_stats.py
    python dataset/venue_category_stats.py --min_checkins 5
    python dataset/venue_category_stats.py --countries US,JP,KR
    python dataset/venue_category_stats.py --countries all
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset")
SRC_DIR = os.path.join(DATA, "foursquare")
POI_FILE = os.path.join(SRC_DIR, "raw_POIs.txt")
CHECKIN_FILE = os.path.join(SRC_DIR, "dataset_WWW_Checkins_anonymized.txt")
OUT_DIR = os.path.join(DATA, "venue_category_stats")

# The 11 countries for which we built per-country datasets
DEFAULT_COUNTRIES = ["US", "JP", "BR", "TR", "ID", "GB", "SG", "ES", "KR", "DE", "IT"]

# Countries included in the "ALL" scope. If None, all 232 countries worldwide.
#   build_travel.py imports this value. If it were written separately in two places,
#   a mismatch between the lists would silently apply the wrong basis, so it is defined only here.
ALL_COUNTRIES = list(DEFAULT_COUNTRIES)

# Fix for corrupted raw category names.
#   raw_POIs.txt contains no 'Café' at all; instead there are only two variants containing the replacement character (U+FFFD).
#     'Caf��'  venues 188,436 (85%)  <- é stored as UTF-8 2 bytes (C3 A9), then corrupted
#     'Caf�'        venues  33,221 (15%)  <- é stored as Latin-1 1 byte (E9), then corrupted
#   The two variants' country distributions correlate at 0.952, i.e. effectively identical, and in the
#   Foursquare taxonomy the only categories starting with 'Caf' are Café and Cafeteria (Cafeteria exists intact).
#   So both are merged into Café.
CATEGORY_FIXES = {
    "Caf�": "Café",
    "Caf��": "Café",
}

# Default threshold for rvr. A user with count(u,v) >= RVR_THETA is treated as a "revisitor".
RVR_THETA = 2

COLUMNS = ["venueCategory", "uvr", "rvr", "rvr_all", "n_venues",
           "n_venues_used", "checkins", "n_pairs", "n_pairs_rvr"]


def load_pois():
    """venueId -> (category, country). Corrupted category names are fixed here."""
    poi = pd.read_csv(
        POI_FILE, sep="\t", header=None,
        names=["venue", "lat", "lon", "cat", "country"],
        dtype={"venue": str, "cat": str, "country": str},
        quoting=3, on_bad_lines="skip", engine="c",
    )
    poi["cat"] = poi["cat"].fillna("").replace(CATEGORY_FIXES)
    fixed = int(poi["cat"].eq("Café").sum())
    print(f"  POIs {len(poi):,}  (category fix: {fixed:,} venues merged into Café)", flush=True)
    return poi


def load_pairs(checkin_file, venue_index):
    """Fold check-ins into (user, venue) pairs and return them.

    Aggregating once at the pair level lets us derive all of the following from it.
        check-ins per venue  = sum of cnt over that venue's pairs
        users per venue      = number of that venue's pairs
        category n_pairs     = number of that category's pairs
    """
    chk = pd.read_csv(
        checkin_file, sep="\t", header=None,
        names=["u", "i", "t", "tz"], usecols=[0, 1],
        dtype={"u": np.int64, "i": str},
        quoting=3, on_bad_lines="skip", engine="c",
    )
    print(f"  checkins {len(chk):,}", flush=True)
    # Convert venue strings to integer codes to reduce memory and groupby cost
    chk["v"] = venue_index.get_indexer(chk["i"])
    chk = chk.drop(columns=["i"])
    missing = int((chk["v"] < 0).sum())
    if missing:
        print(f"  excluded {missing:,} check-in rows for venues not in POIs", flush=True)
        chk = chk[chk["v"] >= 0]
    pairs = chk.groupby(["u", "v"], sort=False).size().rename("cnt").reset_index()
    print(f"  (user,venue) pairs {len(pairs):,}", flush=True)
    return pairs


def category_table(pairs, cat_of_v, min_checkins, rvr_theta=RVR_THETA,
                   extra_thetas=()):
    """Pair table -> per-category uvr / rvr / sample sizes.

    rvr is computed per venue over **the same venue set** as uvr (check-ins >= min_checkins)
    and then averaged. The old definition (rvr_all) is also emitted for comparison.
    """
    v = pairs["v"].to_numpy()
    cnt = pairs["cnt"].to_numpy()
    n_v = len(cat_of_v)

    # Venue-level aggregation
    ck_v = np.bincount(v, weights=cnt, minlength=n_v)        # check-ins per venue
    us_v = np.bincount(v, minlength=n_v)                      # users per venue (= number of pairs)
    used = us_v > 0                                           # venues appearing in this scope

    thetas = sorted({int(rvr_theta), *(int(t) for t in extra_thetas)})
    rvr_v = {t: np.bincount(v[cnt >= t], minlength=n_v) for t in thetas}

    cat = pd.Series(cat_of_v)
    df = pd.DataFrame({
        "cat": cat[used].to_numpy(),
        "ck": ck_v[used],
        "us": us_v[used],
    })
    df["uvr"] = df["us"] / df["ck"]
    for t in thetas:
        df[f"_rvr{t}"] = rvr_v[t][used] / df["us"]

    g = df.groupby("cat")
    out = pd.DataFrame({
        "n_venues": g.size(),
        "checkins": g["ck"].sum().astype(np.int64),
        "n_pairs": g["us"].sum().astype(np.int64),
    })
    # Old definition. At the venue level it is 1/uvr, so nearly redundant with uvr, but kept for comparison.
    out["rvr_all"] = out["checkins"] / out["n_pairs"]

    # uvr and rvr are averaged only over venues with check-ins >= min_checkins
    sel = df[df["ck"] >= min_checkins]
    gu = sel.groupby("cat")
    out["uvr"] = gu["uvr"].mean()
    out["n_venues_used"] = gu.size()
    out["n_venues_used"] = out["n_venues_used"].fillna(0).astype(np.int64)
    out["rvr"] = gu[f"_rvr{int(rvr_theta)}"].mean()
    for t in thetas:
        if t != int(rvr_theta):
            out[f"rvr_t{t}"] = gu[f"_rvr{t}"].mean()
    # Number of pairs with at least one revisitor. Used when a sense of rvr's sample size is needed.
    out["n_pairs_rvr"] = pd.Series(
        np.bincount(v[cnt >= int(rvr_theta)], minlength=n_v)[used],
        index=df.index).groupby(df["cat"]).sum().astype(np.int64)

    out = out.reset_index().rename(columns={"cat": "venueCategory"})
    cols = COLUMNS + [f"rvr_t{t}" for t in thetas if t != int(rvr_theta)]
    out = out[cols]
    for c in ["uvr", "rvr", "rvr_all"] + [f"rvr_t{t}" for t in thetas
                                          if t != int(rvr_theta)]:
        out[c] = out[c].round(6)
    return out.sort_values("uvr", ascending=False, na_position="last")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkin_file", type=str, default=CHECKIN_FILE,
                        help="Set to raw_Checkins_anonymized.txt to use the original 90M check-ins")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--min_checkins", type=int, default=10,
                        help="Minimum check-ins for a venue to be included in uvr/rvr computation")
    parser.add_argument("--rvr_theta", type=int, default=RVR_THETA,
                        help="Revisit criterion for rvr. A user is a revisitor if count(u,v) >= this value")
    parser.add_argument("--rvr_extra_thetas", type=str, default="",
                        help="Extra thetas for ablation. e.g. 3,5,10 -> rvr_t3/rvr_t5/rvr_t10")
    parser.add_argument("--countries", type=str, default=",".join(DEFAULT_COUNTRIES),
                        help="Comma-separated country codes, or 'all' (ALL.csv is always generated)")
    parser.add_argument("--all_countries", type=str, default=",".join(ALL_COUNTRIES),
                        help="Countries to include in ALL.csv. 'world' = the entire world.")
    args = parser.parse_args()
    t0 = time.time()

    print(f"Reading POIs: {POI_FILE}", flush=True)
    poi = load_pois()
    venue_index = pd.Index(poi["venue"])
    cat_of_v = poi["cat"].to_numpy()
    co_of_v = poi["country"].fillna("").to_numpy()
    del poi

    print(f"Reading check-ins: {args.checkin_file}", flush=True)
    pairs = load_pairs(args.checkin_file, venue_index)
    del venue_index

    # Pairs for venues without a category are excluded from all aggregations
    has_cat = cat_of_v != ""
    keep = has_cat[pairs["v"].to_numpy()]
    dropped = int((~keep).sum())
    if dropped:
        print(f"  excluded {dropped:,} pairs for venues without a category", flush=True)
    pairs = pairs[keep]

    if args.countries.strip().lower() == "all":
        codes = sorted({c for c in co_of_v if c})
    else:
        codes = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    scopes = [("ALL", None)] + [(c, c) for c in codes]
    all_cc = (None if args.all_countries.strip().lower() == "world"
              else [c.strip().upper() for c in args.all_countries.split(",") if c.strip()])
    extra = [int(t) for t in args.rvr_extra_thetas.split(",") if t.strip()]
    print(f"\nComputing {len(scopes)} scopes  (min_checkins={args.min_checkins}, "
          f"rvr_theta={args.rvr_theta}"
          + (f", extra={extra}" if extra else "") + ")", flush=True)
    print(f"  ALL scope: {'entire world' if all_cc is None else str(len(all_cc)) + ' countries ' + ','.join(all_cc)}",
          flush=True)

    pair_country = co_of_v[pairs["v"].to_numpy()]
    for name, cc in scopes:
        if cc is None:
            sub = pairs if all_cc is None else pairs[np.isin(pair_country, all_cc)]
        else:
            sub = pairs[pair_country == cc]
        if len(sub) == 0:
            print(f"  {name:>4}: no data - skipped", flush=True)
            continue
        tab = category_table(sub, cat_of_v, args.min_checkins,
                             args.rvr_theta, extra)
        path = os.path.join(args.out_dir, f"{name}.csv")
        tab.to_csv(path, index=False, encoding="utf-8-sig")
        n_uvr = int(tab["uvr"].notna().sum())
        print(f"  {name:>4}: categories {len(tab):>4} (uvr computed {n_uvr:>4}) "
              f"check-ins {int(tab['checkins'].sum()):>10,}  -> {os.path.basename(path)}",
              flush=True)

    print(f"\nDone ({time.time() - t0:.0f}s)  -> {args.out_dir}")


if __name__ == "__main__":
    main()
