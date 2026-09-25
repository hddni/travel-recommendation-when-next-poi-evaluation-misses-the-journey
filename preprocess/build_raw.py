"""Build the raw (all) dataset. It is the untouched source with only k-core applied.

Why use this as the 'raw' arm
    We need a baseline to compare travel (travel only) / routine (routine only) against.
    That baseline is easiest to read if it is **the source itself** from which the two
    arms branch off. Here no routine/travel labeling is done at all; only the same
    preprocessing as the other arms (category/coordinate filters + k-core) is applied.

    **Independent of thresholds.** Since daily_label is not called, the result is the
    same however UVR/RVR are set. So it does not need to be rebuilt for each threshold
    combination, and it serves as a fixed reference point when comparing combinations.

What is kept identical to the other arms
    Same check-in source (dataset_WWW_Checkins_anonymized.txt)
    Drop venues with empty category                  <- same as travel/routine
    Drop venues with missing / out-of-range coords   <- same as travel/routine
    Same k-core (default 5, honoring KCORE_OVERRIDE)
    Same .inter / .item format and category slug rule

    k-core is applied independently per arm. Since raw filters nothing out, the most
    survives, so its user/venue sets are close to a superset of travel/routine; but
    k-core is non-monotonic, so an exact inclusion relation is not guaranteed.

Output:
    dataset/foursquare-<scope>-raw/
        <name>.inter  <name>.item
    recbole/properties/dataset/<name>.yaml

Usage:
    python dataset/build_raw.py
    python dataset/build_raw.py --scopes US,JP,DE,SG,KR
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset")
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

from atomic_common import (  # noqa: E402
    build_item_frame, drop_invalid_coords, iterative_kcore, report,
    slug_token, write_dataset,
)
from build_travel import (  # noqa: E402
    ALL_COUNTRIES, INTER_HEADER, KCORE, KCORE_OVERRIDE, PROPS_DIR, SCOPES,
    dataset_yaml, load_source, scope_subset,
)

INTER_COLS = ["user_id", "item_id", "timestamp", "local_timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", type=str, default=",".join(SCOPES))
    ap.add_argument("--kcore", type=int, default=KCORE)
    ap.add_argument("--all_countries", type=str, default=",".join(ALL_COUNTRIES),
                    help="Countries to include in the ALL scope (comma-separated). 'world' means the entire world.")
    ap.add_argument("--out_root", type=str, default=DATA,
                    help="Parent directory in which to create dataset folders. raw does not depend on thresholds,\n"
                         "so usually keep a single copy under the default (dataset/) and access it via\n"
                         "symlinks from the combination folders")
    args = ap.parse_args()
    scopes = [s.strip().upper() for s in args.scopes.split(",") if s.strip()]
    allc = (None if args.all_countries.strip().lower() == "world"
            else [c.strip().upper() for c in args.all_countries.split(",") if c.strip()])

    poi, chk = load_source()
    pi = poi.set_index("venue")
    coords = pi[["latitude", "longitude"]]
    v2cat = pi["cat"]
    #  The two category columns of .item. Uses the same slug rule as build_travel.py.
    slug = v2cat.map(slug_token)
    cats = pd.DataFrame({"venue_category_id": slug, "venue_category": slug})
    del poi

    print(f"\nNo routine/travel labeling. Only {args.kcore}-core is applied.", flush=True)

    summary = []
    for scope in scopes:
        k = KCORE_OVERRIDE.get(scope, args.kcore)
        name = f"foursquare-{scope.lower()}-raw"
        sub = scope_subset(chk, scope, allc if scope == "ALL" else None)
        origin = len(sub)
        if origin == 0:
            print(f"[{name}] no data for this scope - skipped", flush=True)
            continue

        print(f"\n=== {name}  ({k}-core) ===")
        #  Apply the same filters in the same order as travel/routine. Changing the order
        #  changes the k-core input and breaks the catalog comparison across arms.
        cat = sub["item_id"].map(v2cat)
        sub = sub[cat.notna() & (cat != "")].copy()
        after_cat = len(sub)
        sub = drop_invalid_coords(sub, "item_id", coords)
        after_xy = len(sub)
        sub = iterative_kcore(sub, "user_id", "item_id", k)
        print(f"    original {origin:,} -> category {after_cat:,} -> coordinates {after_xy:,} "
              f"-> {k}-core {len(sub):,} ({len(sub)/origin*100:.1f}% retained)", flush=True)
        if len(sub) == 0:
            print(f"[{name}] no data left after k-core - skipped", flush=True)
            continue

        inter = sub.sort_values(["user_id", "timestamp"])
        item = build_item_frame(inter["item_id"].values, coords, cats)
        out_dir = os.path.join(args.out_root, name)
        write_dataset(out_dir, name, inter[INTER_COLS], item, INTER_HEADER)
        report(name, inter, item, kcore=k)

        os.makedirs(PROPS_DIR, exist_ok=True)
        with open(os.path.join(PROPS_DIR, f"{name}.yaml"), "w", encoding="utf-8") as f:
            f.write(dataset_yaml(
                name, k,
                f"Foursquare {scope} - raw (all). No routine/travel labeling; "
                f"only {k}-core applied. Independent of thresholds."))
        print(f"    -> {out_dir}", flush=True)
        summary.append({"dataset": name, "scope": scope, "kcore": k,
                        "n_users": int(inter["user_id"].nunique()),
                        "n_items": int(inter["item_id"].nunique()),
                        "n_interactions": int(len(inter)),
                        "kept_pct": round(len(sub) / origin * 100, 2)})

    if summary:
        print(f"\n{'='*92}\nSummary")
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
