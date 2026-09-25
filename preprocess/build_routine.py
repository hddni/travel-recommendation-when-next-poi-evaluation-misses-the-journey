"""Build the A' (travel-removed) dataset. It is the **exact complement** of build_travel.py.

build_travel.py (A, travel) deletes the check-ins that daily_label.label() judged as routine,
and this script (A', routine) **keeps only those.** Since both files call the same function,
before the k-core step the two check-in sets are disjoint and their union is the raw data.
This is not code duplication; **calling the same function guarantees the same criterion**.

See dataset/daily_label.py for the full judgment rules.

Note: k-core is applied independently to each. A and A' keep different surviving users/venues,
    so the user sets do not match. A user-matched comparison would require a separate arm.

Output: dataset/foursquare-<scope>-routine/
    <name>.inter  <name>.item  category_quadrants.csv  daily_ratio.csv
    recbole/properties/dataset/foursquare-<scope>-routine.yaml

Usage:
    python preprocess/build_routine.py
    python preprocess/build_routine.py --scopes DE,KR --kcore 5
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset")
sys.path.insert(0, HERE)

from atomic_common import (  # noqa: E402
    build_item_frame, drop_invalid_coords, iterative_kcore, report,
    slug_token, write_dataset,
)
from build_travel import (  # noqa: E402
    ALL_COUNTRIES, INTER_HEADER, KCORE, KCORE_OVERRIDE, PROPS_DIR, SCOPES,
    dataset_yaml, load_source, scope_subset,
)
from daily_label import (  # noqa: E402
    MIN_VENUES_USED, REL_TAU, RVR_THRESHOLD, UVR_THRESHOLD,
)
import daily_label  # noqa: E402

COLS = ["user_id", "item_id", "timestamp", "local_timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", type=str, default=",".join(SCOPES))
    ap.add_argument("--kcore", type=int, default=KCORE)
    ap.add_argument("--all_countries", type=str, default=",".join(ALL_COUNTRIES))
    ap.add_argument("--rvr", type=float, default=RVR_THRESHOLD)
    ap.add_argument("--uvr", type=float, default=UVR_THRESHOLD)
    ap.add_argument("--min_venues_used", type=int, default=MIN_VENUES_USED)
    ap.add_argument("--rel_tau", type=float, default=REL_TAU)
    ap.add_argument("--out_root", type=str, default=DATA,
                    help="Parent directory in which to create dataset folders. To split by\n"
                         "threshold combination, pass dataset/<combination name>")
    args = ap.parse_args()
    scopes = [s.strip().upper() for s in args.scopes.split(",") if s.strip()]
    all_cc = [c.strip().upper() for c in args.all_countries.split(",") if c.strip()]

    poi, chk = load_source()
    pidx = poi.set_index("venue")
    coords = pidx[["latitude", "longitude"]]
    v2cat = pidx["cat"]
    slug = v2cat.map(slug_token)
    cats = pd.DataFrame({"venue_category_id": slug, "venue_category": slug})
    del poi

    summary = []
    for scope in scopes:
        name = f"foursquare-{scope.lower()}-routine"
        k = KCORE_OVERRIDE.get(scope, args.kcore)
        sub = scope_subset(chk, scope, all_cc if scope == "ALL" else None)
        if len(sub) == 0:
            print(f"[{name}] no data for this scope - skipped", flush=True)
            continue

        print(f"\n=== {name}  ({k}-core) ===")
        params = daily_label.category_params(scope, args.rvr, args.uvr,
                                             args.min_venues_used)
        cat_col = sub["item_id"].map(v2cat)
        sub = sub[cat_col.notna() & (cat_col != "")]
        sub = drop_invalid_coords(sub, "item_id", coords)
        origin = len(sub)

        #  Same judgment as travel. Here only those judged as routine are kept.
        is_daily, st = daily_label.label(sub, v2cat, params, rel_tau=args.rel_tau,
                                         uvr_th=args.uvr)
        daily_label.report(st)
        sub = sub[is_daily]
        print(f"    cumulative: {origin:,} -> {len(sub):,} ({len(sub)/origin*100:.1f}% retained)")

        sub = sub[COLS].copy()
        sub = iterative_kcore(sub, "user_id", "item_id", k)
        if len(sub) == 0:
            print(f"    No data left after {k}-core.", flush=True)
            continue

        inter = sub.sort_values(["user_id", "timestamp"])
        item = build_item_frame(inter["item_id"].values, coords, cats)
        out_dir = os.path.join(args.out_root, name)
        write_dataset(out_dir, name, inter, item, INTER_HEADER)
        params.to_csv(os.path.join(out_dir, "category_quadrants.csv"),
                      encoding="utf-8-sig")

        #  This dataset is all routine by definition, so daily_ratio is 1.0.
        #  Recorded so that dataset_stats.py can read the per-arm routine ratio.
        pu = inter.groupby("user_id").size().rename("n_checkins").reset_index()
        pu["n_daily"] = pu["n_checkins"]
        pu["daily_ratio"] = 1.0
        pu.to_csv(os.path.join(out_dir, "daily_ratio.csv"),
                  index=False, encoding="utf-8-sig")

        os.makedirs(PROPS_DIR, exist_ok=True)
        note = (f"Foursquare {scope} - travel removed (A', complement of travel: "
                f"only routine categories + habitual pairs kept)")
        if scope == "ALL":
            note += f" / restricted to {len(all_cc)} countries"
        with open(os.path.join(PROPS_DIR, f"{name}.yaml"), "w", encoding="utf-8") as f:
            f.write(dataset_yaml(name, k, note))
        report(name, inter, item, kcore=k)
        print(f"    -> {out_dir}", flush=True)
        summary.append((name, inter["user_id"].nunique(),
                        inter["item_id"].nunique(), len(inter)))

    if summary:
        print("\n" + "=" * 76)
        print(f"  {'dataset':<34}{'users':>9}{'items':>10}{'inter':>14}")
        for n, u, i, c in summary:
            print(f"  {n:<34}{u:>9,}{i:>10,}{c:>14,}")


if __name__ == "__main__":
    main()
