"""Statistics tables for the generated datasets. Shows size and composition after applying the new labeling rules at a glance.

Usage:  python dataset/dataset_stats.py [--scopes ALL,DE,SG,KR,JP,US]
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset")
sys.path.insert(0, HERE)
SCOPES = ["ALL", "DE", "SG", "KR", "JP", "US"]


def read_atomic(d, name):
    f = os.path.join(d, f"{name}.inter")
    df = pd.read_csv(f, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", type=str, default=",".join(SCOPES))
    a = ap.parse_args()
    scopes = [s.strip().upper() for s in a.scopes.split(",") if s.strip()]

    # ── 1. Label composition (based on raw check-ins) ─────────────
    import daily_label as dl
    lab = []
    for s in scopes:
        p = dl.category_params(s, verbose=False)
        q = p["quadrant"].value_counts()
        lab.append({"scope": s, "labelable": int((p["quadrant"] != "E").sum()),
                    "A": int(q.get("A", 0)), "B": int(q.get("B", 0)),
                    "C": int(q.get("C", 0)), "E": int(q.get("E", 0)),
                    "own_stats": int((p["source"] == s).sum()),
                    "ALL_fallback": int((p["source"] == "ALL").sum())})
    print("\n[Table 1] Stage-1 category labeling (A=strong routine B=strong travel C=weak signal E=unlabelable)")
    print(pd.DataFrame(lab).to_string(index=False))

    # ── 2. travel / routine / raw ─────────────────────────────────
    rows = []
    for s in scopes:
        for arm in ("travel", "routine", "raw"):
            name = f"foursquare-{s.lower()}-{arm}"
            d = os.path.join(DATA, name)
            if not os.path.isdir(d):
                continue
            df = read_atomic(d, name)
            nu, ni, n = df.user_id.nunique(), df.item_id.nunique(), len(df)
            r = {"scope": s, "arm": arm,
                 "users": nu, "venues": ni, "checkins": n,
                 "density%": round(n / (nu * ni) * 100, 4),
                 "per_user": round(n / nu, 1), "per_venue": round(n / ni, 1)}
            dr = os.path.join(d, "daily_ratio.csv")
            if os.path.exists(dr):
                t = pd.read_csv(dr, encoding="utf-8-sig")
                r["routine%"] = round(t["n_daily"].sum() / t["n_checkins"].sum() * 100, 2)
            else:
                r["routine%"] = 0.0
            rows.append(r)
    print("\n[Table 2] travel / routine / raw  (after 5-core)")
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 3. travelset ──────────────────────────────────────────────
    ts = []
    for s in scopes:
        d = os.path.join(DATA, f"foursquare-{s.lower()}-travelset")
        if not os.path.isdir(d):
            continue
        m = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        u = pd.read_csv(os.path.join(d, "users.tsv"), sep="\t")
        vc = u["n_answer"].value_counts()
        ts.append({"scope": s, "users": m["n_users"], "venues": m["n_items"],
                   "checkins": m["n_interactions"],
                   "train": m["n_train"], "valid": m["n_valid"], "test": m["n_test"],
                   "answer_mean": m["answer_items_mean"],
                   "|T|=1": int(vc.get(1, 0)), "|T|=2": int(vc.get(2, 0)),
                   "|T|=3": int(vc.get(3, 0)),
                   "prefix_median": m["prefix_median"],
                   "revisit%": round(m["test_seen_rate"] * 100, 1),
                   "routine%": round(m["daily_ratio"] * 100, 2),
                   "retained%": round(m["n_users"] / m["raw_users"] * 100, 1),
                   "rounds": m["rounds"]})
    print("\n[Table 3] travelset (set-valued answer)")
    print(pd.DataFrame(ts).to_string(index=False))

    # ── 4. Verify travel + routine = raw ──────────────────────────
    print("\n[Table 4] travel / routine mutual exclusivity (see logs for pre-k-core figures)")
    chk = []
    for s in scopes:
        t = os.path.join(DATA, f"foursquare-{s.lower()}-travel")
        l = os.path.join(DATA, f"foursquare-{s.lower()}-routine")
        if not (os.path.isdir(t) and os.path.isdir(l)):
            continue
        dt = read_atomic(t, os.path.basename(t)); dl_ = read_atomic(l, os.path.basename(l))
        kt = set(map(tuple, dt[["user_id", "item_id", "timestamp"]].to_numpy()))
        kl = set(map(tuple, dl_[["user_id", "item_id", "timestamp"]].to_numpy()))
        chk.append({"scope": s, "travel": len(kt), "routine": len(kl),
                    "intersection": len(kt & kl),
                    "common_users": len(set(dt.user_id) & set(dl_.user_id)),
                    "common_venues": len(set(dt.item_id) & set(dl_.item_id))})
    print(pd.DataFrame(chk).to_string(index=False))


if __name__ == "__main__":
    main()
