"""Routine/travel labeling. Every build script uses this single place.

Metrics (produced by dataset/venue_category_stats.py)
    uvr(v) = |U_v| / ck_v                                   higher = more travel
    rvr(v) = |{u in U_v : count(u,v) >= THETA}| / |U_v|      higher = more routine (THETA=2)
    Category values are the mean of per-venue values (only venues with check-ins >= min_checkins)

The two metrics are tied by the identity uvr = 1/(1 + r(m-1)) (r=rvr, m=mean number
of visits per revisitor). corr(uvr, rvr) is very strong at -0.91 to -0.98, but uvr mixes
r and m together, whereas rvr isolates r alone. That difference is the basis for stage 2.

────────────────────────────────────────────────────────────────────────
Stage 1 - category level. If the signal is strong, all check-ins at venues of that category are decided.

    A  rvr >= 0.5  AND  uvr <= 0.25   ->  all routine
    B  rvr <  0.5  AND  uvr >  0.25   ->  all travel
    C  otherwise (the two metrics conflict) ->  to stage 2
    E  undecidable (insufficient sample in both own country and ALL) -> to stage 2

    If n_venues_used in the own-country stats is below MIN_VENUES_USED, fall back to ALL.csv.

    Observed: D(rvr<0.5 & uvr<=0.25) is empty in all 12 scopes. uvr<=0.25
    means r(m-1)>=3, and with r<0.5 this requires m>7, but that combination does not
    appear in category means. At the venue level 2.7-9.0% fall in D, so this is
    not a real phenomenon but an artifact of category averaging. It means D-type venues
    are hidden inside C, which is the reason for taking stage 2 down to the check-in level.

────────────────────────────────────────────────────────────────────────
Stage 2 - check-in ((user, venue) pair) level. Applied only to C/E.

    Separates by how repetitive a user is **compared to other users** who visited the same venue.

        LOO_mean(u,v)     = (ck_v - count(u,v)) / (|U_v| - 1)      excluding oneself
        rel_multiple(u,v) = count(u,v) / LOO_mean(u,v)
        routine  <=>  count(u,v) >= THETA  AND  rel_multiple >= REL_TAU

    Excluding oneself from the denominator matters. At a venue where one of 3 visitors came
    100 times, dividing by the mean including oneself gives a multiple of only 2.9, so it is not caught.

────────────────────────────────────────────────────────────────────────
Stage 2 fallback - venues with no comparison group (|U_v| = 1).

    The relative judgment is undefined (division by 0). In this case **the same thresholds
    as stage 1 are applied at the venue level**. With |U_v| = 1,

        uvr(v) = 1 / count(u,v),   rvr(v) = 1[count(u,v) >= THETA]

    so the stage 1 condition (rvr >= RVR_TH AND uvr <= UVR_TH) is equivalent to

        count(u,v) >= max(THETA, ceil(1 / UVR_TH))  =  4   (with defaults)

    **No new threshold is introduced.** 0.5/0.25 are used consistently across three
    levels: category stage 1, venue stage 2, and venue fallback.

    This segment is 2.6-10.7% of all check-ins, and 75% of it has count=1, so it is
    travel under any policy. What actually splits is the remainder, and changing the policy
    changes the final routine ratio by at most 4.3%p (see the sensitivity table in the README).
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset")
STATS_DIR = os.path.join(DATA, "venue_category_stats")

RVR_THRESHOLD = 0.5      # stage 1: revisitor ratio at or above this is a routine signal
UVR_THRESHOLD = 0.25     # stage 1: unique-user ratio at or below this is a routine signal
MIN_VENUES_USED = 10     # minimum sample to trust own-country stats. Below it, fall back to ALL.csv
REL_TAU = 4.0            # stage 2: threshold on relative multiple vs. other users
RVR_THETA = 2            # definition of "revisit". Must match venue_category_stats.py
USE_WHITELIST = False    # whether to apply the travel whitelist. Off by default for the reasons in the comment above

QUADRANTS = ("A", "B", "C", "E")

#  List of travel categories. **Not used by default** (USE_WHITELIST = False).
#
#  It was a safeguard from the old rvr (mean visit count) era, but with the new rvr (revisitor ratio)
#  24 of the 30 (on ALL) land in B (strong travel) by the rules alone, so the list is unnecessary.
#  Empirically, only 0-2 per scope (0.000-0.272% of check-ins) were misclassified as A
#  and needed rescuing, and for two of them the rules were actually right.
#
#      Garden (DE)  uvr 0.105 / rvr 0.884  - parks that are genuinely visited repeatedly in Germany
#      Temple (US)  uvr 0.129 / rvr 0.863  - in the US, places of worship rather than travel venues
#
#  In other words, the whitelist was overriding regional usage patterns. The remaining one (Castle)
#  is a small sample with only 653 check-ins even in ALL. Hence it is not applied by default.
#  When enabled, it only sends A-classified ones back to C (stage 2) - it does not decide them as travel.
TOURISM_WHITELIST = {
    "Museum", "Art Museum", "History Museum", "Science Museum", "Planetarium",
    "Aquarium", "Zoo", "Theme Park", "Theme Park Ride / Attraction", "Water Park",
    "Amusement Park", "Historic Site", "Monument / Landmark", "Castle", "Palace",
    "Temple", "Shrine", "Beach", "Scenic Lookout", "National Park", "Garden",
    "Sculpture Garden", "Observatory", "Hotel", "Hostel", "Resort",
    "Tourist Information Center", "Airport Gate", "Airport Terminal", "Airport Lounge",
}


def solo_min_count(uvr_th=UVR_THRESHOLD, theta=RVR_THETA):
    """Lower bound on count that is equivalent to the stage 1 condition for venues with |U_v|=1."""
    return max(int(theta), int(np.ceil(1.0 / uvr_th)))


def category_params(scope, rvr_th=RVR_THRESHOLD, uvr_th=UVR_THRESHOLD,
                    min_used=MIN_VENUES_USED, whitelist=None, verbose=True):
    """Per-category (uvr, rvr, source, quadrant). Falls back to ALL if the sample is thin."""
    own = pd.read_csv(os.path.join(STATS_DIR, f"{scope}.csv"), encoding="utf-8-sig")
    own = own.set_index("venueCategory")
    alls = pd.read_csv(os.path.join(STATS_DIR, "ALL.csv"), encoding="utf-8-sig")
    alls = alls.set_index("venueCategory")

    trust = (own["n_venues_used"] >= min_used) & own["uvr"].notna() & own["rvr"].notna()
    uvr = np.where(trust, own["uvr"], own.index.map(alls["uvr"]))
    rvr = np.where(trust, own["rvr"], own.index.map(alls["rvr"]))
    out = pd.DataFrame({"uvr": uvr, "rvr": rvr,
                        "source": np.where(trust, scope, "ALL")}, index=own.index)
    judged = out["uvr"].notna() & out["rvr"].notna()
    a = judged & (out["rvr"] >= rvr_th) & (out["uvr"] <= uvr_th)
    b = judged & (out["rvr"] < rvr_th) & (out["uvr"] > uvr_th)
    out["quadrant"] = np.where(~judged, "E", np.where(a, "A", np.where(b, "B", "C")))
    #  Whitelist (optional): only blocks those going to A (decided routine) and sends them to C (stage 2).
    #  The default is None, so it is not applied. See the USE_WHITELIST comment.
    if whitelist is True:
        whitelist = TOURISM_WHITELIST
    if whitelist:
        caught = out.index.isin(whitelist) & (out["quadrant"] == "A")
        if caught.any():
            out.loc[caught, "quadrant"] = "C"
            if verbose:
                print(f"    [whitelist] reverted {int(caught.sum())} categories "
                      f"from A -> C: {', '.join(out.index[caught][:6])}",
                      flush=True)
    return out


def label(df, v2cat, params, user_col="user_id", item_col="item_id",
          theta=RVR_THETA, rel_tau=REL_TAU, uvr_th=UVR_THRESHOLD):
    """Return is_daily (bool) per row and diagnostic stats.

    df must be the entire labeling population. Per-venue check-in and visitor counts are
    computed within df, so passing only a subset changes the comparison group for stage 2.
    """
    cat = df[item_col].map(v2cat)
    quad = cat.map(params["quadrant"]).fillna("E").to_numpy()

    # Per-venue check-in count / unique visitor count, per (user,venue) pair visit count
    ic, _ = pd.factorize(df[item_col])
    n_i = ic.max() + 1 if len(ic) else 0
    ck_v = np.bincount(ic, minlength=n_i)
    pair = df.groupby([user_col, item_col])[item_col].transform("size").to_numpy()
    first = ~df.duplicated([user_col, item_col]).to_numpy()
    us_v = np.bincount(ic[first], minlength=n_i)
    CK, US = ck_v[ic], us_v[ic]

    daily = np.zeros(len(df), bool)
    daily[quad == "A"] = True                       # stage 1 decided routine
    second = np.isin(quad, ("C", "E"))              # stage 2 targets (B is decided travel)

    # stage 2: relative multiple vs. other users
    multi = second & (US >= 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        loo = (CK - pair) / np.maximum(US - 1, 1)
        ratio = np.where(loo > 0, pair / np.maximum(loo, 1e-12), np.inf)
    rel_daily = multi & (pair >= theta) & np.isfinite(ratio) & (ratio >= rel_tau)
    daily |= rel_daily

    # stage 2 fallback: no comparison group -> stage 1 thresholds at the venue level
    smin = solo_min_count(uvr_th, theta)
    solo = second & (US == 1)
    solo_daily = solo & (pair >= smin)
    daily |= solo_daily

    stats = {
        "n_rows": len(df), "solo_min_count": smin,
        **{f"quad_{q}": int((quad == q).sum()) for q in QUADRANTS},
        "second_multi": int(multi.sum()), "second_solo": int(solo.sum()),
        "daily_stage1": int((quad == "A").sum()),
        "daily_relative": int(rel_daily.sum()), "daily_solo": int(solo_daily.sum()),
        "daily_total": int(daily.sum()),
    }
    return daily, stats


def report(stats, prefix="    "):
    s, n = stats, max(stats["n_rows"], 1)
    print(f"{prefix}stage 1  A(routine) {s['quad_A']:>10,} ({s['quad_A']/n*100:5.2f}%)  "
          f"B(travel) {s['quad_B']:>10,} ({s['quad_B']/n*100:5.2f}%)  "
          f"C(weak signal) {s['quad_C']:>9,}  E(undeterminable) {s['quad_E']:>8,}", flush=True)
    print(f"{prefix}stage 2  relative rule {s['second_multi']:>10,} -> routine {s['daily_relative']:>9,}  |  "
          f"fallback(|U_v|=1, count>={s['solo_min_count']}) {s['second_solo']:>9,} -> routine {s['daily_solo']:>8,}",
          flush=True)
    print(f"{prefix}final routine {s['daily_total']:,} / {n:,} ({s['daily_total']/n*100:.2f}%)",
          flush=True)
