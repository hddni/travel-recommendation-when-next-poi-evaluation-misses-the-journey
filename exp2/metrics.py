"""EXP2 (travelset) beyond-accuracy metrics.

RecBole's evaluator cannot be used. The ground truth is a **set**, not a single item,
so FullSortEvalDataLoader's "1 user = 1 positive" assumption breaks, and the split
is specified in the split column of inter.tsv, so it cannot be reproduced via eval_args.

What is produced here (accuracy and contribution decomposition are produced by evaluate.py)

    revisitamp@k          how much the revisit bias was amplified and propagated

                            sum_u |R_k(u) ∩ H_u| / k
                          ────────────────────────────────────────────
                           sum_u |{i : 2<=i<=|S_u|, s_i ∈ {s_1..s_i-1}}| / |S_u|

                          The numerator is the fraction of already-visited venues among
                          the k recommendations; the denominator is the user's original
                          revisit tendency (fraction of check-ins within the prefix that
                          are revisits). 1.0 means the tendency as-is; greater than 1
                          means the model amplified the tendency.
                          The mean of per-user ratios (macro) diverges for users with a
                          small denominator, so it is reported as a ratio of sums (micro).

    category_coverage@k   |{cat(v) : v ∈ ∪_u R_k(u)}| / |C|
                          The denominator |C| is **the number of categories in the
                          original scope**. Using the dataset's own catalog as the
                          denominator would give a free high value to whichever side
                          filtered out more. None for corpora without categories.

    unseen_cat_rate@k     user mean of |{v ∈ R_k(u) : cat(v) ∉ cat(H_u)}| / k
                          Formerly called Category serendipity. Regardless of hits, it
                          only looks at how much it puts out "kinds of venues never visited".

    locinside@k           fraction of users whose recommendations fall inside that user's existing activity radius
    locdev@k / userrog@k  auxiliary values in km for reading the above
    loccompact@k          how tightly the k recommendations cluster with each other,
                          **independent of** the activity radius. Mean distance to the
                          k recommendations' own centroid (= RoG of the recommendation
                          list). Smaller means concentrated in one neighborhood.

    tourist_purity@k      only when --tourist is given. Fraction of recommendations that are actual travel visits.

Geometric definitions follow geo.py. The user profile is built only from the unique
venues of the **training prefix (split==0)**.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

from geo import UserGeo, centroid, from_ecef, haversine, to_ecef

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAT_DIR = os.path.join(ROOT, "dataset", "venue_category_stats")
TOURIST_DIR = os.path.join(ROOT, "dataset", "venue_tourist_label")


class BeyondAccuracy:
    """Read the dataset once and reuse it. Takes a top-k matrix and computes the metrics."""

    def __init__(self, data, ks=(5, 10, 20), tourist=False):
        self.ks = ks
        self.n_items = data.n_items                       # 0 is padding
        self.ds_size = data.n_items - 1
        self.scope = data.meta["scope"].upper()
        self.train_items = data.train_items
        self.item_coord = data.item_coord
        self.want_tourist = bool(tourist)

        items = pd.read_csv(os.path.join(data.dir, "items.tsv"), sep="\t",
                            dtype={"venue_id": str})

        #  Global popularity rank. Used by the rule baselines' fill-in and the popularity rule.
        cnt = Counter()
        for s in data.seq_item.values():
            cnt.update(s.tolist())
        self.train_count = cnt
        self.pop_rank = [i for i, _ in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))]

        #  ── Category ──────────────────────────────────────────────────
        #  Corpora without categories, like gowalla, are all empty values. Measuring as-is
        #  gives 0, which is misread as "diversity 0", so we skip computing it and report None.
        raw = items["category"].fillna("").astype(str).str.strip()
        self.has_category = bool((raw != "").any())
        if self.has_category:
            codes = {c: i for i, c in enumerate(sorted(raw.unique()))}
            self.item_cat = np.full(self.n_items, -1, dtype=np.int64)
            self.item_cat[items["iid"].values] = [codes[c] for c in raw]
            cat_path = os.path.join(CAT_DIR, f"{self.scope}.csv")
            if not os.path.exists(cat_path):
                raise SystemExit(f"Original category statistics not found: {cat_path}")
            self.orig_cat_size = len(pd.read_csv(cat_path, encoding="utf-8-sig"))
            self.ds_cat_size = int((np.unique(self.item_cat) >= 0).sum())
        else:
            self.item_cat = np.full(self.n_items, -1, dtype=np.int64)
            self.orig_cat_size = None
            self.ds_cat_size = 0

        #  ── Per-user history properties ───────────────────────────────
        #  Denominator p_u of revisitamp, and the history category set of unseen_cat_rate.
        self.user_revisit_hist = {}
        self.user_cats = {}
        for u, seq in data.seq_item.items():
            seen, rev = set(), 0
            for it in seq.tolist():
                if it in seen:
                    rev += 1
                else:
                    seen.add(it)
            self.user_revisit_hist[u] = rev / max(len(seq), 1)
            c = self.item_cat[seq]
            self.user_cats[u] = set(c[c >= 0].tolist())

        self.geo = UserGeo(data)
        self.tourist = None
        if self.want_tourist:
            self._load_tourist(data, items)

    # ------------------------------------------------------------------
    def _load_tourist(self, data, items):
        """Three-tier travel labels. Population P = raw check-ins - valid/test.

        1) If v is in user u's prefix  -> follow the record of that (u,v) pair
             0 if the category is routine or it is in pair.tsv, else 1
        2) Otherwise  -> the venue's overall classification. 0 if the category is routine,
             else 1[frac_tourist(v) > 0.5]   (user-level majority vote)
        3) If not observed even in the raw population P -> undecidable. Excluded from the denominator.
        """
        vpath = os.path.join(TOURIST_DIR, f"{self.scope}.venue.tsv")
        ppath = os.path.join(TOURIST_DIR, f"{self.scope}.pair.tsv")
        if not os.path.exists(vpath):
            raise SystemExit(f"Travel label table not found: {vpath}\n"
                             f"  python dataset/build_tourist_label.py "
                             f"--scopes {self.scope}")

        vdf = pd.read_csv(vpath, sep="\t", dtype={"venue_id": str})
        frac = dict(zip(vdf["venue_id"], vdf["frac_tourist"]))
        self.item_tourist_frac = np.full(self.n_items, np.nan)
        self.item_tourist_frac[items["iid"].values] = \
            [frac.get(v, np.nan) for v in items["venue_id"]]

        #  The category-level routine classification is already carried by items.tsv (same removal_set).
        self.item_cat_daily = np.zeros(self.n_items, dtype=bool)
        self.item_cat_daily[items["iid"].values] = items["is_daily"].astype(bool).values
        chk = dict(zip(vdf["venue_id"], vdf["is_cat_daily"]))
        mism = sum(1 for v, d in zip(items["venue_id"], items["is_daily"])
                   if v in chk and int(chk[v]) != int(d))
        if mism:
            print(f"  [warn] {mism:,} venues whose category-level routine classification disagrees with items.tsv. "
                  f"Check the thresholds in build_tourist_label.py and build_travelset.py.",
                  flush=True)

        #  Tier-1 exception: (u, v) pairs whose category is travel but which are routine for that user.
        users = pd.read_csv(os.path.join(data.dir, "users.tsv"), sep="\t",
                            dtype={"user_token": str})
        u2i = dict(zip(users["user_token"], users["uid"]))
        v2i = dict(zip(items["venue_id"], items["iid"]))
        self.pair_daily = set()
        if os.path.exists(ppath):
            pdf = pd.read_csv(ppath, sep="\t",
                              dtype={"user_token": str, "venue_id": str})
            for a, b in zip(pdf["user_token"], pdf["venue_id"]):
                if a in u2i and b in v2i:
                    self.pair_daily.add((int(u2i[a]), int(v2i[b])))
        self.tourist = True

    def tourist_labels(self, top, users):
        """(N, kmax) hard / soft travel labels. Undecidable is nan."""
        cat_d = self.item_cat_daily[top]
        frac = self.item_tourist_frac[top]
        with np.errstate(invalid="ignore"):
            hard = np.where(cat_d, 0.0, np.where(frac > 0.5, 1.0, 0.0))
        soft = np.where(cat_d, 0.0, frac)
        undef = np.isnan(frac) & ~cat_d
        hard[undef] = np.nan
        soft[undef] = np.nan

        #  Tier 1 — for venues in the user's prefix, the record of that pair takes precedence.
        for r, u in enumerate(users):
            hist = self.train_items.get(u)
            if not hist:
                continue
            for j, v in enumerate(top[r]):
                v = int(v)
                if v in hist:
                    t = 0.0 if (cat_d[r, j] or (u, v) in self.pair_daily) else 1.0
                    hard[r, j] = soft[r, j] = t
        return hard, soft

    def _tourist_at(self, hard, soft, k):
        h, s = hard[:, :k], soft[:, :k]
        undef = np.isnan(h)
        alive = ~np.all(undef, axis=1)          # exclude users that are entirely undecidable
        if not alive.any():
            return {f"tourist_purity@{k}": float("nan"),
                    f"tourist_purity_soft@{k}": float("nan"),
                    f"tourist_undecided@{k}": 1.0}
        with np.errstate(invalid="ignore"):
            pu = np.nanmean(h[alive], axis=1)
            ps = np.nanmean(s[alive], axis=1)
        return {
            f"tourist_purity@{k}": float(pu.mean()),
            f"tourist_purity_soft@{k}": float(ps.mean()),
            f"tourist_undecided@{k}": float(undef.mean()),
        }

    # ------------------------------------------------------------------
    def rec_metrics(self, top, users):
        """Metrics that look only at the recommendation list. They do not look at the ground truth, so there is only one set regardless of group."""
        out = {"n_eval_users": int(top.shape[0])}

        #  Is it a kind of venue never visited? Venues with unknown category (-1) are counted as
        #  "a kind never visited". Corpora with no categories at all are handled wholesale as None below.
        cat_top = self.item_cat[top]
        newcat = np.zeros(top.shape, dtype=bool)
        for r, u in enumerate(users):
            hc = self.user_cats.get(u, set())
            for j in range(top.shape[1]):
                c = int(cat_top[r, j])
                newcat[r, j] = (c < 0) or (c not in hc)

        dist, mad = self._user_distances(top, users)
        compact = self._compactness(top)
        labels = self.tourist_labels(top, users) if self.tourist else None

        for k in self.ks:
            if self.has_category:
                c = cat_top[:, :k]
                n_cat = float(np.unique(c[c >= 0]).shape[0])
                out[f"category_coverage@{k}"] = n_cat / self.orig_cat_size
                out[f"unseen_cat_rate@{k}"] = float(newcat[:, :k].mean())
            else:
                out[f"category_coverage@{k}"] = None
                out[f"unseen_cat_rate@{k}"] = None
            out.update(self._geo_at(dist, mad, k))
            out[f"loccompact@{k}"] = float(np.nanmean(compact[k]))
            if labels is not None:
                out.update(self._tourist_at(*labels, k))
        return out

    # ------------------------------------------------------------------
    def revisit_amp(self, top, users):
        """RevisitAmp@k — how much the revisit bias was amplified and propagated.

            q_u = |R_k(u) ∩ H_u| / k          fraction of already-visited among recommendations
            p_u = fraction of check-ins within the prefix that are revisits   (user's original tendency)

        RevisitAmp@k = sum_u q_u / sum_u p_u.
        As auxiliaries, the means of q and p are also reported (needed to read where the ratio comes from).
        """
        out = {}
        for k in self.ks:
            q, p = [], []
            for r, u in enumerate(users):
                hist = self.train_items.get(u, set())
                q.append(float(np.mean([int(int(v) in hist) for v in top[r, :k]])))
                p.append(self.user_revisit_hist.get(u, 0.0))
            if not q:
                continue
            sq, sp = float(np.sum(q)), float(np.sum(p))
            out[f"revisitamp@{k}"] = sq / sp if sp else float("nan")
            out[f"revisit_rec@{k}"] = float(np.mean(q))
            out[f"revisit_hist@{k}"] = float(np.mean(p))
        return out

    # ------------------------------------------------------------------
    def _user_distances(self, top, users):
        """(N, kmax) haversine distance to the user centroid + user RoG."""
        ua = np.asarray(users, dtype=np.int64)
        lat = self.item_coord[top][:, :, 0]
        lon = self.item_coord[top][:, :, 1]
        g = self.geo
        d = haversine(lat, lon, g.c_lat[ua][:, None], g.c_lon[ua][:, None])
        return d, g.mad[ua]

    def _geo_at(self, dist, mad, k):
        """locinside@k and auxiliary values in km for reading it."""
        ok = np.isfinite(mad)
        if not ok.any():
            return {f"locinside@{k}": float("nan"), f"locdev@{k}": float("nan"),
                    f"userrog@{k}": float("nan")}
        dev = dist[:, :k].mean(axis=1)[ok]      # mean distance of the k recommendations
        exc = dev - mad[ok]                     # excess over the radius (negative if inside)
        return {
            f"locinside@{k}": float((exc < 0).mean()),
            f"locdev@{k}": float(dev.mean()),
            f"userrog@{k}": float(mad[ok].mean()),
        }

    def _compactness(self, top):
        """{k: (N,) array} RoG (km) of the top-k recommendations.

        Looks at how tightly the k recommendations cluster with each other, **independent of**
        the user's activity radius. Takes the spherical centroid of the k recommendations
        themselves and uses the mean distance to it.

        Users number in the 70k range, so a per-user loop is infeasible. For each k, all users
        are computed at once (only the three k values in self.ks are needed).
        """
        lat = self.item_coord[top][:, :, 0]
        lon = self.item_coord[top][:, :, 1]
        out = {}
        for k in self.ks:
            la, lo = lat[:, :k], lon[:, :k]
            m = to_ecef(la, lo).mean(axis=1)                 # (N, 3)
            norm = np.linalg.norm(m, axis=1)
            ca, co = from_ecef(m)
            #  Degenerate case where exactly antipodal points cancel out. Use the first point as the centroid.
            bad = ~np.isfinite(norm) | (norm < 1e-12)
            if bad.any():
                ca = np.where(bad, la[:, 0], ca)
                co = np.where(bad, lo[:, 0], co)
            out[k] = haversine(la, lo, ca[:, None], co[:, None]).mean(axis=1)
        return out

    # ------------------------------------------------------------------
    def reference(self, data, phase="test"):
        """Baseline computing the same metrics on the ground-truth set T itself.

        A yardstick is needed to read the model values. Without knowing where the ground truth
        actually is and how spread out it is, one cannot say whether locinside 0.4 is high or low.

        |T| varies from 1 to 3, so padding into a matrix puts extra weight on the last ground-truth
        item of short users. Average per user over only their own |T|.
        """
        ans = data.answers[phase]
        users = [u for u in data.eval_users if u in ans and len(ans[u])]
        if not users:
            return {}
        kmax = max(len(ans[u]) for u in users)
        top = np.zeros((len(users), kmax), dtype=np.int64)
        n_gt = np.zeros(len(users), np.int64)
        for r, u in enumerate(users):
            a = ans[u]
            top[r, :len(a)] = a
            n_gt[r] = len(a)
        pad = np.arange(kmax)[None, :] >= n_gt[:, None]      # padding slots

        out = {"n_eval_users": len(users), "gt_size_mean": float(n_gt.mean())}

        #  Revisit fraction in the ground truth = test_revisit_rate. A property of the dataset.
        #  revisitamp@gt is the same fraction divided by the user's original revisit tendency.
        #  Measured with the same yardstick as the model's revisitamp@k, it serves as the baseline
        #  "how repetitive is the ground truth itself relative to the tendency" (near 1 means as-is).
        rv, uc, pu = [], [], []
        for r, u in enumerate(users):
            h = self.train_items.get(u, set())
            hc = self.user_cats.get(u, set())
            a = [int(x) for x in top[r, :n_gt[r]]]
            rv.append(np.mean([int(x in h) for x in a]))
            pu.append(self.user_revisit_hist.get(u, 0.0))
            if self.has_category:
                cc = self.item_cat[np.array(a)]
                uc.append(np.mean([(int(c) < 0) or (int(c) not in hc) for c in cc]))
        out["test_revisit_rate"] = float(np.mean(rv))
        sp = float(np.sum(pu))
        out["revisitamp@gt"] = float(np.sum(rv)) / sp if sp else float("nan")
        out["revisit_hist@gt"] = float(np.mean(pu))
        out["unseen_cat_rate@gt"] = float(np.mean(uc)) if uc else None

        if self.has_category:
            c = self.item_cat[top]
            c = np.where(pad, -1, c)
            out["category_coverage@gt"] = float(
                np.unique(c[c >= 0]).shape[0]) / self.orig_cat_size
        else:
            out["category_coverage@gt"] = None

        dist, mad = self._user_distances(top, users)
        d = np.where(pad, np.nan, dist)
        ok = np.isfinite(mad)
        if ok.any():
            with np.errstate(invalid="ignore"):
                dev = np.nanmean(d, axis=1)[ok]
            exc = dev - mad[ok]
            out["locinside@gt"] = float((exc < 0).mean())
            out["locdev@gt"] = float(dev.mean())
            out["userrog@gt"] = float(mad[ok].mean())

        #  Clustering within the ground-truth set. |T| differs per user, so each is measured with its own length.
        comp = []
        for r, u in enumerate(users):
            k = int(n_gt[r])
            if k < 2:
                comp.append(0.0)
                continue
            la = self.item_coord[top[r, :k], 0]
            lo = self.item_coord[top[r, :k], 1]
            ca, co = centroid(la, lo)
            comp.append(float(haversine(la, lo, ca, co).mean()))
        out["loccompact@gt"] = float(np.mean(comp))

        if self.tourist:
            hard, soft = self.tourist_labels(top, users)
            hard[pad] = np.nan
            soft[pad] = np.nan
            alive = ~np.all(np.isnan(hard), axis=1)
            with np.errstate(invalid="ignore"):
                out["tourist_purity@gt"] = float(
                    np.nanmean(hard[alive], axis=1).mean())
                out["tourist_purity_soft@gt"] = float(
                    np.nanmean(soft[alive], axis=1).mean())
        return out
