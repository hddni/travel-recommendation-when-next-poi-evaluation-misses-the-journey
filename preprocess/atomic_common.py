"""Shared logic for generating RecBole atomic files.

Collected here so that every dataset uses the same k-core filter and the same file format.
Implementing it separately per dataset leads to subtle differences that make result comparisons meaningless.

.inter : user_id:token  item_id:token  [rating:float]  timestamp:float
.item  : item_id:token  venue_category_id:token  venue_category:token_seq
         latitude:float  longitude:float

For datasets without category information (gowalla), the two category columns are left empty.
"""

import os
import re

import numpy as np
import pandas as pd

ITEM_COLUMNS = [
    "item_id",
    "venue_category_id",
    "venue_category",
    "latitude",
    "longitude",
]
ITEM_HEADER = [
    "item_id:token",
    "venue_category_id:token",
    "venue_category:token_seq",
    "latitude:float",
    "longitude:float",
]

_WS = re.compile(r"\s+")


def slug_token(name):
    """Turn a category name into a single token.

    Since field_separator is tab and seq_separator is space, leaving spaces in
    splits the token_seq word by word. Replace spaces with '_' to keep it as one chunk.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = _WS.sub(" ", str(name)).strip()
    return text.replace(" ", "_") if text else ""


def slug_seq(names):
    """Pack multiple categories into a single token_seq cell (space-separated)."""
    tokens = [slug_token(n) for n in names]
    return " ".join(t for t in tokens if t)


def iterative_kcore(df, user_col, item_col, k):
    """Iterative k-core. Alternately filters until both users and items remain with at least k occurrences.

    Same logic as what dataset/foursquare/convert_to_atomic.py used for NYC/TKY.
    """
    if k is None or k <= 1:
        return df
    while True:
        if len(df) == 0:
            return df
        u_cnt = df[user_col].value_counts()
        i_cnt = df[item_col].value_counts()
        keep = (df[user_col].map(u_cnt) >= k) & (df[item_col].map(i_cnt) >= k)
        if keep.all():
            return df
        df = df[keep]


def _to_epoch(series, fmt):
    """Parse with the given format and return unix epoch (seconds).

    The source may contain corrupted rows (e.g. 1 row out of 22.8M in Foursquare
    Global), so parse with errors='coerce' to leave NaN; attach_timestamp does the dropping.
    """
    ts = pd.to_datetime(series, format=fmt, utc=True, errors="coerce")
    epoch = ts.astype("int64") // 10**9
    return epoch.where(ts.notna())


def parse_foursquare_time(series):
    """'Tue Apr 03 18:00:07 +0000 2012' -> unix epoch (seconds)."""
    return _to_epoch(series, "%a %b %d %H:%M:%S %z %Y")


def parse_iso_time(series):
    """'2010-10-19T23:55:27Z' -> unix epoch (seconds)."""
    return _to_epoch(series, "%Y-%m-%dT%H:%M:%SZ")


def parse_naive_time(series):
    """'2018-07-07 22:09:11' -> unix epoch (seconds). Treated as UTC when there is no timezone info."""
    return _to_epoch(series, "%Y-%m-%d %H:%M:%S")


def attach_timestamp(df, src_col, parser, out_col="timestamp"):
    """Parse and attach timestamps; rows that fail to parse are reported and dropped."""
    epoch = parser(df[src_col])
    bad = int(epoch.isna().sum())
    if bad:
        print(
            f"    [warn] excluded {bad:,} rows with timestamp parse failures (corrupted source)", flush=True
        )
        df = df[epoch.notna()]
        epoch = epoch.dropna()
    out = df.copy()
    out[out_col] = epoch.astype("int64")
    return out


def drop_invalid_coords(df, item_col, coords):
    """Drop interactions on items whose coordinates are missing or outside Earth's valid range.

    SNAP Gowalla contains corrupted coordinates such as latitude 402.03; leaving them
    in makes haversine distances and RoG meaningless. Must be filtered before k-core.
    """
    lat, lon = coords["latitude"], coords["longitude"]
    ok = lat.between(-90, 90) & lon.between(-180, 180)
    bad_items = set(coords.index[~ok])
    if not bad_items:
        return df
    mask = df[item_col].isin(bad_items)
    print(
        f"    [warn] excluded {len(bad_items):,} items with out-of-range coordinates / "
        f"{int(mask.sum()):,} interaction rows",
        flush=True,
    )
    return df[~mask]


def build_item_frame(item_ids, coords, categories=None):
    """Build the DataFrame for .item.

    Args:
        item_ids: array of item ids actually remaining in .inter.
        coords (pandas.DataFrame): index=item_id, columns=[latitude, longitude].
        categories (pandas.DataFrame, optional): index=item_id,
            columns=[venue_category_id, venue_category]. If None, filled with empty values.
    """
    ids = pd.Index(pd.unique(pd.Series(item_ids))).sort_values()
    item = pd.DataFrame({"item_id": ids})
    item["latitude"] = item["item_id"].map(coords["latitude"])
    item["longitude"] = item["item_id"].map(coords["longitude"])
    if categories is None:
        item["venue_category_id"] = ""
        item["venue_category"] = ""
    else:
        item["venue_category_id"] = (
            item["item_id"].map(categories["venue_category_id"]).fillna("")
        )
        item["venue_category"] = (
            item["item_id"].map(categories["venue_category"]).fillna("")
        )
    return item[ITEM_COLUMNS]


def write_dataset(out_dir, name, inter, item, inter_header):
    """Save .inter and .item as a pair."""
    os.makedirs(out_dir, exist_ok=True)
    inter_path = os.path.join(out_dir, f"{name}.inter")
    item_path = os.path.join(out_dir, f"{name}.item")
    inter.to_csv(
        inter_path, sep="\t", index=False, header=inter_header, encoding="utf-8"
    )
    item.to_csv(
        item_path, sep="\t", index=False, header=ITEM_HEADER, encoding="utf-8"
    )
    return inter_path, item_path


def report(name, inter, item, user_col="user_id", item_col="item_id", kcore=None):
    n_u = inter[user_col].nunique()
    n_i = inter[item_col].nunique()
    density = len(inter) / (n_u * n_i) * 100 if n_u and n_i else 0.0
    has_cat = int((item["venue_category_id"].astype(str) != "").sum())
    missing_xy = int(item[["latitude", "longitude"]].isna().any(axis=1).sum())
    print(
        f"[{name}] kcore={kcore}\n"
        f"    interactions : {len(inter):,}\n"
        f"    users        : {n_u:,}\n"
        f"    items        : {n_i:,}  (.item rows {len(item):,})\n"
        f"    density      : {density:.4f}%\n"
        f"    category     : present in {has_cat:,}/{len(item):,} items\n"
        f"    missing xy   : {missing_xy:,}",
        flush=True,
    )
