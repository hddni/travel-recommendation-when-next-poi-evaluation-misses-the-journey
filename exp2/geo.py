"""Spherical geometry for location metrics.

Taking the centroid as the arithmetic mean of lat/lon breaks for users crossing the meridian
and in the ALL scope (users spanning continents). Average in unit spherical coordinates (ECEF),
then convert back to lat/lon. All distances are haversine (km).

The user profile is built only from the **training prefix (split==0)**, and the basis is
**unique venues**. How many times the same venue was visited is not counted. Measuring with
check-in weighting lets frequently revisited venues pull the centroid toward themselves, so the
activity radius comes out narrower than the user's actual range of activity.

    c[u]   = centroid of the unique venues that appear in the prefix
    mad[u] = mean distance from those venues to the centroid   ("user's activity radius RoG")
"""

import numpy as np

R_KM = 6371.0088


def to_ecef(lat, lon):
    """lat/lon (degrees) -> unit spherical xyz. (..., 3)"""
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)


def from_ecef(xyz):
    """xyz (normalization not needed) -> (lat, lon) in degrees."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))


def haversine(lat1, lon1, lat2, lon2):
    """km. Broadcastable."""
    a1, o1 = np.radians(np.asarray(lat1, np.float64)), np.radians(np.asarray(lon1, np.float64))
    a2, o2 = np.radians(np.asarray(lat2, np.float64)), np.radians(np.asarray(lon2, np.float64))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def centroid(lat, lon):
    """Spherical mean position. Input order and duplicates are up to the caller."""
    m = to_ecef(lat, lon).mean(axis=0)
    if not np.isfinite(m).all() or np.allclose(m, 0.0):
        #  Degenerate case where exactly antipodal points cancel out. Use the first point (does not occur for even 1 user).
        return float(np.asarray(lat)[0]), float(np.asarray(lon)[0])
    la, lo = from_ecef(m)
    return float(la), float(lo)


class UserGeo:
    """Per-user centroid and activity radius (mean distance based on unique venues). Index 0 is padding."""

    def __init__(self, data):
        n = data.n_users
        coord = data.item_coord
        self.c_lat = np.full(n, np.nan)
        self.c_lon = np.full(n, np.nan)
        self.mad = np.full(n, np.nan)
        self.n_uniq = np.zeros(n, np.int64)

        for u, seq in data.seq_item.items():
            uq = coord[np.unique(seq)]                      # unique venues
            self.n_uniq[u] = uq.shape[0]
            lats, lons = uq[:, 0], uq[:, 1]
            a, o = centroid(lats, lons)
            self.c_lat[u], self.c_lon[u] = a, o
            self.mad[u] = float(haversine(lats, lons, a, o).mean())

    def valid(self, u):
        return np.isfinite(self.c_lat[u])
