import numpy as np

from skyway.ephemerides.ephemerides import (
    galactic_to_equatorial,
)

def galactic_plane_boundaries(g_lat_limit: float = 10.0, n: int = 2000) -> dict:
    """
    Equatorial coordinates of the galactic-plane boundary curves, for plotting.

    Returns a dict with keys top_/bottom_/plane_ (the +g_lat_limit, -g_lat_limit and
    g_lat=0 curves), each suffixed _ra / _dec, in radians.
    """
    glon = np.linspace(0.0, 2 * np.pi, n)
    out: dict[str, np.ndarray] = {}
    for key, b_deg in (("top", g_lat_limit), ("bottom", -g_lat_limit), ("plane", 0.0)):
        b = np.full_like(glon, np.radians(b_deg))
        ra, dec = galactic_to_equatorial(glon, b)
        out[f"{key}_ra"] = np.asarray(ra)
        out[f"{key}_dec"] = np.asarray(dec)
    return out


