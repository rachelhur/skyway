from __future__ import annotations

import numpy as np
import healpy as hp

from skyway.ephemerides.ephemerides import (
    equatorial_to_galactic,
)


def galactic_plane_mask(nside: int, g_lat_limit: float = 10.0) -> np.ndarray:
    """
    Boolean exclusion mask over a healpix grid.

    Returns a length-npix array; True where the pixel centre falls within
    g_lat_limit of the galactic plane.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))   # colat, lon (radians)
    ra = phi
    dec = np.pi / 2 - theta
    _, b = equatorial_to_galactic(ra, dec)
    return np.abs(np.degrees(b)) < g_lat_limit
