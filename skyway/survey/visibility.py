from __future__ import annotations

from typing import Literal

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from skyway.telescope import get_telescope


def visible_fields(
    field_radec_rad: np.ndarray,                 # (n, 2) [ra, dec] in radians
    window_utc: tuple[str, str],                 # (start, end) ISO UTC strings
    site: EarthLocation = None,                  # default: blanco profile site
    airmass_limit: float = None,                 # default: blanco max_airmass
    step_minutes: float = 10.0,
    require: Literal["any", "all"] = "any",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter fields by airmass over a time window.

    Returns
    -------
    mask : (n,) bool
        require="any" -> field clears the airmass cut at >=1 sampled time;
        require="all" -> field clears it at every sampled time.
    frac_observable : (n,) float
        Fraction of sampled times each field is above the airmass cut.
    """
    telescope = get_telescope("blanco")
    if site is None:
        site = telescope.site.earth_location()
    if airmass_limit is None:
        airmass_limit = telescope.constraints.max_airmass

    ra_deg = np.degrees(field_radec_rad[:, 0])
    dec_deg = np.degrees(field_radec_rad[:, 1])

    t0 = Time(window_utc[0], scale="utc")
    t1 = Time(window_utc[1], scale="utc")
    span_min = (t1 - t0).to_value(u.min)
    n_steps = max(2, int(np.ceil(span_min / step_minutes)) + 1)
    times = t0 + np.linspace(0.0, span_min, n_steps) * u.min

    # Plane-parallel airmass X = 1/sin(alt) -> elevation floor.
    elevation_limit = np.degrees(np.arcsin(1.0 / airmass_limit))

    targets = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    alt = np.empty((targets.size, len(times)), dtype=float)   # (n_fields, n_times)
    for j, t in enumerate(times):
        aa = targets.transform_to(AltAz(obstime=t, location=site))
        alt[:, j] = aa.alt.to_value(u.deg)

    up = alt > elevation_limit
    frac = up.mean(axis=1)
    mask = up.any(axis=1) if require == "any" else up.all(axis=1)
    return mask, frac
