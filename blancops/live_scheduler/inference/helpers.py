from blancops.configs.experiment_schema import ActionConstraints
from blancops.environment.live_env import LiveBlancoEnv

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as au

from blancops.math import units

def build_env(cfg, norm_stats, lookups, telemetry_now):
    constraints_cfg = ActionConstraints()
    env = LiveBlancoEnv(
        cfg=cfg,
        constraints_cfg=constraints_cfg,
        lookups=lookups,
        norm_stats=norm_stats,
        telemetry_init=telemetry_now
    )
    return env

CTIO = EarthLocation(lat=-30.1652778 * au.deg, lon=-70.815 * au.deg, height=2215 * au.m)
DEPLOYMENT_WINDOW = ("2026-06-24T04:47:00", "2026-06-24T10:16:00")


def get_visible_targets(
    target_radecs: np.ndarray = None,
    obs_window_utc=DEPLOYMENT_WINDOW,
    step_minutes: float = 10.0,
    airmass_limit: float = 1.3,
    require: str = "any",          # "any" -> observable at >=1 sampled time; "all" -> entire window
    site: EarthLocation = CTIO,
):
    """
    Args
    ----
    target_radecs : np.ndarray | None
        (n_fields, 2) target [ra, dec] in RADIANS. If None, a default sky grid
        (in degrees) is used.
    obs_window_utc : (str, str)
        (start, end) UTC ISO strings bounding the night window.
    step_minutes : float
        Sampling cadence across the window.
    airmass_limit : float
        Plane-parallel airmass cut (airmass = 1/sin(alt)), matching your convention.
    require : {"any","all"}
        "any" returns fields observable at some point in the window (the union you
        asked for); "all" returns fields observable for the entire window.

    Returns
    -------
    valid_ra, valid_dec : np.ndarray (deg)
    frac_observable : np.ndarray
        Fraction of sampled times each *returned* field is above the airmass limit
        (1.0 = up all window; small = only briefly up -> rising/setting edge case).
    """
    if target_radecs is None:
        ra_bins = np.linspace(0, 360, 100)
        dec_bins = np.linspace(-80, 20, 50)
        ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)
        ra_deg = ra_grid.ravel()
        dec_deg = dec_grid.ravel()
        out_shape = ra_grid.shape
    else:
        ra_deg = np.degrees(target_radecs[:, 0])   # rad -> deg, unambiguous
        dec_deg = np.degrees(target_radecs[:, 1])
        out_shape = None

    # --- sample the window ---
    t0 = Time(obs_window_utc[0], scale="utc")
    t1 = Time(obs_window_utc[1], scale="utc")
    n_steps = max(2, int(np.ceil((t1 - t0).to_value(au.min) / step_minutes)) + 1)
    times = t0 + np.linspace(0, (t1 - t0).to_value(au.min), n_steps) * au.min

    elevation_limit = np.degrees(np.arcsin(1.0 / airmass_limit))   # deg, = 50.3 for am=1.3
    alt = _alt_deg(ra_deg, dec_deg, times, loc=site)               # (n_fields, n_times)
    up = alt > elevation_limit
    frac = up.mean(axis=1)

    mask = up.any(axis=1) if require == "any" else up.all(axis=1)

    valid_ra = ra_deg[mask]
    valid_dec = dec_deg[mask]
    return valid_ra, valid_dec, frac[mask]

def _alt_deg(ra_deg, dec_deg, times, loc=CTIO):
    """Altitude (deg) of each (ra,dec) at each time -> array (n_fields, n_times)."""
    targets = SkyCoord(ra=np.atleast_1d(ra_deg) * au.deg,
                       dec=np.atleast_1d(dec_deg) * au.deg, frame="icrs")
    out = np.empty((targets.size, len(times)), dtype=float)
    for j, t in enumerate(times):
        aa = targets.transform_to(AltAz(obstime=t, location=loc))
        out[:, j] = aa.alt.to_value(au.deg)
    return out
