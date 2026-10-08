"""Vectorized ephemerides built on astropy; the ephem-based equivalents live in `ephemerides.py`."""
import astropy.units as u
import numpy as np
from astropy.time import Time


def hour_angle(ra, time, lon_deg):
    """
    Vectorized hour angle wrap_to_pi(LST - RA), with the apparent local sidereal time from astropy.
    Uses the (-pi, pi] convention of `equatorial_to_hour_angle`, without its per-coordinate ephem loop.

    Arguments
    ---------
    ra : float or array of floats
        Right ascension in radians
    time : float
        Unix timestamp (UTC)
    lon_deg : float
        Site longitude in degrees

    Returns
    -------
    hour_angle : float or array of floats
        Hour angle in radians.
    """
    lst = float(Time(time, format="unix", scale="utc").sidereal_time(
        "apparent", longitude=lon_deg * u.deg).radian)
    return (lst - np.asarray(ra, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
