from pathlib import Path
from skybright.skybright import MoonSkyModel
from blancops.math import units
from astropy.time import Time
import numpy as np
from configparser import ConfigParser

from blancops.configs.paths import DECAM_SKY_CONFIG


def estimate_sky_brightness(time, ra, dec, band, config_path=None):
    """
    Calculate the sky brightness at a particular time, sky location, and band. Wraps the
    MoonSkyModel from the skybright package.

    Arguments
    ---------
    time : float or list-like of float
        Time (Unix timestamp, in UTC) at which to define observer.
    ra, dec : float or list-like of float
        Right ascension and declination in radians of the target position(s).
    band : str or list-like of str
        Name of filter (e.g., 'u', 'g', 'r', 'i', 'z', 'Y').
    config_path : str or Path, optional
        Path to the sky brightness configuration file. If not provided, uses the default
        'decam_sky.conf' under 'blancops/blanco/'.

    Returns
    -------
    float or np.ndarray of float
        Sky brightness in units of mag/arcsec^2.
    """
    # check for scalar inputs to preserve output shape
    scalar_input = np.all([np.ndim(x) == 0 for x in [time, ra, dec, band]])

    # parse config file
    if config_path is None:
        config_path = DECAM_SKY_CONFIG
    else:
        config_path = Path(config_path)
    model_config = ConfigParser()
    model_config.read(str(config_path))

    # initiate sky brightness model with chosen model values
    sky_model = MoonSkyModel(model_config)

    # compute sky brightness
    brightness = sky_model(
        mjd=Time(time, format="unix").mjd,
        ra_deg=np.asarray(ra) / units.deg,
        decl_deg=np.asarray(dec) / units.deg,
        band=band,
    )  # * units.mag / units.arcsec**2
    return np.squeeze(brightness) if scalar_input else brightness
