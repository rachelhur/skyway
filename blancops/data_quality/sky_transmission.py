import numpy as np
from pathlib import Path
from blancops.configs.paths import DECAM_SKY_CONFIG
from blancops.math import units
from configparser import ConfigParser


def estimate_transmission(el, band, config_path=None):
    """
    Estimate the atmospheric transmission at a given elevation and band. Assumes the
    magnitude of atmospheric extinction is linear with airmass X and an extinction
    coefficient k such that transmission eta = 10 ** (-2 / 5 * k * (X - 1)). For more
    info, see https://doi.org/10.2172/1574836 Appendix A.3.

    Arguments
    ---------
    el : float or list-like of float
        Elevation(s) (in radians) at which to estimate transmission.
    band : str or list-like of str
        Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').
    config_path : str or Path, optional
        Path to the sky brightness configuration file. If not provided, uses the default
        paths.DECAM_SKY_CONFIG (blancops/blanco/decam_sky.conf).

    Returns
    -------
    float or np.ndarray of float
        Estimated transmission value(s).
    """
    # check for scalar inputs to preserve output shape
    scalar_input = np.all([np.ndim(x) == 0 for x in [el, band]])

    # parse config file containing extinction coefficients
    if config_path is None:
        config_path = DECAM_SKY_CONFIG
    else:
        config_path = Path(config_path)
    config = ConfigParser()
    config.read(str(config_path))

    # get extinction coefficients as a function of band
    filters = config.get("sky", "filters").split()
    k = [float(x) for x in config.get("sky", "k").split()]
    k = dict(zip(filters, k))

    # get extinction coefficients for requested bands
    k_vals = np.asarray([k[band] for band in np.atleast_1d(band)])

    # compute airmass for input elevations
    X = 1.0 / np.sin(el)

    # compute transmission
    transmission = 10 ** (-2 / 5 * k_vals * (X - 1))
    return np.squeeze(transmission) if scalar_input else transmission


def convert_transmission(
    transmission,
    to_band,
    to_el,
    from_band="i",
    from_el=90 * units.deg,
    config_path=None,
):
    """
    Convert transmission from one band and elevation to another. Assumes the magnitude
    of atmospheric extinction is linear with airmass X and an extinction coefficient k
    such that transmission eta = 10 ** (-2 / 5 * k * (X - 1)). For more info, see
    https://doi.org/10.2172/1574836 Appendix A.3.

    Arguments
    ---------
    transmission : float or list-like of float
        Measured transmission value to convert.
    to_band : str or list-like of str
        Target band for conversion ('u', 'g', 'r', 'i', 'z', or 'Y').
    to_el : float or list-like of float
        Target elevation for conversion (in radians).
    from_band : str or list-like of str ["i"]
        Original band of the measurements ('u', 'g', 'r', 'i', 'z', or 'Y').
    from_el : float or list-like of float [90 * units.deg]
        Original elevation of the measurements (in radians). Default is zenith.
    config_path : str or Path, optional
        Path to the sky brightness configuration file. If not provided, uses the default
        paths.DECAM_SKY_CONFIG (blancops/blanco/decam_sky.conf).

    Returns
    -------
    float or list-like of float
        Converted transmission value.
    """
    # check for scalar inputs to preserve output shape
    scalar_input = np.all(
        [np.ndim(x) == 0 for x in [transmission, to_el, to_band, from_el, from_band]]
    )

    # parse config file containing extinction coefficients
    if config_path is None:
        config_path = DECAM_SKY_CONFIG
    else:
        config_path = Path(config_path)
    config = ConfigParser()
    config.read(str(config_path))

    # get extinction coefficients as a function of band
    filters = config.get("sky", "filters").split()
    k = [float(x) for x in config.get("sky", "k").split()]
    k = dict(zip(filters, k))

    # get extinction coefficients for requested bands
    from_k = np.asarray([k[band] for band in np.atleast_1d(from_band)])
    to_k = np.asarray([k[band] for band in np.atleast_1d(to_band)])

    # compute airmass for input and output elevations
    from_X = 1.0 / np.sin(from_el)
    to_X = 1.0 / np.sin(to_el)

    # convert transmission
    converted = np.asarray(transmission) * 10 ** (
        -2 / 5 * (to_k * (to_X - 1) - from_k * (from_X - 1))
    )
    return np.squeeze(converted) if scalar_input else converted
