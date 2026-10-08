"""Effective exposure time (teff) projection following Neilsen et al. (2019).

    tau = eta ** 2 * (eps_o / fwhm) ** 2 * (b_dark / b)

where eta is the atmospheric transparency, fwhm is the delivered PSF FWHM,
eps_o is the reference seeing at which tau is unity, and b_dark / b is the ratio
of the fiducial dark-sky surface brightness to the actual sky surface
brightness.

Each factor has its own function below. Everything here follows the paper directly.

Quantities calibrated against the DES exposure table instead live in
blancops.data_quality.teff_des_qc, which builds on this module.

Equations implemented:
    A.2  eq 12  tau from transparency, seeing, and sky brightness
    A.3  eq 16  eta = 10 ** (-0.4 * k * (X - 1))
    A.3  eq 17  b_dark / b for a moonless sky, van Rhijn airglow plus extinction
    A.3  eq 18  fwhm ** 2 = eps_i ** 2 + (eps_zenith * mu ** (-3/5)) ** 2
    A.3  eq 24  the three combined, as predict_teff() evaluates them
    B    eq 28  Cassini airmass for a uniform spherical shell atmosphere
    D.2  eq 63  van Rhijn airglow surface brightness against zenith distance
"""

from configparser import ConfigParser
from pathlib import Path

import numpy as np

from blancops.configs.paths import DECAM_SKY_CONFIG
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.ephemerides.ephemerides import equatorial_to_topographic
from blancops.math import units

__all__ = [
    "airmass",
    "extinction_coefficient",
    "extinction_transmission",
    "cloud_transmission",
    "delivered_fwhm",
    "seeing_factor",
    "airglow_ratio",
    "sky_factor",
    "sky_factor_from_brightness",
    "combine_teff",
    "predict_teff",
]

# Ratio of the Earth's radius to the height of the Cassini uniform-shell
# atmosphere, appendix B eq 27. The paper adopts 470km, which matches measured
# airmass to ~1% out to a zenith distance of 70 deg and is never > 20%
# even at the horizon.
_CASSINI_A = 470.0 # R_Earth / h_C

# Radius of the Earth and height of the airglow emitting layer used by the van
# Rhijn model of eq 17. Appendix A.3 puts the airglow layer at 90 km.
_EARTH_RADIUS = 6375.0
_AIRGLOW_HEIGHT = 90.0

# Reference seeing eps_o
_FWHM_REF = 0.9 * units.arcsec

# Instrumental contribution eps_i to the delivered PSF. Fig 7 caption.
# Note: blancops.data_quality.seeing defaults to a
# more conservative 0.5".
_INSTRUMENT_FWHM = 0.45 * units.arcsec


def _read_sky_config(key, config_path=None):
    """
    Read one per-band parameter row out of the sky brightness configuration.

    Args:
        key: Name of the parameter row ('k', 'm_zen', 'h', ...).
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        dict of str to float: Parameter value keyed by band name.
    """
    config_path = DECAM_SKY_CONFIG if config_path is None else Path(config_path)
    config = ConfigParser()
    config.read(str(config_path))

    filters = config.get("sky", "filters").split()
    values = [float(x) for x in config.get("sky", key).split()]
    return dict(zip(filters, values))


def _per_band(table, band):
    """
    Look up a per-band coefficient for scalar or array-like band inputs.

    Args:
        table: Coefficient keyed by band name.
        band: Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').

    Returns:
        float or np.ndarray of float: Coefficient(s) matching the shape of band.
    """
    values = np.asarray([table[b] for b in np.atleast_1d(band)], dtype=float)
    return values.item() if np.ndim(band) == 0 else values


# ======================================================================================
# geometry
# ======================================================================================


def airmass(el, a=_CASSINI_A):
    """
    Airmass of a pointing under the Cassini uniform spherical shell model.

    Appendix B eq 28:

        X = sqrt(a ** 2 * mu ** 2 + 2 * a + 1) - a * mu,    mu = cos(z) = sin(el)

    At the horizon this gives sqrt(2 * a + 1), or 30.7 for the default a,
    against a measured value of roughly 38.

    Args:
        el: Elevation of the pointing (in radians).
        a: Ratio of the Earth's radius to the height of the modelled atmosphere.
            If None, falls back to the flat-slab approximation X = sec(z).

    Returns:
        float or np.ndarray of float: Airmass, dimensionless.
    """
    mu = np.sin(el)
    if a is None:
        return 1.0 / mu
    return np.sqrt(a**2 * mu**2 + 2 * a + 1) - a * mu


# ======================================================================================
# component 1: atmospheric transparency, eta
# ======================================================================================


def extinction_coefficient(band, config_path=None):
    """
    Per-band atmospheric extinction coefficient k, in mag per airmass.

    Args:
        band: Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        float or np.ndarray of float: Extinction coefficient(s), mag per airmass.
    """
    return _per_band(_read_sky_config("k", config_path), band)


def extinction_transmission(el, band, a=_CASSINI_A, config_path=None):
    """
    Atmospheric transparency from clean-air extinction, appendix A.3 eq 16:

        eta = 10 ** (-0.4 * k * (X - 1))

    Normalized to 1 at zenith, so this is the transparency relative to a clear
    zenith pointing rather than an absolute above-atmosphere transmission.

    Args:
        el: Elevation of the pointing (in radians).
        band: Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').
        a: Cassini shell parameter passed through to airmass().
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        float or np.ndarray of float: Transparency, normalized to 1 at zenith.
    """
    k = extinction_coefficient(band, config_path)
    return 10 ** (-0.4 * k * (airmass(el, a=a) - 1.0))


def cloud_transmission(attenuation):
    """
    Atmospheric transparency from an attenuation expressed in magnitudes:

        eta = 10 ** (-0.4 * attenuation)

    Generalizes eq 16, whose attenuation is k * (X - 1), to the clouds that
    appendix A.1 folds into eta alongside clean-air extinction.

    Args:
        attenuation: Atmospheric attenuation (in mag). Zero for a clear zenith
            pointing.

    Returns:
        float or np.ndarray of float: Transparency, normalized to 1 at zero
        attenuation.
    """
    return 10 ** (-0.4 * np.asarray(attenuation, dtype=float))


# ======================================================================================
# component 2: delivered PSF FWHM
# ======================================================================================


def delivered_fwhm(zenith_seeing, el, instrument_fwhm=_INSTRUMENT_FWHM):
    """
    Delivered PSF FWHM from an atmospheric seeing at zenith, appendix A.3 eq 18:

        fwhm ** 2 = eps_i ** 2 + (eps_zenith * mu ** (-3/5)) ** 2

    The mu ** (-3/5) factor is the Kolmogorov airmass scaling of eq 29. The
    zenith seeing is assumed to already be in the target band; use
    blancops.data_quality.seeing.convert_seeing to move a measurement between
    bands via the lambda ** (-1/5) relation of eq 30.

    Args:
        zenith_seeing: Atmospheric contribution to the FWHM at zenith,
            eps_zenith (in radians).
        el: Elevation of the pointing (in radians).
        instrument_fwhm: Instrumental contribution eps_i, summed in quadrature
            (in radians).

    Returns:
        float or np.ndarray of float: Delivered PSF FWHM (in radians).
    """
    mu = np.sin(el)
    return np.hypot(instrument_fwhm, np.asarray(zenith_seeing) * mu ** (-3 / 5))


def seeing_factor(fwhm, fwhm_ref=_FWHM_REF):
    """
    Seeing contribution to tau, the middle factor of eq 12:

        (eps_o / fwhm) ** 2

    Args:
        fwhm: Delivered PSF FWHM including the instrumental contribution
            (in radians).
        fwhm_ref: Reference seeing eps_o (in radians).

    Returns:
        float or np.ndarray of float: Seeing factor, equal to 1 at the reference
        seeing.
    """
    return (fwhm_ref / np.asarray(fwhm, dtype=float)) ** 2


# ======================================================================================
# component 3: sky brightness ratio, b_dark / b
# ======================================================================================


def airglow_ratio(
    el,
    band,
    height=_AIRGLOW_HEIGHT,
    radius=_EARTH_RADIUS,
    a=_CASSINI_A,
    config_path=None,
):
    """
    Sky brightness ratio for a moonless sky, appendix A.3 eq 17:

        b_dark / b = sqrt((h + R * mu ** 2) / (h + R)) * 10 ** (0.4 * k * (X - 1))

    The square root is the van Rhijn (1921) model of eq 63, in which the airglow
    is emitted by a thin shell at height h so the emitting path lengthens away
    from zenith; the power of ten is the extinction of that airglow, which
    partially offsets it. This is airglow only, and so is the appropriate sky
    term when the moon is down and there is no twilight. Use
    sky_factor_from_brightness() with the skybright model otherwise.

    Args:
        el: Elevation of the pointing (in radians).
        band: Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').
        height: Height of the airglow layer (in km).
        radius: Radius of the Earth (in km).
        a: Cassini shell parameter passed through to airmass().
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        float or np.ndarray of float: Sky brightness ratio b_dark / b, equal to 1
        at zenith.
    """
    mu = np.sin(el)
    k = extinction_coefficient(band, config_path)
    van_rhijn = np.sqrt((height + radius * mu**2) / (height + radius))
    return van_rhijn * 10 ** (0.4 * k * (airmass(el, a=a) - 1.0))


def sky_factor(sky_excess):
    """
    Sky brightness ratio from a surface brightness excess in magnitudes:

        b_dark / b = 10 ** (-0.4 * delta_m)

    Follows from b proportional to 10 ** (-0.4 * m) for the b_dark / b of eq 12.

    Args:
        sky_excess: Sky surface brightness relative to the fiducial dark sky
            (in mag), positive for a brighter sky.

    Returns:
        float or np.ndarray of float: Sky brightness ratio b_dark / b, equal to 1
        at the fiducial dark sky.
    """
    return 10 ** (-0.4 * np.asarray(sky_excess, dtype=float))


def sky_factor_from_brightness(sky_brightness, band, config_path=None):
    """
    Sky brightness ratio from an absolute surface brightness in mag/arcsec^2,
    referenced to the zenith airglow 'm_zen' of the sky configuration:

        b_dark / b = 10 ** (-0.4 * (m_zen - m_sky))

    Appendix E defines b_dark as the sky flux in moonless conditions at zenith,
    which is what m_zen holds. Pair this with the surface brightness returned by
    blancops.data_quality.sky_brightness, which adds scattered moonlight and
    twilight to the airglow.

    Args:
        sky_brightness: Sky surface brightness (in mag/arcsec^2).
        band: Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        float or np.ndarray of float: Sky brightness ratio b_dark / b.
    """
    m_zen = _per_band(_read_sky_config("m_zen", config_path), band)
    return 10 ** (-0.4 * (m_zen - np.asarray(sky_brightness, dtype=float)))


# ======================================================================================
# combination and prediction
# ======================================================================================


def combine_teff(transmission, fwhm, sky_ratio, fwhm_ref=_FWHM_REF):
    """
    Combine the three exposure-quality factors into tau, appendix A.2 eq 12:

        tau = eta ** 2 * (eps_o / fwhm) ** 2 * (b_dark / b)

    tau is an exposure time scaling factor: an exposure of length t taken under
    these conditions reaches the limiting magnitude of a reference exposure of
    length tau * t.

    Args:
        transmission: Atmospheric transparency eta, from cloud_transmission() or
            extinction_transmission().
        fwhm: Delivered PSF FWHM (in radians).
        sky_ratio: Sky brightness ratio b_dark / b, from sky_factor(),
            sky_factor_from_brightness(), or airglow_ratio().
        fwhm_ref: Reference seeing eps_o (in radians).

    Returns:
        float or np.ndarray of float: Exposure time scaling factor tau.
    """
    eta = np.asarray(transmission, dtype=float)
    return eta**2 * seeing_factor(fwhm, fwhm_ref) * np.asarray(sky_ratio)


def predict_teff(
    time,
    ra,
    dec,
    band,
    el=None,
    fwhm=None,
    zenith_seeing=None,
    sky_brightness=None,
    sky_excess=None,
    cloud=0.0,
    attenuation=None,
    instrument_fwhm=_INSTRUMENT_FWHM,
    fwhm_ref=_FWHM_REF,
    a=_CASSINI_A,
    config_path=None,
):
    """
    Evaluate tau for a pointing at a given time.

    With no overrides this is a cold prediction: elevation from the pointing and
    time, sky brightness from the skybright model, delivered FWHM from eq 18, and
    transparency from eq 16. Supplying 'fwhm', 'sky_excess', 'sky_brightness', or
    'attenuation' evaluates the same expression on measured conditions instead.

    Args:
        time: Time (Unix timestamp, in UTC) of the observation.
        ra: Right ascension of the pointing (in radians).
        dec: Declination of the pointing (in radians).
        band: Observed filter ('u', 'g', 'r', 'i', 'z', or 'Y').
        el: Elevation of the pointing (in radians). Computed from time, ra, and
            dec if not provided; note that the fallback resolves one observer
            time, so pass this explicitly for array input.
        fwhm: Measured delivered PSF FWHM (in radians). Overrides the eq 18
            model.
        zenith_seeing: Atmospheric seeing at zenith in the observed band
            (in radians), used by the eq 18 model. Required unless fwhm is given.
        sky_brightness: Sky surface brightness (in mag/arcsec^2). Computed from
            the skybright model if neither this nor sky_excess is provided.
        sky_excess: Sky surface brightness relative to the fiducial dark sky
            (in mag). Takes precedence over sky_brightness.
        cloud: Cloud-only attenuation (in mag), added to the modelled clean-air
            extinction k * (X - 1) of eq 16. Ignored if attenuation is given.
        attenuation: Total atmospheric attenuation (in mag), replacing the
            modelled k * (X - 1) + cloud rather than adding to it.
        instrument_fwhm: Instrumental contribution eps_i (in radians).
        fwhm_ref: Reference seeing eps_o (in radians).
        a: Cassini shell parameter passed through to airmass().
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        float or np.ndarray of float: Exposure time scaling factor tau.
    """
    if el is None:
        _, el = equatorial_to_topographic(ra, dec, time=time)

    if fwhm is None:
        if zenith_seeing is None:
            raise ValueError("predict_teff requires either 'fwhm' or 'zenith_seeing'.")
        fwhm = delivered_fwhm(zenith_seeing, el, instrument_fwhm=instrument_fwhm)

    if sky_excess is not None:
        sky_ratio = sky_factor(sky_excess)
    else:
        if sky_brightness is None:
            sky_brightness = estimate_sky_brightness(time, ra, dec, band, config_path)
        sky_ratio = sky_factor_from_brightness(sky_brightness, band, config_path)

    if attenuation is None:
        k = extinction_coefficient(band, config_path)
        attenuation = k * (airmass(el, a=a) - 1.0) + np.asarray(cloud, dtype=float)
    transmission = cloud_transmission(attenuation)

    return combine_teff(transmission, fwhm, sky_ratio, fwhm_ref=fwhm_ref)
