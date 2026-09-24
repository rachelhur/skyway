"""DES data-quality calibration on top of the paper-exact teff_projection model.

blancops.data_quality.teff_projection implements Neilsen et al. (2019) with no
free parameters. This module holds everything that had to be measured against
the DES exposure table instead: the meaning and validity range of the 'qc_'
columns, the band dependence of the fiducial seeing, and the nightly re-anchoring
of the skybright model. Constants are fits to the DES exposure table unless they
cite Morganson et al. (2018).

Column mapping for ``d = load_and_process_historic_data(workspace().des_fits)``:

    d['el']       -> elevation in radians, feeds teff_projection.airmass()
    d['filter']   -> band, feeds every per-band coefficient lookup
    d['qc_fwhm']  -> delivered PSF FWHM in arcsec, feeds seeing_factor()
    d['qc_sky']   -> sky surface brightness excess in mag, feeds sky_factor()
    d['qc_cloud'] -> atmospheric attenuation in mag, feeds measured_attenuation()
    d['teff']     -> DES-DM value, reproduced by teff_from_dataframe()

Fitting the three exponents of eq 12 freely over 77873 DES wide-survey exposures
returns -2.0 for the seeing, -0.40 for the sky, and -0.80 for the transparency
above the qc_cloud threshold, matching the paper. Two things do not come from
the paper and are needed to reproduce the DES-DM 'teff' column: the qc_cloud
threshold, and a fiducial seeing that varies by band. Both are stated in
Morganson et al. (2018), section 4.7.
"""

import numpy as np
import pandas as pd

from blancops.data_quality import teff as tp
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.math import units

__all__ = [
    "fiducial_fwhm",
    "seeing_factor",
    "measured_attenuation",
    "modelled_sky_excess",
    "sky_anchor_offset",
    "valid_qc",
    "teff_from_dataframe",
]

# Physical bounds outside which a data-quality column is a failed measurement
# rather than a bad night. A delivered FWHM at or below the instrumental floor
# is a failed measurement, having nothing left after removing eps_i in
# quadrature. From season 2015 on, qc_sky tracks the DES-DM teff column as a
# true delta-magnitude out to 3 mag (measured -0.480, -0.673, -0.855 dex against
# a predicted -0.468, -0.660, -0.832 in the 1.0-1.5, 1.5-2.0 and 2.0-3.0 bins),
# so the ceiling only has to reject nonsense.
_QC_FWHM_LIMITS = (tp._INSTRUMENT_FWHM, 5.0 * units.arcsec)
_QC_SKY_LIMITS = (-1.0, 3.0)
_QC_CLOUD_LIMITS = (-1.0, 5.0)

# First night on which qc_sky is on its final scale. The 37 nights from
# 2013-08-31 to 2013-10-30 report qc_sky up to 78 mag, overstating the sky by as
# much as 17 mag against their own teff column; every night from 2013-11-05 on is
# self-consistent, and there is no anomalous night anywhere later in the survey.
# The switch falls inside the 2013-10-31 to 2013-11-04 observing gap. No bound on
# the value separates the two populations, so the early epoch has to be excluded
# by date.
_QC_SKY_EPOCH_START = "2013-11-01"

# Threshold below which DES-DM treats an exposure as fully transparent, in mag.
# qc_cloud is the robust median offset of First Cut stellar magnitudes against
# APASS (g, r) or 2MASS J via NOMAD (i, z, Y), and Morganson et al. (2018) eq 5
# converts it to a transmission factor F_trans = 1 for qc_cloud <= 0.2 and
# 10 ** (-0.8 * (qc_cloud - 0.2)) above. Fitting a hinge to the DES-DM teff column
# recovers the break at 0.200 +- 0.002 in every band and season. The clear-sky
# mode of the column itself sits at 0.11-0.14 mag with a scatter of ~0.1 mag,
# so the threshold is not the photometric reading of the column.
_QC_CLOUD_THRESHOLD = 0.20

# Band dependence of the fiducial seeing, from Morganson et al. (2018) table 4,
# expressed as each band's FWHM_fid over the i-band 0.9". Table 4 has no u band,
# so it uses the Kolmogorov factor (lambda_u / lambda_i) ** (-1/5)
# from blancops.data_quality.seeing
_FWHM_BAND_FACTORS = {
    "u": 1 / 0.86603,
    "g": 1.103,
    "r": 1.041,
    "i": 1.0,
    "z": 0.965,
    "Y": 0.950,
}


def fiducial_fwhm(band, fwhm_ref=tp._FWHM_REF):
    """
    Reference seeing scaled per band, following Morganson et al. (2018) table 4.

    Args:
        band: Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').
        fwhm_ref: Reference seeing in i band (in radians).

    Returns:
        float or np.ndarray of float: Reference seeing (in radians).
    """
    return fwhm_ref * tp._per_band(_FWHM_BAND_FACTORS, band)


def seeing_factor(qc_fwhm, band, fwhm_ref=tp._FWHM_REF):
    """
    Seeing contribution to tau using the band-dependent fiducial seeing.

    Args:
        fwhm: Delivered PSF FWHM (in radians), as from 'qc_fwhm'.
        band: Name of filter(s) ('u', 'g', 'r', 'i', 'z', or 'Y').
        fwhm_ref: Reference seeing in i band (in radians).

    Returns:
        float or np.ndarray of float: Seeing factor, dimensionless.
    """
    return (fiducial_fwhm(band, fwhm_ref) / np.asarray(qc_fwhm, dtype=float)) ** 2


def measured_attenuation(qc_cloud, threshold=_QC_CLOUD_THRESHOLD):
    """
    Convert a measured 'qc_cloud' reading into the attenuation in magnitudes
    that DES-DM applies to teff.

    Implements Morganson et al. (2018) eq 5, which counts readings at or below
    the threshold as fully transparent:

        attenuation = max(qc_cloud - threshold, 0)

    Args:
        qc_cloud: Raw 'qc_cloud' reading (in mag).
        threshold: Reading below which DES-DM sets the transmission to one
            (in mag).

    Returns:
        float or np.ndarray of float: Attenuation (in mag), zero at or below the
        threshold, ready for teff.cloud_transmission().
    """
    return np.clip(np.asarray(qc_cloud, dtype=float) - threshold, 0.0, None)


def modelled_sky_excess(time, ra, dec, band, config_path=None):
    """
    Modelled counterpart of the 'qc_sky' column: the skybright surface brightness
    expressed as an excess over the band's fiducial dark sky, in magnitudes.

    Args:
        time: Time (Unix timestamp, in UTC) of the observation.
        ra: Right ascension of the pointing (in radians).
        dec: Declination of the pointing (in radians).
        band: Observed filter ('u', 'g', 'r', 'i', 'z', or 'Y').
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        float or np.ndarray of float: Sky surface brightness relative to the
        fiducial dark sky (in mag), positive for a brighter sky.
    """
    brightness = estimate_sky_brightness(time, ra, dec, band, config_path)
    return -2.5 * np.log10(tp.sky_factor_from_brightness(brightness, band, config_path))


def sky_anchor_offset(time, ra, dec, band, sky_excess, config_path=None):
    """
    Difference between a measured sky excess and the modelled one for the same
    exposure, for re-anchoring later predictions on the same night.

    Appendix D.1 lists what the skybright model leaves out: zodiacal light, light
    pollution, and any decline of airglow over the night. Its residual against
    'qc_sky' is correspondingly dominated by an offset that drifts slowly through
    a night, and measuring that offset on exposures already taken removes most of
    it. Over 72283 consecutive DES exposure pairs the offset from the immediately
    preceding exposure cuts the sky residual from 0.25 to 0.16 mag, and flattens
    both the twilight bias (-0.13 dex in tau at a solar elevation of -12 deg) and
    the season-to-season drift to under 0.01 dex.

    Args:
        time: Time (Unix timestamp, in UTC) of the reference observation.
        ra: Right ascension of the reference pointing (in radians).
        dec: Declination of the reference pointing (in radians).
        band: Filter of the reference observation.
        sky_excess: Measured sky excess of the reference observation (in mag), as
            from 'qc_sky'.
        config_path: Path to the sky brightness configuration file. If not
            provided, uses the default 'decam_sky.conf'.

    Returns:
        float or np.ndarray of float: Offset to add to modelled_sky_excess()
        (in mag). Take a median over several reference exposures to damp the
        per-exposure noise.
    """
    modelled = modelled_sky_excess(time, ra, dec, band, config_path)
    return np.asarray(sky_excess, dtype=float) - modelled


def valid_qc(d, exclude_early_epoch=True):
    """
    Flag rows whose data-quality columns hold usable measurements.

    Args:
        d: Processed exposure table carrying 'qc_fwhm', 'qc_sky', and 'qc_cloud',
            and optionally 'night'.
        exclude_early_epoch: Also reject the 37 nights before 2013-11-01, whose
            qc_sky column is on an inconsistent scale. Requires a 'night' column;
            ignored if absent.

    Returns:
        np.ndarray of bool: True where every data-quality column is finite and
        within physical bounds, one per row.
    """
    fwhm = d["qc_fwhm"].to_numpy(dtype=float) * units.arcsec
    sky = d["qc_sky"].to_numpy(dtype=float)
    cloud = d["qc_cloud"].to_numpy(dtype=float)

    finite = np.isfinite(fwhm) & np.isfinite(sky) & np.isfinite(cloud)
    within = (
        (fwhm >= _QC_FWHM_LIMITS[0])
        & (fwhm <= _QC_FWHM_LIMITS[1])
        & (sky >= _QC_SKY_LIMITS[0])
        & (sky <= _QC_SKY_LIMITS[1])
        & (cloud >= _QC_CLOUD_LIMITS[0])
        & (cloud <= _QC_CLOUD_LIMITS[1])
    )
    if exclude_early_epoch and "night" in d:
        within &= pd.to_datetime(d["night"]).to_numpy() >= np.datetime64(
            _QC_SKY_EPOCH_START
        )
    return finite & within


def teff_from_dataframe(d, fwhm_ref=tp._FWHM_REF, band_scaled=True, mask_invalid=True):
    """
    Evaluate tau on the measured data-quality columns of a processed exposure
    table, for comparison against its 'teff' column.

    Uses 'qc_cloud' for the transparency, 'qc_fwhm' for the seeing, and 'qc_sky'
    for the sky, so no factor is modelled. 'qc_cloud' measures the total
    atmospheric attenuation against a photometric reference and so stands in for
    the whole of eta: no separate eq-16 extinction term is added on top of it.

    Args:
        d: Output of blancops.data.preprocessing.load_and_process_historic_data,
            carrying 'filter', 'qc_fwhm', 'qc_sky', and 'qc_cloud'.
        fwhm_ref: Reference seeing in i band (in radians).
        band_scaled: Use the band-dependent fiducial seeing fitted to the DES-DM
            teff column. Set False for the paper's single reference seeing, which
            leaves a per-band offset of up to 15%.
        mask_invalid: Return NaN for rows failing valid_qc() rather than
            propagating a failed measurement into tau.

    Returns:
        np.ndarray of float: Exposure time scaling factor tau, one per row.
    """
    fwhm = d["qc_fwhm"].to_numpy(dtype=float) * units.arcsec
    band = d["filter"].to_numpy()
    factor = (
        seeing_factor(fwhm, band, fwhm_ref)
        if band_scaled
        else tp.seeing_factor(fwhm, fwhm_ref)
    )
    eta = tp.cloud_transmission(measured_attenuation(d["qc_cloud"].to_numpy(dtype=float)))
    tau = eta**2 * factor * tp.sky_factor(d["qc_sky"].to_numpy(dtype=float))
    return np.where(valid_qc(d), tau, np.nan) if mask_invalid else tau
