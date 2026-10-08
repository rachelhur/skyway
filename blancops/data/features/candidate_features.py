"""Per-candidate positional features shared by the candidate pipelines (bin and field-level candidates).

A candidate grid is any object with ``lon``, ``lat``, ``is_azel`` and the HealpixGrid ephemeris methods:
a ``HealpixGrid`` for bin action spaces, a ``FieldGrid`` for field_filter. The helpers here evaluate at
the grid's points without knowing which kind of candidate they are.
"""
import warnings

import numpy as np

from blancops.ephemerides import ephemerides

# Marks inactive candidates during intermediate computation; normalization converts it to the sentinel.
_INTERNAL_SENTINEL = np.nan


def compute_candidate_ephemeris_features(timestamp, pointing_radec, grid, night_duration_in_sec):
    """Per-timestep ephemeris features for every point on a candidate grid.

    ``delta_az``/``delta_el`` are always in true topographic coordinates regardless of ``grid.is_azel``.
    ``t_until_set`` = time until the point sets / night duration, NaN when it never sets.

    Parameters
    ----------
    timestamp : float
        Unix timestamp (UTC).
    pointing_radec : tuple
        Current telescope (RA, Dec) in radians.
    grid : HealpixGrid
        Candidate grid (HealpixGrid or FieldGrid).
    night_duration_in_sec : float
        Night duration used to scale t_until_set.

    Returns
    -------
    dict
        Keys ``ra``, ``dec``, ``az``, ``el``, ``ha``, ``airmass``, ``moon_distance``, ``sun_distance``,
        ``pointing_distance``, ``delta_az``, ``delta_el``, ``t_until_set``; each an (n_candidates,) array.
    """
    features = {}
    lon, lat = grid.lon, grid.lat

    if grid.is_azel:
        ra, dec = ephemerides.topographic_to_equatorial(
            az=lon, el=lat, time=timestamp
        )
        features['az'], features['el'] = lon, lat
        features['ra'], features['dec'] = ra, dec
        pointing_az, pointing_el = ephemerides.equatorial_to_topographic(
            ra=pointing_radec[0], dec=pointing_radec[1], time=timestamp
        )
        pointing_in_grid = (pointing_az, pointing_el)
    else:
        az, el = ephemerides.equatorial_to_topographic(
            ra=lon, dec=lat, time=timestamp
        )
        features['ra'], features['dec'] = lon, lat
        features['az'], features['el'] = az, el
        pointing_az, pointing_el = ephemerides.equatorial_to_topographic(
            ra=pointing_radec[0], dec=pointing_radec[1], time=timestamp
        )
        pointing_in_grid = pointing_radec

    features['ha'] = grid.get_hour_angle(time=timestamp)
    features['airmass'] = grid.get_airmass(timestamp)
    features['moon_distance'] = grid.get_source_angular_separations(
        'moon', time=timestamp
    )
    features['sun_distance'] = grid.get_source_angular_separations(
        'sun', time=timestamp
    )
    features['pointing_distance'] = grid.get_angular_separations(
        lon=pointing_in_grid[0], lat=pointing_in_grid[1]
    )
    features['delta_az'], features['delta_el'] = get_delta_az_el(
        features['az'], features['el'], pointing_az, pointing_el
    )
    t_until_set_raw = grid.get_time_until_set(time=timestamp)
    # The above method outputs np.inf. Convert to NaN.
    # This will be handled by StateNormalizer at the end of its pipeline.
    features['t_until_set'] = np.where(
        np.isfinite(t_until_set_raw),
        t_until_set_raw / night_duration_in_sec,
        _INTERNAL_SENTINEL
    )
    return features


def get_relative_feature(feat_arr, el_mask):
    """Subtract the per-timestep mean over above-horizon candidates: x_rel = x - mean(x[el > 0]).

    NaN values are excluded from the mean and stay NaN in the output.

    Parameters
    ----------
    feat_arr : np.ndarray
        Feature values, candidates on the last axis.
    el_mask : np.ndarray
        True for candidates above the horizon.

    Returns
    -------
    np.ndarray
        Relative feature values, same shape as ``feat_arr``.
    """
    valid_cols = np.where(el_mask, feat_arr, np.nan)
    with warnings.catch_warnings():
        # nanmean over an all-NaN slice warns and returns NaN, which is intended.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local_mean = np.nanmean(valid_cols, axis=-1, keepdims=True)
    return feat_arr - local_mean


def get_delta_az_el(cand_azs, cand_els, target_az, target_el):
    """Angular differences to a target; az is wrapped to [-pi, pi), el is a plain difference.

    Parameters
    ----------
    cand_azs, cand_els : np.ndarray
        Candidate azimuths and elevations in radians.
    target_az, target_el : float
        Target azimuth and elevation in radians.

    Returns
    -------
    tuple of np.ndarray
        (delta_az, delta_el) per candidate.
    """
    azs = (cand_azs - target_az + np.pi) % (2 * np.pi) - np.pi
    els = cand_els - target_el
    return azs, els
