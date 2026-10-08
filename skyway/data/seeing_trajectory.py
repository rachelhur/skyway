"""Extract a validation night's measured seeing trajectory from the cached
validation dataset, for replay in an offline forward simulation.

Sibling to ``obs_history.py``: both read a saved file and produce an
env-seeding input. Here the output feeds ``OfflineBlancoEnv``'s seeing model
so a forward sim on new fields experiences a real night's seeing instead of a
flat constant. The trajectory is keyed by seconds-since-sunset so the consumer
can re-align it to each sim night's sunset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from skyway.survey.profiles import DES, SurveyProfile
from skyway.data.features.glob_features import get_night_boundaries

import logging

logger = logging.getLogger(__name__)


def _load_val_df(cache_path: Path) -> pd.DataFrame:
    """Load the split-night DataFrame from a ``<split>_dataset_cache.pt``.

    Accepts both storage forms (plain dict and ``TransitionDatasetCache`` instance) and
    both key generations: ``split_df`` as written now, and ``val_df`` as
    written before the split rename.

    Args:
        cache_path: Path to the torch-saved split dataset cache.

    Returns:
        The split-night DataFrame (all enriched columns, split nights only).
    """
    data = torch.load(cache_path, weights_only=False)
    if isinstance(data, dict):
        val_df = data.get("split_df", data.get("val_df"))
    else:
        val_df = getattr(data, "split_df", None)
        if val_df is None:
            val_df = getattr(data, "val_df", None)
    if val_df is None:
        raise KeyError(
            f"Could not find 'split_df' or 'val_df' in {cache_path}. Got a "
            f"{type(data).__name__} without either entry/attribute."
        )
    return val_df


def extract_night_seeing_trajectory(
    cache_path, val_night: str, sun_el_limit: float, survey: SurveyProfile = DES
) -> pd.DataFrame:
    """Extract one validation night's measured seeing as a replayable trajectory.

    Loads the cached validation DataFrame, selects the requested night, drops
    rows with no FWHM measurement, and expresses each measurement's time as
    seconds since that night's sunset so the consumer can re-align it onto an
    arbitrary sim night.

    Args:
        cache_path: Path to ``val_dataset_cache.pt``.
        val_night: Night key (the value in the ``night`` column) to extract.
        sun_el_limit: Sun-elevation limit (deg) defining the night, used to
            compute the night's sunset for the time offset.
        survey: Survey whose filter indices the cache uses; unknown indices
            fall back to its seeing reference band.

    Returns:
        DataFrame with columns ``sec_since_sunset`` (s), ``fwhm`` (arcsec),
        ``band`` (filter letter), ``el`` (radians), and ``timestamp`` (raw unix
        seconds, kept for debugging), sorted by ``sec_since_sunset``.

    Raises:
        KeyError: If ``val_night`` is not present in the dataset.
        ValueError: If the night has no valid (non-NaN) FWHM measurements.
    """
    cache_path = Path(cache_path)
    val_df = _load_val_df(cache_path)

    target_night = pd.to_datetime(val_night).date()
    night_dates = pd.to_datetime(val_df["night"]).dt.date
    night_mask = night_dates == target_night
    if not night_mask.any():
        sample = ", ".join(str(k) for k in pd.unique(night_dates)[:5])
        raise KeyError(
            f"val_night={val_night!r} (parsed {target_night}) not found in "
            f"{cache_path}. {night_dates.nunique()} nights available; "
            f"e.g. {sample}."
        )
    night_df = val_df[night_mask]

    fwhm_vals = night_df["fwhm"].to_numpy(dtype=float)
    filt_vals = night_df["filter_idx"].to_numpy(dtype=float)
    el_vals = night_df["el"].to_numpy(dtype=float)
    ts_vals = night_df["timestamp"].to_numpy(dtype=float)
    valid = ~(
        np.isnan(fwhm_vals) | np.isnan(filt_vals)
        | np.isnan(el_vals) | np.isnan(ts_vals)
    )
    if not valid.any():
        raise ValueError(
            f"val_night={val_night!r} has no rows with a complete (non-NaN) "
            f"fwhm/filter_idx/el/timestamp measurement."
        )

    fwhm_vals = fwhm_vals[valid]
    timestamps = ts_vals[valid]
    idx2filter, ref_band = survey.idx2filter, survey.seeing_ref_filter
    bands = [idx2filter.get(int(f), ref_band) for f in filt_vals[valid]]
    el = el_vals[valid]

    sunset_ts, _ = get_night_boundaries(val_night, sun_el_limit=sun_el_limit)

    out = pd.DataFrame(
        {
            "sec_since_sunset": timestamps - sunset_ts,
            "fwhm": fwhm_vals,
            "band": bands,
            "el": el,
            "timestamp": timestamps,
        }
    ).sort_values("sec_since_sunset", ignore_index=True)

    logger.info(
        f"Extracted {len(out)} seeing measurements for night {val_night} "
        f"(fwhm range {out['fwhm'].min():.2f}-{out['fwhm'].max():.2f} arcsec)."
    )
    return out
