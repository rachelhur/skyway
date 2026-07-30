
"""Bin feature computation for both online (live) and offline (batch) pipelines.
 
The module-level helpers below define the canonical per-timestep computation.
Both `BaseBlancoEnv._calculate_bin_features` (live, 1 timestep) and
`BinFeatureEngineer.transform` (offline, batch) drive these primitives — the
offline pipeline writing per-step outputs into pre-allocated arrays for
cache-friendliness.
 
Sentinel convention: inactive bins are marked with NaN internally during all
intermediate computation (rel, cyclical, validate). `replace_nan_with_sentinel`
converts NaN -> RADEC/AZEL_BIN_FEAT_SENTINEL (-1.0) at the very end, so
downstream consumers see the on-disk convention.

To add a new feature:
    1. Implement in one of the appropriate canonical helpers:
        - `compute_bin_ephemeris_features`
        - `compute_bin_progress_features`
        - `apply_relative_bin_features`
        - 'get_relative_feature'
    or add a new helper
    2. Add feature key to `BinFeatureEngineer._pre_allocate_arrays()`
    3. Update _ALLOWED_NORMS_PER_FEATURE constant in configs/constants.py 
    4. Add feature key to _BIN_FEATURE_NAMES in configs/constants.py
    5. Add feature key and default normalization to _DEFAULT_NORM_MAPPING in configs/constants.py
    
    If adding a new helper, must also update blancops/environment/base.py `_calculate_bin_features`
    method and `BinFeatureEngineer.transform` method.
"""
import warnings
 
import numpy as np
from einops import rearrange
from tqdm import tqdm
from blancops.data.features.glob_features import get_night_boundaries
from blancops.data.features.normalizations import apply_cyclical_features
from blancops.ephemerides import ephemerides
from blancops.configs.constants import *
 
import logging

from blancops.survey.profiles import DES
logger = logging.getLogger(__name__)
 
 
# History-feature base names that this pipeline knows about. Any
# requested feature whose name contains one of these substrings will
# trigger the history pass.
_SURVEY_PROGRESS_BASE_KEYS = [
        'num_unvisited_fields',
        'num_incomplete_fields',
        'min_tiling',
    ]
_STALENESS_BASE_KEYS = ['t_since_last_visit']
_INTERNAL_SENTINEL = np.nan

# Sun-elevation horizon (degrees) is used to define night duration in seconds
# Currently defined to match the value used in 
# `data/preprocessing `build_DES_lookups()` so that night2ot_clock_seconds 
# is consistent. This param is used by features:
# `t_until_set` and *can* be used by `t_since_last_visit` (not advised).
_SUN_EL_LIMIT_DEG = DES.sun_el_limit

 
# ============================================================================
# Canonical per-timestep helpers — single source of truth, shared between
# BinFeatureEngineer (offline batch) and BaseBlancoEnv (live single-step).
# ============================================================================
 
 
def compute_bin_ephemeris_features(timestamp, pointing_radec, hpGrid, night_duration_in_sec):
    """Per-timestep ephemeris features for all bins on the hpGrid.
 
    Returns a dict with keys: ``ra``, ``dec``, ``az``, ``el``, ``ha``,
    ``airmass``, ``moon_distance``, ``sun_distance``, ``pointing_distance``,
    ``delta_az``, ``delta_el``, ``time_till_set`` — each a length-``nbins`` array.
 
    ``pointing_radec`` is ``(ra, dec)`` in radians. ``delta_az``/``delta_el``
    are always in true topographic coords regardless of ``hpGrid.is_azel``;
    this fixes a subtle bug in the previous offline pipeline that passed
    ``(ra, dec)`` to ``get_delta_az_el`` when the grid was RaDec, producing
    delta_ra / delta_dec mislabeled as delta_az / delta_el.
    """
    features = {}
    lon, lat = hpGrid.lon, hpGrid.lat
 
    if hpGrid.is_azel:
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
 
    features['ha'] = hpGrid.get_hour_angle(time=timestamp)
    features['airmass'] = hpGrid.get_airmass(timestamp)
    features['moon_distance'] = hpGrid.get_source_angular_separations(
        'moon', time=timestamp
    )
    features['sun_distance'] = hpGrid.get_source_angular_separations(
        'sun', time=timestamp
    )
    features['pointing_distance'] = hpGrid.get_angular_separations(
        lon=pointing_in_grid[0], lat=pointing_in_grid[1]
    )
    features['delta_az'], features['delta_el'] = get_delta_az_el(
        features['az'], features['el'], pointing_az, pointing_el
    )
    t_until_set_raw = hpGrid.get_time_until_set(time=timestamp)
    # The above method outputs np.inf. Convert to NaN.
    # This will be handled by StateNormalizer at the end of its pipeline. 
    features['t_until_set'] = np.where(
        np.isfinite(t_until_set_raw), 
        t_until_set_raw / night_duration_in_sec,
        _INTERNAL_SENTINEL
        
    )
    return features
 
 
def compute_bin_progress_features(
    current_counts,
    target_counts,
    bins_per_field,
    v_mask,
    nbins,
    do_filt,
    idx2filter=None,
    timestamp=None,
    last_visit_timestamps=None,
    t_since_last_visit_divisor=None
):
    """Per-timestep survey-progress features per bin.
 
    Inactive bins (no in-plan fields contributing) are marked with NaN.
    The caller is responsible for converting NaN -> external sentinel via
    ``replace_nan_with_sentinel`` at the end of the pipeline.
 
    Parameters
    ----------
    current_counts : np.ndarray
        ``(nfields,)`` for non-filter or ``(nfields, nfilters)`` for filter
        mode; current survey-wide visit counts.
    target_counts : np.ndarray
        Same shape as ``current_counts``; survey targets.
    bins_per_field : np.ndarray
        ``(nfields,)`` int; bin index of each field at this time. May contain
        ``ZENITH_BIN_NUM`` for invalid mappings.
    v_mask : np.ndarray
        ``(nfields,)`` bool; fields to include (above-horizon AND in a valid
        bin).
    nbins : int
        Total number of bins on the hpGrid.
    do_filt : bool
        True iff filter is part of the action space; controls whether
        per-filter outputs are produced.
    idx2filter : dict, optional
        Filter idx -> name mapping. Defaults to ``IDX2FILTER``.
    timestamp : float, optional
        Current time, same units as ``last_visit_timestamps``. Required
        together with ``last_visit_timestamps`` to emit the staleness
        (``t_since_last_visit``) features; ignored otherwise.
    last_visit_timestamps : np.ndarray, optional
        Per-field last-visit times, ``(nfields,)`` or ``(nfields, nfilters)``;
        NaN for never-visited. In non-filter mode a 2D array is collapsed to
        per-field via ``np.nanmax`` over filters. When omitted, no staleness
        features are produced.
    t_since_last_visit_divisor : float, optional
        Divisor applied to each age ``timestamp - last_visit`` to normalize
        staleness (typically the night's total OT span). Defaults to 1 (no
        normalization).

 
    Returns
    -------
    dict
        Keys ``num_unvisited_fields``, ``num_incomplete_fields``, and
        ``min_tiling`` (always), plus per-filter variants when ``do_filt``.
        Also ``t_since_last_visit`` (or per-filter ``t_since_last_visit_<filter>``)
        when ``timestamp`` and ``last_visit_timestamps`` are supplied.
 
    Note on normalization: the "adjusted max" used in ratios is
    ``max(current_at_this_step, target)``, matching the live env. This differs
    from the legacy offline computation which precomputed
    ``max(night_total, target)`` once per night; the new behavior keeps the
    normalization "what you see is what you get" from the agent's perspective.
    """
    if t_since_last_visit_divisor is None:
        t_since_last_visit_divisor = 1
            
    if idx2filter is None:
        idx2filter = IDX2FILTER
 
    if do_filt and current_counts.ndim != 2:
        raise ValueError(
            f"do_filt=True requires 2D current_counts; got shape "
            f"{current_counts.shape}"
        )
    if current_counts.shape != target_counts.shape:
        raise ValueError(
            f"current_counts shape {current_counts.shape} != target_counts "
            f"shape {target_counts.shape}"
        )
    
    features = {}
    bins_mem = bins_per_field[v_mask].astype(np.int32)
 
    # Aggregate per-field counts (sum over filters for the 1D family of features)
    if current_counts.ndim == 2:
        cur_field_vis = current_counts.sum(axis=1)
        tgt_field_vis = target_counts.sum(axis=1)
    else:
        cur_field_vis = current_counts
        tgt_field_vis = target_counts
 
    v_cur_field = cur_field_vis[v_mask]
    v_tgt_field = tgt_field_vis[v_mask]
    in_plan = v_tgt_field > 0
    max_adj = np.maximum(v_cur_field, v_tgt_field)
 
    # Active bins: bins that contain at least one in-plan field
    bin_in_plan_counts = np.bincount(
        bins_mem, weights=in_plan, minlength=nbins
    )
    act_s = bin_in_plan_counts > 0
 
    def _assign_staleness(last_visit_per_field, in_plan_mask, key):
        """Per-bin staleness = freshest in-plan field's age, OT-normalized if provided.

        Single division: the OT delta is divided by the total OT span
        once, inside the ``np.where``. The reduction (``np.minimum.at``)
        picks the freshest (smallest age) in-plan field per bin. NaN
        last-visit values become +inf age so they don't pull the min
        down; bins with no contributing in-plan field stay +inf and get
        converted to the internal NaN sentinel.
        """
        age_v = np.where(
            in_plan_mask & ~np.isnan(last_visit_per_field),
            np.maximum(0.0, (timestamp - last_visit_per_field) / t_since_last_visit_divisor),
            np.inf, # XXX the np.max() is for the case in live env rollout where timestamp < field's last visit
        )
        res = np.full(nbins, np.inf, dtype=np.float32)
        np.minimum.at(res, bins_mem, age_v)
        # np.clip(res, 0.0, 1.0, out=res)
        res[~act_s | np.isinf(res)] = _INTERNAL_SENTINEL
        features[key] = res

    def _assign_fraction(field_mask, key):
        """Fraction of in-plan fields in each bin satisfying ``field_mask``."""
        res = np.full(nbins, _INTERNAL_SENTINEL, dtype=np.float32)
        num = np.bincount(bins_mem, weights=field_mask, minlength=nbins)
        np.divide(num, bin_in_plan_counts, out=res, where=act_s)
        features[key] = res
 
    _assign_fraction(
        (v_cur_field == 0) & in_plan, "num_unvisited_fields"
    )
    _assign_fraction(
        (v_cur_field < max_adj) & in_plan, "num_incomplete_fields"
    )
 
    # Per-bin min tiling. Out-of-plan fields contribute +inf so they're ignored
    # by np.minimum.at; bins with no in-plan fields stay +inf and are converted
    # to NaN at the end.
    tiling = np.divide(
        v_cur_field.astype(np.float32),
        max_adj.astype(np.float32),
        out=np.full(v_cur_field.shape, np.inf, dtype=np.float32),
        where=in_plan,
    )
    s_mins = np.full(nbins, np.inf, dtype=np.float32)
    np.minimum.at(s_mins, bins_mem, tiling)
    s_mins[~act_s | np.isinf(s_mins)] = _INTERNAL_SENTINEL
    features["min_tiling"] = np.minimum(s_mins, 1.0)
 
    if not do_filt:
        if timestamp is not None and last_visit_timestamps is not None:
            if last_visit_timestamps.ndim == 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    last_visit_field = np.nanmax(last_visit_timestamps, axis=1)
            else:
                last_visit_field = last_visit_timestamps
            # Restrict staleness to in-plan AND incomplete fields. Completed
            # fields would otherwise contribute monotonically-growing values
            # that conflate "this region is forgotten" with "this region is
            # done." `in_plan` is already `v_tgt_field > 0`, so the AND is
            # equivalent to `v_cur_field < v_tgt_field`.
            incomplete = v_cur_field < v_tgt_field
            _assign_staleness(
                last_visit_field[v_mask],
                in_plan & incomplete,
                "t_since_last_visit",
            )
        return features
 
    # Per-filter family of features
    nfilters = current_counts.shape[1]
    v_cur_ff = current_counts[v_mask]
    v_tgt_ff = target_counts[v_mask]
    in_ff_plan = v_tgt_ff > 0                           # (nv, nf) bool
    max_ff_adj = np.maximum(v_cur_ff, v_tgt_ff)

    ff_tiling = np.divide(
        v_cur_ff.astype(np.float32),
        max_ff_adj.astype(np.float32),
        out=np.full(v_cur_ff.shape, np.inf, dtype=np.float32),
        where=in_ff_plan,
    )

    incomplete_ff = v_cur_ff < v_tgt_ff                 # (nv, nf) bool
    unvisited_ff  = (v_cur_ff == 0) & in_ff_plan        # (nv, nf) bool

    # Batch all three per-filter bincounts into single (nbins, nfilters) passes.
    bc_all  = np.zeros((nbins, nfilters), dtype=np.float32)
    unv_all = np.zeros((nbins, nfilters), dtype=np.float32)
    inc_all = np.zeros((nbins, nfilters), dtype=np.float32)
    np.add.at(bc_all,  bins_mem, in_ff_plan.astype(np.float32))
    np.add.at(unv_all, bins_mem, unvisited_ff.astype(np.float32))
    np.add.at(inc_all, bins_mem, (incomplete_ff & in_ff_plan).astype(np.float32))
    act_f_all = bc_all > 0                              # (nbins, nf) bool

    for f, filt_name in idx2filter.items():
        if timestamp is not None and last_visit_timestamps is not None:
            _assign_staleness(
                last_visit_timestamps[v_mask, f],
                in_ff_plan[:, f] & incomplete_ff[:, f],
                f"t_since_last_visit_{filt_name}",
            )
        act_f = act_f_all[:, f]
        bc_f  = bc_all[:, f]

        unv = np.full(nbins, _INTERNAL_SENTINEL, dtype=np.float32)
        np.divide(unv_all[:, f], bc_f, out=unv, where=act_f)
        features[f"num_unvisited_fields_{filt_name}"] = unv

        inc = np.full(nbins, _INTERNAL_SENTINEL, dtype=np.float32)
        np.divide(inc_all[:, f], bc_f, out=inc, where=act_f)
        features[f"num_incomplete_fields_{filt_name}"] = inc

        s_f_mins = np.full(nbins, np.inf, dtype=np.float32)
        np.minimum.at(s_f_mins, bins_mem, ff_tiling[:, f])
        s_f_mins[~act_f | np.isinf(s_f_mins)] = _INTERNAL_SENTINEL
        features[f"min_tiling_{filt_name}"] = np.minimum(s_f_mins, 1.0)
    return features
 
 
def apply_relative_bin_features(features, el_mask, has_historical, do_filt):
    """Add ``rel_*`` features in place. Works on ``(nbins,)`` or batched
    ``(..., nbins)`` arrays — the relative subtraction operates on the
    trailing axis via ``np.nanmean``.
 
    NaN values (inactive bins) propagate correctly: they're excluded from
    the local mean and stay NaN in the rel output.
    """
    if 'ha' in features:
        features['rel_ha'] = get_relative_feature(features['ha'], el_mask)
    if 'moon_distance' in features:
        features['rel_moon_distance'] = get_relative_feature(features['moon_distance'], el_mask)
 
    if not has_historical:
        return
    
    if do_filt:
        base_keys = _SURVEY_PROGRESS_BASE_KEYS + _STALENESS_BASE_KEYS
        keys = [f"{bk}_{filt}" for bk in base_keys for filt in FILTER2IDX.keys()]
    else:
        keys = _SURVEY_PROGRESS_BASE_KEYS + _STALENESS_BASE_KEYS
    for k in keys:
        if k in features:
            features[f"rel_{k}"] = get_relative_feature(features[k], el_mask)
 
 
def validate_history_bin_features(features, do_filt, idx2filter=None):
    """Sanity-check history features. NaN-aware (NaN = inactive bin).
    
    Does not check mean_tiling
 
    Raises ``RuntimeError`` on:
      * Bounds: fractions outside [0, 1]
      * Subset rule: unvisited > incomplete
      * Tiling floor: unvisited > 0 in a bin where min_tiling > 0
 
    Works on ``(nbins,)`` or batched ``(n_timestamps, nbins)`` arrays.
    """
    if idx2filter is None:
        idx2filter = IDX2FILTER
 
    check_groups = [{
        'unv': 'num_unvisited_fields',
        'inc': 'num_incomplete_fields',
        'til': 'min_tiling',
        'name': 'survey (base)',
    }]
    if do_filt:
        for filt_name in idx2filter.values():
            check_groups.append({
                'unv': f'num_unvisited_fields_{filt_name}',
                'inc': f'num_incomplete_fields_{filt_name}',
                'til': f'min_tiling_{filt_name}',
                'name': f'survey ({filt_name})',
            })
 
    for grp in check_groups:
        unv_k, inc_k, til_k = grp['unv'], grp['inc'], grp['til']
        if not all(k in features for k in (unv_k, inc_k, til_k)):
            continue
        unv, inc, til = features[unv_k], features[inc_k], features[til_k]
 
        v_unv = ~np.isnan(unv)
        v_inc = ~np.isnan(inc)
        v_til = ~np.isnan(til)
 
        # 1. Bounds [0, 1]
        bad_unv = v_unv & ((unv < 0.0) | (unv > 1.0))
        if np.any(bad_unv):
            bad = np.where(bad_unv)
            raise RuntimeError(
                f"FATAL BOUNDS: {unv_k} out of [0,1] at idx {bad[0][0]}. "
                f"Val: {unv[bad][0]}"
            )
        bad_inc = v_inc & ((inc < 0.0) | (inc > 1.0))
        if np.any(bad_inc):
            bad = np.where(bad_inc)
            raise RuntimeError(
                f"FATAL BOUNDS: {inc_k} out of [0,1] at idx {bad[0][0]}. "
                f"Val: {inc[bad][0]}"
            )
 
        # 2. Subset rule: unvisited <= incomplete
        both = v_unv & v_inc
        subset_violation = both & (unv > (inc + 1e-5))
        if np.any(subset_violation):
            bad = np.where(subset_violation)
            raise RuntimeError(
                f"FATAL LOGIC LEAK: {grp['name']} unvisited > incomplete at "
                f"idx {bad[0][0]}. Unv: {unv[bad][0]}, Inc: {inc[bad][0]}"
            )
 
        # 3. Tiling floor: unvisited > 0 implies min_tiling == 0
        both_til = v_unv & v_til
        has_unv = unv > 1e-5
        tiling_violation = both_til & has_unv & (til > 1e-5)
        if np.any(tiling_violation):
            bad = np.where(tiling_violation)
            raise RuntimeError(
                f"FATAL LOGIC LEAK: {grp['name']} has unvisited fields but "
                f"min_tiling > 0 at idx {bad[0][0]}. "
                f"Unv: {unv[bad][0]}, Til: {til[bad][0]}"
            )

    # Staleness keys: should sit in [0, 1] with the OT-clock normalization.
    # Tolerate a tiny float epsilon above 1.0 from accumulated rounding.
    stale_keys = (
        _STALENESS_BASE_KEYS if not do_filt else
        [f"{bk}_{filt_name}"
         for bk in _STALENESS_BASE_KEYS
         for filt_name in idx2filter.values()]
    )
    for bk in stale_keys:
        if bk not in features:
            continue
        arr = features[bk]
        valid = ~np.isnan(arr)
        # bad = valid & ((arr < 0.0) | (arr > 1.0 + 1e-5))
        bad = valid & (arr < 0.0)
        if np.any(bad):
            b = np.where(bad)
            raise RuntimeError(
                f"FATAL BOUNDS: {bk} out of [0,1] at idx {b[0][0]}. "
                f"Val: {arr[b][0]}"
            )
                

# Keep just in case. Remove when StateNormalizer.fit_transform() is confidently implemented.
# def replace_nan_with_sentinel(features, sentinel_val, keys=None):
#     """Convert NaN -> ``sentinel_val`` in place for the given keys
#     (or every key if ``keys`` is None). Only writes when a NaN exists,
#     leaving non-history features untouched.
#     """
#     if keys is None:
#         keys = list(features.keys())
#     for k in keys:
#         if k not in features:
#             continue
#         arr = features[k]
#         nan_mask = np.isnan(arr)
#         if nan_mask.any():
#             arr[nan_mask] = sentinel_val
            
def replace_invalid_with_sentinel(features, sentinel_val, keys=None):
    """Convert NaN and ±inf -> ``sentinel_val`` in place for the given keys
    (or every key if ``keys`` is None). Only writes when an invalid value
    exists, leaving clean arrays untouched.

    Both NaN (inactive history bins) and inf (circumpolar / below-horizon
    bins from `get_time_until_set`) get the same sentinel — they're both
    "this bin is unavailable" markers semantically.
    """
    if keys is None:
        keys = list(features.keys())
    for k in keys:
        if k not in features:
            continue
        arr = features[k]
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = sentinel_val
 
 
# ============================================================================
# Small math helpers — used by callers across the codebase, kept module-level.
# ============================================================================
 
 
def get_relative_feature(feat_arr, el_mask):
    """Subtract the per-timestep mean (over above-horizon bins) from
    ``feat_arr``. NaN-aware: NaN values are excluded from the mean and
    stay NaN in the output.
    """
    valid_cols = np.where(el_mask, feat_arr, np.nan)
    with warnings.catch_warnings():
        # nanmean over an all-NaN slice warns and returns NaN — that's fine.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local_mean = np.nanmean(valid_cols, axis=-1, keepdims=True)
    return feat_arr - local_mean
 
 
def get_delta_az_el(bin_azs, bin_els, target_az, target_el):
    """Angular differences. ``az`` is wrapped to ``[-π, π)``; ``el`` is
    naive subtraction (elevation isn't periodic in the relevant range).
    """
    azs = (bin_azs - target_az + np.pi) % (2 * np.pi) - np.pi
    els = bin_els - target_el
    return azs, els
 
 
# ============================================================================
# BinFeatureEngineer — offline batch pipeline, driven by the helpers above.
# ============================================================================
 
 
class BinFeatureEngineer:
    """Offline batch feature engineering.
 
    Drives the shared per-timestep helpers in a tight row loop, writing
    outputs into pre-allocated ``(n_timestamps, n_bins)`` arrays for
    cache-friendliness. Cyclical, relative, validation, and sentinel
    conversion run once over the full batch at the end.
    """
 
    # How often (in seconds) to refresh field->bin mapping in AzEl mode.
    # The sky rotates ~0.25 deg / min, so 5 minutes (~1.25 deg) is well
    # under typical healpix bin sizes for the surveys we run.
    _AZEL_CACHE_REFRESH_S = 300
 
    def __init__(
        self,
        hpGrid,
        base_features,
        cyclical_features,
        action_space,
        lookups,
        do_cyclical_norm=True,
        do_local_mean_z_score=True,
    ):
        self.hpGrid = hpGrid
        self.base_features = list(base_features)
        self.cyclical_features = cyclical_features
        self.action_space = action_space
        self.lookups = lookups
        self.do_cyclical_norm = do_cyclical_norm
        self.do_local_mean_z_score = do_local_mean_z_score
        self.do_filt = 'filter' in action_space
        self.is_azel = hpGrid.is_azel
        self.has_historical = any(
            hk in f
            for f in self.base_features
            for hk in _SURVEY_PROGRESS_BASE_KEYS + _STALENESS_BASE_KEYS
        )
        self.sentinel_val = (
            AZEL_BIN_FEAT_SENTINEL if self.is_azel else RADEC_BIN_FEAT_SENTINEL
        )
 
 
    def transform(self, pt_df, requested_features) -> np.ndarray:
        """Run the full pipeline and return a ``(nrows, nbins, nfeats)`` tensor.
 
        ``requested_features`` controls the final stack order; any feature
        name there must be produced by the pipeline (or appear after rel/
        cyclical expansion).
        """
        timestamps = pt_df['timestamp'].values
        assert all(np.diff(timestamps) > 0), \
            "Timestamps must be strictly increasing."
 
        ntimestamps = len(timestamps)
        nbins = len(self.hpGrid.idx_lookup)
 
        features = self._pre_allocate_arrays(ntimestamps, nbins)
 
        # Per-timestep core: drives the shared helpers, writes into pre-allocated arrays.
        self._fill_features_per_timestep(features, pt_df)
 
        # Once-over post-processing on the batched arrays. el_mask is
        # broadcast-shaped (n_t, n_b); rel/validate work on the trailing axis.
        el_mask = features['el'] > 0
 
        if self.do_local_mean_z_score:
            apply_relative_bin_features(
                features, el_mask, self.has_historical, self.do_filt
            )
 
        if self.do_cyclical_norm:
            apply_cyclical_features(
                features, self.base_features, self.cyclical_features
            )
 
        if self.has_historical:
            validate_history_bin_features(features, self.do_filt)

        # NOTE: internal NaN -> external sentinel conversion happens in the
        # StateNormalizer pipeline (so the sentinel mask can be saved for
        # downstream plotting), not here.

        return self._stack_and_rearrange(features, requested_features)
 
 
    def _pre_allocate_arrays(self, ntimestamps, nbins):
        """Allocate the ``(n_t, n_b)`` arrays we'll fill in the row loop.
 
        We pre-allocate every key the shared ephemeris helper produces,
        plus history keys when needed. The shared helpers always produce
        the full ephemeris set per timestep; pre-allocating all of them
        is ~10 small float32 arrays — cheap compared to the history work.
        """
        shape = (ntimestamps, nbins)
        ephemeris_keys = [
            'ra', 'dec', 'az', 'el', 'ha', 'airmass',
            'moon_distance', 'sun_distance', 'pointing_distance', 'delta_az', 'delta_el',
            't_until_set',
        ]
        features = {
            k: np.full(shape, np.nan, dtype=np.float32) for k in ephemeris_keys
        }
        if self.has_historical:
            for bk in _SURVEY_PROGRESS_BASE_KEYS + _STALENESS_BASE_KEYS:
                features[bk] = np.full(shape, np.nan, dtype=np.float32)
                if self.do_filt:
                    for filt in FILTER2IDX.keys():
                        features[f"{bk}_{filt}"] = np.full(
                            shape, np.nan, dtype=np.float32
                        )
        return features
 
 
    def _fill_features_per_timestep(self, features, pt_df):
        """Drive the shared helpers for each row, writing into ``features``.
 
        Iterates rows via ``pt_df.groupby('night', sort=False)``. This is
        what the original implementation did and we keep it for two reasons:
 
          1. The groupby key is the same pandas Timestamp the upstream
             code used to build ``night2{fid,fidfilt}_visit_hist``, so the
             dict lookup is type-safe. (Going through ``pt_df['night'].values``
             gives ``numpy.datetime64`` instead, which won't match.)
          2. Night boundaries are implicit in the iteration, so we don't
             need separate "did the night change?" bookkeeping.
 
        With timestamps strictly increasing (asserted in ``transform``) and
        ``sort=False``, groups iterate in row order, so the global counter
        ``i`` matches each row's slot in the pre-allocated arrays.
 
        Features at row i reflect counts from rows 0..i-1, 
        i.e., the world state before the observation at row i was made

        OT-clock handling: ``last_visit_*_ot`` and ``timestamp`` passed to
        ``compute_bin_progress_features`` are both in observing-time seconds,
        sharing the clock built by ``build_train_lookups.py``. Per-row
        conversion is ``obs_t = ot_at_sunset + (t - sunset_ts)``: inside a
        night, OT advances at 1 s per real second from the night's sunset
        OT anchor.
        """
        nbins = len(self.hpGrid.idx_lookup)
 
        # Cheap path: ephemeris only.
        if not self.has_historical:
            self._fill_ephemeris_only(features, pt_df)
            return
 
        # History path setup.
        ra_arr = self.lookups.fields['ra'].to_numpy()
        dec_arr = self.lookups.fields['dec'].to_numpy()
 
        # Map filter strings to indices once for the whole frame; we then
        # index per-group inside the loop.
        filt_idx_full = (
            pt_df['filter'].map(FILTER2IDX)
            .fillna(ZENITH_FILTER_IDX).astype(np.int32)
            .to_numpy()
        )
        pt_df_with_filt = pt_df.assign(_filt_idx=filt_idx_full)
 
        # RaDec: field->bin mapping is static for the whole run.
        if self.is_azel:
            bins_static, vmask_static = None, None
        else:
            bins_raw = self.hpGrid.ang2idx(lon=ra_arr, lat=dec_arr)
            bins_static = np.array(
                [b if b is not None else ZENITH_BIN_NUM for b in bins_raw],
                dtype=np.int32,
            )
            vmask_static = bins_static != ZENITH_BIN_NUM
 
        # AzEl cache, persisted across groups so a night boundary doesn't
        # invalidate a still-fresh field->bin assignment.
        cache_time = -1e9
        cache_bins, cache_vmask = None, None

        # Resolve the OT divisor and the OT lookups once. These are
        # required for staleness; raise loudly if absent so a stale-feature
        # bug never silently degrades to all-sentinel.
        # total_ot_sec = self._require_lookup_attr("total_ot_sec")
        ot_clock_dict = self._require_lookup_attr("night2ot_clock_seconds")
        if self.do_filt:
            last_visit_ot_dict = self._require_lookup_attr(
                "night2fidfilt_last_visit_ot"
            )
        else:
            last_visit_ot_dict = self._require_lookup_attr(
                "night2fid_last_visit_ot"
            )

        pbar = tqdm(total=len(pt_df), desc='Computing bin features')
        i = 0  # Global row index — matches the row's slot in pre-allocated arrays.

        for night, group in pt_df_with_filt.groupby('night', sort=False):
            # Per-night setup — done ONCE per night, not once per row.
            sunset_ts, sunrise_ts = get_night_boundaries(
                group['timestamp'], sun_el_limit=_SUN_EL_LIMIT_DEG
            )
            step_night_duration_sec = sunrise_ts - sunset_ts
            ot_at_sunset = ot_clock_dict[night]  # scalar OT(sunset_n)

            # Per-night running counters + last-visit OT, seeded from the
            # full-survey history snapshot at the start of this night.
            if self.do_filt:
                cur_s_f_vis = (
                    self.lookups.night2fidfilt_visit_hist[night]
                    .copy().astype(np.int32)
                )
                last_visit_ot = (
                    last_visit_ot_dict[night].copy().astype(np.float64)
                )
                cur_s_vis = None
                last_visit_ot_1d = None
            else:
                cur_s_vis = (
                    self.lookups.night2fid_visit_hist[night]
                    .copy().astype(np.int32)
                )
                last_visit_ot_1d = (
                    last_visit_ot_dict[night].copy().astype(np.float64)
                )
                cur_s_f_vis = None
                last_visit_ot = None

            # Extract group columns as ndarrays for the inner loop.
            step_timestamps = group['timestamp'].to_numpy()
            step_pointing_ras = group['ra'].to_numpy()
            step_pointing_decs = group['dec'].to_numpy()
            step_fids = group['field_id'].to_numpy(dtype=np.int32)
            step_filts = group['_filt_idx'].to_numpy(dtype=np.int32)

            for j in range(len(group)):
                t = step_timestamps[j]
                # Convert unix timestamp -> OT seconds for this row. OT
                # advances at 1 s/s while inside the night; this row is
                # inside [sunset_ts, sunrise_ts] by construction (group
                # belongs to night `night`).
                obs_t = ot_at_sunset + (t - sunset_ts)

                # --- Ephemeris (always) ---
                eph = compute_bin_ephemeris_features(
                    timestamp=t,
                    pointing_radec=(step_pointing_ras[j], step_pointing_decs[j]),
                    hpGrid=self.hpGrid,
                    night_duration_in_sec=step_night_duration_sec
                )
                
                for k, v in eph.items():
                    if k in features:
                        features[k][i] = v
 
                # --- Field->bin mapping (static for RaDec, cached for AzEl) ---
                if self.is_azel:
                    if abs(t - cache_time) > self._AZEL_CACHE_REFRESH_S:
                        az, el = ephemerides.equatorial_to_topographic(
                            ra=ra_arr, dec=dec_arr, time=t
                        )
                        bins = np.array(
                            [b if b is not None else ZENITH_BIN_NUM
                             for b in self.hpGrid.ang2idx(lon=az, lat=el)],
                            dtype=np.int32,
                        )
                        cache_vmask = (el > 0) & (bins != ZENITH_BIN_NUM)
                        cache_bins = bins
                        cache_time = t
                    bpf, vm = cache_bins, cache_vmask
                else:
                    bpf, vm = bins_static, vmask_static
 
                # --- History features via shared helper ---
                if self.do_filt:
                    hist = compute_bin_progress_features(
                        current_counts=cur_s_f_vis,
                        target_counts=self.lookups.target_fidfilt_counts,
                        bins_per_field=bpf, v_mask=vm,
                        nbins=nbins, do_filt=True,
                        timestamp=obs_t,
                        last_visit_timestamps=last_visit_ot,
                        t_since_last_visit_divisor=None,#total_ot_sec,
                    )
                else:
                    hist = compute_bin_progress_features(
                        current_counts=cur_s_vis,
                        target_counts=self.lookups.target_fid_counts,
                        bins_per_field=bpf, v_mask=vm,
                        nbins=nbins, do_filt=False,
                        timestamp=obs_t,
                        last_visit_timestamps=last_visit_ot_1d,
                        t_since_last_visit_divisor=None,#total_ot_sec,
                    )
                for k, v in hist.items():
                    if k in features:
                        features[k][i] = v

                # --- Increment running counter / refresh last_visit for next row's view ---
                obs_fid, obs_filt = step_fids[j], step_filts[j]
                if obs_fid != ZENITH_FIELD_ID:
                    if self.do_filt:
                        if obs_filt != ZENITH_FILTER_IDX:
                            cur_s_f_vis[obs_fid, obs_filt] += 1
                            last_visit_ot[obs_fid, obs_filt] = obs_t
                    else:
                        cur_s_vis[obs_fid] += 1
                        last_visit_ot_1d[obs_fid] = obs_t

                i += 1
                pbar.update(1)
 
        pbar.close()
        # Sanity: every pre-allocated row should have been written.
        assert i == len(pt_df), (
            f"Row loop wrote {i} rows but pt_df has {len(pt_df)}. "
            f"Did groupby reorder relative to pt_df?"
        )

    def _require_lookup_attr(self, name):
        """Fetch a required attribute from ``self.lookups`` or raise loudly.

        We used to fall back to all-NaN when these were missing; that
        silently corrupted staleness for an entire training run and was
        the cause of the recent all-zeros and all-sentinel debugging
        rounds. Better to crash here than to silently train on garbage.
        """
        if not hasattr(self.lookups, name):
            raise AttributeError(
                f"LookupTables is missing required attribute '{name}' for "
                f"OT-clock staleness computation. Rebuild lookups via "
                f"`build_train_lookups.py` (or set has_historical=False / "
                f"drop t_since_last_visit from base_features if you don't "
                f"need staleness)."
            )
        return getattr(self.lookups, name)

    def _fill_ephemeris_only(self, features, pt_df):
        """Fast path for configs that don't request any history features."""
        i = 0
        for _, group in tqdm(
            pt_df.groupby('night', sort=False),
            total=pt_df['night'].nunique(),
            desc='Computing bin ephemeris',
        ):
            sunset_ts, sunrise_ts = get_night_boundaries(
                group['timestamp'], sun_el_limit=_SUN_EL_LIMIT_DEG
            )
            night_duration_sec = sunrise_ts - sunset_ts
            ts = group['timestamp'].to_numpy()
            ras = group['ra'].to_numpy()
            decs = group['dec'].to_numpy()
            for j in range(len(group)):
                eph = compute_bin_ephemeris_features(
                    timestamp=ts[j],
                    pointing_radec=(ras[j], decs[j]),
                    hpGrid=self.hpGrid,
                    night_duration_in_sec=night_duration_sec,
                )
                for k, v in eph.items():
                    if k in features:
                        features[k][i] = v
                i += 1
 
    # ------------------------------------------------------------------
    # Stacking
    # ------------------------------------------------------------------
 
    def _stack_and_rearrange(self, features, requested_features) -> np.ndarray:
        """Pop requested arrays, validate they exist, reshape via einops."""
        final_arrays = []
        for key in requested_features:
            if key not in features:
                raise ValueError(
                    f"Requested feature '{key}' was not calculated by the pipeline."
                )
            final_arrays.append(features.pop(key))
 
        assert len(final_arrays) == len(requested_features)
        bin_states = np.array(final_arrays)
        return rearrange(
            bin_states, 'nfeats nrows nbins -> nrows nbins nfeats'
        )
