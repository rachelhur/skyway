"""Global (per-timestep, single-row) feature computation.

The module-level helpers at the top define the canonical per-timestep
computation. Both ``BaseBlancoEnv._calculate_global_features`` (live,
1 timestep) and ``GlobalFeatureEngineer.transform`` (offline, batch) drive
them — the offline pipeline iterating rows and writing into DataFrame
columns; the live env returning a single dict.

Two helpers split the computation by data dependency:

- ``compute_global_time_only_features``: depends only on ``timestamp``.
  Sun/Moon positions, Moon phase, LST. Used by both pipelines.
- ``compute_global_pointing_features``: depends on ``(timestamp, ra, dec)``.
  Topocentric az/el/ha, plane-parallel airmass, per-filter sky brightness.
  Used by the live env only — the offline pipeline reads az/el/ha from the
  raw FITS measurements (which differ from ephemeris values due to
  atmospheric refraction and pointing error) and uses its own vectorized
  sky-brightness pass for batch efficiency.

A third helper, ``apply_cyclical_global_features``, expands cyclical
features to ``_cos``/``_sin`` pairs. It works on either a dict (live) or a
pandas DataFrame (offline) via duck-typed `in` / ``[k]`` / ``[k] = v``.
"""
from datetime import time, timezone, timedelta
import datetime
from typing import Union
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
import torch
import ephem
from astropy.time import Time
from tqdm import tqdm
from datetime import date, datetime, timedelta, timezone

from blancops.data.features.normalizations import apply_cyclical_features
from blancops.ephemerides.time_utils import standardize_time, unix_to_datetime
from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.data_quality.seeing import Seeing
from blancops.configs.constants import (
    BLANCO_LON, IDX2FILTER, ZENITH_BIN_NUM, ZENITH_FIELD_ID, ZENITH_WAVELENGTH,
    FILTER2WAVE, FILTERWAVENORM, FILTER2IDX, ZENITH_FILTER, IDX2WAVE,
    FWHM_REF_WAVELENGTH, FWHM_REF_FILTER,
    ZENITH_AZ, ZENITH_EL, ZENITH_AIRMASS, ZENITH_ZD, ZENITH_HA, ZENITH_OBJECT
)

import logging

from blancops.math.geometry import angular_separation
logger = logging.getLogger(__name__)


# ============================================================================
# Canonical per-timestep helpers — single source of truth, shared between
# BaseBlancoEnv (live, single-step) and GlobalFeatureEngineer (offline batch).
# ============================================================================


def compute_global_time_only_features(*, timestamp) -> dict:
    """Time-only global ephemeris features (no pointing dependence).

    Returns a dict with: ``lst``, ``sun_ra``, ``sun_dec``, ``sun_az``,
    ``sun_el``, ``moon_ra``, ``moon_dec``, ``moon_az``, ``moon_el``,
    ``moon_phase`` — each a scalar at the given timestamp.

    Used by both pipelines: live calls this once per step; offline calls
    it inside its row loop to fill the sun/moon/lst columns.
    """
    features = {}

    astro_time = Time(timestamp, format='unix', scale='utc')
    features['lst'] = float(
        astro_time.sidereal_time('apparent', longitude=BLANCO_LON).radian
    )

    sun_radec, sun_azel, moon_radec, moon_azel = calc_sun_and_moon_positions(timestamp)
    features['sun_ra'], features['sun_dec'] = sun_radec
    features['sun_az'], features['sun_el'] = sun_azel
    features['moon_ra'], features['moon_dec'] = moon_radec
    features['moon_az'], features['moon_el'] = moon_azel
    features['moon_phase'] = calc_moon_phase(timestamp)

    return features


def compute_global_pointing_features(timestamp, ra, dec, moon_radec) -> dict:
    """Pointing-dependent global ephemeris features.

    Returns a dict with: ``az``, ``el`` (clipped to ``[0, pi/2]``), ``ha``,
    ``airmass``, ``moon_distance``, and ``sky_brightness_<filter>`` for each
    filter in ``FILTER2IDX``.

    The elevation clip absorbs the precision issue where ``el`` can be
    slightly negative just before sunrise/sunset and propagate into the
    airmass calculation as a divergent value.

    Used by the live env. The offline pipeline takes ``az``/``el``/``ha``
    from FITS measurements (which differ from ephemeris values due to
    atmospheric refraction and pointing error) and uses a vectorized
    sky-brightness pass for batch efficiency, so it does not call this
    helper.

    Parameters
    ----------
    timestamp : float
        Unix time of the pointing.
    ra, dec : float
        Target equatorial coordinates (radians).
    moon_radec : tuple of float
        Moon (ra, dec) at ``timestamp``, used for the ``moon_distance`` term.
    """
    features = {}

    az, el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=timestamp)
    el = max(min(el, np.pi / 2), 0.0)
    features['az'] = az
    features['el'] = el
    features['ha'] = ephemerides.equatorial_to_hour_angle(
        ra=ra, dec=dec, time=timestamp
    )
    features['airmass'] = 1.0 / np.cos(np.pi / 2 - el)
    features['moon_distance'] = calc_distance_to_moon(moon_radec, (ra, dec))

    for filt in FILTER2IDX.keys():
        features[f"sky_brightness_{filt}"] = estimate_sky_brightness(
            time=timestamp, ra=ra, dec=dec, band=filt
        )

    return features


def compute_global_mean_tiling_features(running_counts, target_counts) -> dict:
    """Survey-completion scalars from a running visit-count snapshot.

    Computes "what fraction of the survey is done" without referencing
    calendar time. Replaces `t_survey` / `survey_progress` as a
    survey-agnostic strategic-shift signal: 0 means nothing done, 1 means
    everything at or above target, intermediate values encode partial
    completion that the agent can use to shift strategy without leaking
    the calendar identity of this specific survey.

    Returns a dict with:
      - 'global_mean_tiling': scalar in [0, 1]. Mean of per-(field, filter)
        tiling across all in-plan fields (target > 0). Per-(field,filter) tiling
        is `current / max(current, target)`, naturally capped at 1.
      - 'global_mean_tiling_{filt}' for each filter in FILTER2IDX: scalar in
        [0, 1]. Mean of per-field tiling across in-plan fields in
        that filter. Useful when filter strategy shifts late-survey
        (e.g., Y dominating after g/r/i complete).

    Args:
        running_counts: (nfields, nfilters) int — visit counts so far.
        target_counts: (nfields, nfilters) int — survey targets, fixed.
    """
    features = {}
    in_plan = target_counts > 0

    if not in_plan.any():
        # Defensive: empty lookup. Return zeros for every key the caller
        # might request rather than crashing.
        features['global_mean_tiling'] = 0.0
        for filt in FILTER2IDX.keys():
            features[f'global_mean_tiling_{filt}'] = 0.0
        return features

    max_adj = np.maximum(running_counts, target_counts)
    with np.errstate(divide='ignore', invalid='ignore'):
        tiling = np.where(
            in_plan,
            running_counts / max_adj.astype(np.float32),
            0.0,
        )

    features['global_mean_tiling'] = float(tiling[in_plan].mean())

    for filt, fidx in FILTER2IDX.items():
        col_in_plan = in_plan[:, fidx]
        if col_in_plan.any():
            features[f'global_mean_tiling_{filt}'] = float(
                tiling[col_in_plan, fidx].mean()
            )
        else:
            features[f'global_mean_tiling_{filt}'] = 0.0

    return features


def compute_global_survey_progress_features(tracker, idx2filter=None) -> dict:
    """Per-filter fractional progress, summed across fields.

    Returns {'survey_progress_{filt}': float} for every filter in `idx2filter`.
    Requires a field-filter (2D) tracker. Returns an empty dict if the tracker
    is field-only — callers requesting these features in 1D mode should be
    caught by `_validate_feature_config` upstream.
    """
    if idx2filter is None:
        idx2filter = IDX2FILTER
    if not tracker._is_field_filter:
        return {}
    return {
        f"survey_progress_{filt}": tracker.get_filter_progress(fidx)
        for fidx, filt in idx2filter.items()
    }


def compute_global_urgency_features(
    tracker, survey_night_idx, survey_nights_total, idx2filter=None,
) -> dict:
    """Per-filter urgency = remaining-progress / remaining-time.

    Mirrors the inline computation in the previous env loop. Returns
    {'urgency_{filt}': float} for every filter; 0.0 for filters with
    target == 0 (matches the prior live-env behavior).
    """
    if idx2filter is None:
        idx2filter = IDX2FILTER
    if not tracker._is_field_filter:
        return {}

    out = {}
    raw = tracker.raw_counts
    tgt = tracker.target_counts
    for fidx, filt in idx2filter.items():
        target = int(tgt[:, fidx].sum())
        if target == 0:
            out[f"urgency_{filt}"] = 0.0
            continue
        visits = int(raw[:, fidx].sum())
        out[f"urgency_{filt}"] = calc_urgency(
            filter_counts_arr=visits,
            filter_counts_max=target,
            survey_night_indices=survey_night_idx,
            survey_nights_max=survey_nights_total,
        )
    return out

# Registry of per-filter / tracker-derived global feature families.
# To add a new family:
#   1. Write `compute_global_<thing>_features(...)` above.
#   2. Add an entry here with the requested-name prefix and a small lambda
#      that pulls its inputs out of `ctx`.
# Nothing in base.py needs to change.
_GLOBAL_TRACKER_FAMILIES = [
    {
        "prefix": "survey_progress",
        "needs_tracker_2d": True,
        "fn": lambda ctx: compute_global_survey_progress_features(
            tracker=ctx["tracker"], idx2filter=ctx["idx2filter"],
        ),
    },
    {
        "prefix": "urgency",
        "needs_tracker_2d": True,
        "fn": lambda ctx: compute_global_urgency_features(
            tracker=ctx["tracker"],
            survey_night_idx=ctx["survey_night_idx"],
            survey_nights_total=ctx["survey_nights_total"],
            idx2filter=ctx["idx2filter"],
        ),
    },
    {
        "prefix": "global_mean_tiling",
        "needs_tracker_2d": False,  # works for 1D too
        "fn": lambda ctx: compute_global_mean_tiling_features(
            running_counts=ctx["tracker"].raw_counts,
            target_counts=ctx["tracker"].target_counts,
        ),
    },
]


def compute_global_tracker_features(requested_names, tracker, ctx, force_all=False) -> dict:
    """Drive every tracker-derived feature family.

    For each family in `_GLOBAL_TRACKER_FAMILIES`, check whether any
    requested feature name starts with the family's prefix; if so, call
    the family's compute function with `ctx` and merge the result.

    Skips families that need a 2D tracker when `tracker._is_field_filter`
    is False, so 1D configs don't blow up requesting context they can't
    populate.

    `ctx` is a dict carrying everything any family might need:
    `tracker`, `idx2filter`, and lazily-evaluated hooks like
    `survey_night_idx` / `survey_nights_total`. Pass them eagerly from
    the env; this function never calls back into env hooks.
    """
    requested_set = set(requested_names)
    out = {}
    for fam in _GLOBAL_TRACKER_FAMILIES:
        wanted = force_all or any(
            n == fam["prefix"] or n.startswith(f"{fam['prefix']}_")
            for n in requested_names
        )
        if not wanted:
            continue
        if fam["needs_tracker_2d"] and not tracker._is_field_filter:
            continue
        out.update(fam["fn"](ctx))
    return out
# ============================================================================
# GlobalFeatureEngineer — offline batch pipeline.
# ============================================================================


class GlobalFeatureEngineer:
    """Pipeline for calculating global state features for blancops RL."""

    def __init__(self, lookups, hpGrid, base_features, cyclical_features,
                 do_cyclical_norm=True, do_filt=True):
        self.lookups = lookups
        self.hpGrid = hpGrid
        self.base_features = base_features
        self.cyclical_features = cyclical_features
        self.do_cyclical_norm = do_cyclical_norm
        self.do_filt = do_filt

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the full feature engineering pipeline."""
        return (df
            .pipe(self._add_zenith_rows)
            .pipe(self._add_time_dependent_features)
            .pipe(self._add_moon_distance)
            .pipe(self._map_bins_and_fields)
            .pipe(self._add_current_filter)
            .pipe(self._add_sky_brightness)
            .pipe(self._add_global_mean_tiling)
            .pipe(self._apply_cyclical_norms)
            .pipe(self._ensure_32_bit)
        )

    def _add_zenith_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        zenith_df = get_zenith_features(original_df=df)
        df = _backfill_zenith_states(merge_zenith_df(df, zenith_df))
        return df

    def _add_time_dependent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill LST + sun/moon + moon_phase columns via the shared helper,
        then add ``t_night`` (which needs per-night normalization).
        """
        timestamps = df['timestamp'].values

        # Drive the shared helper per row; accumulate into per-column lists,
        # then assign back to df in one shot.
        feat_lists = defaultdict(list)
        for t in tqdm(timestamps, total=len(timestamps),
                      desc='Calculating sun/moon/lst ephemeris'):
            feats = compute_global_time_only_features(timestamp=t)
            for k, v in feats.items():
                feat_lists[k].append(v)

        for k, vs in feat_lists.items():
            df[k] = vs

        # Preserve the lst_hours debugging column when LST is requested —
        # it's not in any standard feature set, but several offline tools
        # consume it. Cheap to compute vectorized.
        if 'lst' in self.base_features:
            _, df['lst_hours'] = calc_lst(df['datetime'].values)

        df['t_night'] = df.groupby('night')['timestamp'].transform(normalize_times, sun_el_limit=-10.0)
        return df

    def _add_moon_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Angular separation between pointing and Moon. Live env computes
        this per-step in compute_global_pointing_features; offline does it
        vectorized here using the moon_ra/moon_dec columns populated by
        _add_time_dependent_features."""
        df['moon_distance'] = calc_distance_to_moon(
            np.array([df['moon_ra'].values, df['moon_dec'].values]),
            np.array([df['ra'].values, df['dec'].values]),
        )
        return df

    def _map_bins_and_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps RA/Dec to field_id and bin number, using lookups and hpGrid if provided."""
        # df['field_id'] = df['field'].map({v: k for k, v in self.fid2name.items()})
        df['field_id'] = df['field'].map({v: k for k, v in self.lookups.fields['field'].to_dict().items()})

        if self.hpGrid is not None:
            lon = df['az'] if self.hpGrid.is_azel else df['ra']
            lat = df['el'] if self.hpGrid.is_azel else df['dec']

            df['bin'] = self.hpGrid.ang2idx(lon=lon, lat=lat)

            # Re-assign zenith specifics
            zenith_mask = df['field'] == 'zenith'
            df.loc[zenith_mask, "bin"] = ZENITH_BIN_NUM
            df.loc[zenith_mask, "field_id"] = ZENITH_FIELD_ID

        return df

    def _add_current_filter(self, df: pd.DataFrame):
        df['filter_wave'] = df['filter'].map(FILTER2WAVE)
        df['filter_wave'] = df['filter_wave'].fillna(ZENITH_WAVELENGTH) / FILTERWAVENORM # zenith "filter" set to 0, then normalize
        df['filter_idx'] = df['filter'].map(FILTER2IDX)
        for feat_name in self.base_features:
            if feat_name == 'is_filter':
                for filt in FILTER2IDX.keys():
                    df[f'{feat_name}_{filt}'] = (df['filter'] == filt).astype(np.float32)
        return df

    def _add_sky_brightness(self, df):
        """Vectorized per-filter sky brightness. Kept separate from the
        shared per-timestep helper because ``estimate_sky_brightness``
        accepts arrays — calling it five times with N-element arrays is
        much faster than calling it per row in the time-features loop.
        """
        # if any(('sky_brightness' in base_feat) for base_feat in self.base_features):
        for filt in FILTER2WAVE.keys():
            if filt != ZENITH_FILTER:
                if any(('sky_brightness' == base_feat) for base_feat in self.base_features):
                    df[f'sky_brightness_{filt}'] = estimate_sky_brightness(
                        time=df['timestamp'].values,
                        ra=df['ra'].values,
                        dec=df['dec'].values,
                        band=filt,
                    )
        return df

    def _add_global_mean_tiling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-row survey-completion scalars: `global_mean_tiling` and per-filter
        variants. Survey-agnostic — depends only on visit counts and targets.

        Maintains a running visit counter per night, seeded from the
        start-of-night snapshot in `lookups.night2fidfilt_visit_hist`. The
        counter is incremented row-by-row; `global_mean_tiling` values are computed
        BEFORE each row's increment so row i's value describes the state
        going into row i's action — same semantics as bin_features.

        Skips work entirely if no `global_mean_tiling*` feature is in the requested
        base_features list.
        """
        requested = [
            f for f in self.base_features
            if f == 'global_mean_tiling' or f.startswith('global_mean_tiling_')
        ]
        if not requested:
            return df

        if not hasattr(self.lookups, 'night2fidfilt_visit_hist'):
            raise AttributeError(
                "LookupTables is missing `night2fidfilt_visit_hist`; cannot "
                "compute global_mean_tiling. Rebuild lookups via build_train_lookups.py."
            )

        target_counts = self.lookups.target_fidfilt_counts
        nfields, nfilters = target_counts.shape

        # Per-row output buffers, indexed by df row order (df has a clean
        # 0..n-1 index after _add_zenith_rows's reset_index).
        out_overall = np.full(len(df), np.nan, dtype=np.float32)
        out_per_filt = {
            f: np.full(len(df), np.nan, dtype=np.float32)
            for f in FILTER2IDX.keys()
        }

        visit_hist_dict = self.lookups.night2fidfilt_visit_hist

        for night, group in df.groupby('night', sort=False):
            running = visit_hist_dict[night].copy().astype(np.int32)

            # Extract once per night for the inner loop.
            fids = group['field_id'].to_numpy(dtype=np.int64)
            # filter_idx is float64 with NaN for zenith rows; fillna(-1)
            # marks zenith so the inner check skips it.
            filt_idxs = (
                group['filter_idx'].fillna(-1).to_numpy(dtype=np.int64)
            )
            row_idxs = group.index.to_numpy(dtype=np.int64)

            for j in range(len(group)):
                mt = compute_global_mean_tiling_features(
                    running_counts=running, target_counts=target_counts,
                )
                out_overall[row_idxs[j]] = mt['global_mean_tiling']
                for f in FILTER2IDX.keys():
                    out_per_filt[f][row_idxs[j]] = mt[f'global_mean_tiling_{f}']

                # Increment for next row. Zenith rows (ZENITH_FIELD_ID or
                # zenith filter, which maps to -1 via fillna) don't count.
                # Note: no teff gating here — matches the bin_features.py
                # inner-loop semantics. If you ever add teff gating
                # consistently, do it in both places.
                fid_j, filt_j = fids[j], filt_idxs[j]
                if fid_j != ZENITH_FIELD_ID and 0 <= filt_j < nfilters:
                    running[fid_j, filt_j] += 1

        # Attach only the columns the user actually requested.
        if 'global_mean_tiling' in self.base_features:
            df['global_mean_tiling'] = out_overall
        for filt in FILTER2IDX.keys():
            if 'global_mean_tiling' in self.base_features:
                df[f'global_mean_tiling_{filt}'] = out_per_filt[filt]

        return df

    def _ensure_32_bit(self, df):
        for bin_str, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]):
            cols = df.select_dtypes(include=[bin_str]).columns
            df[cols] = df[cols].astype(np_bit)
        return df

    def _apply_cyclical_norms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cos/sin pairs for cyclical features via the shared helper.

        Uses the corrected matching rule (exact name OR ``_<cyc>`` suffix);
        the previous in-class version used a bare ``endswith`` that also
        matched ``rel_*`` features against their unprefixed cyclical roots.
        """
        apply_cyclical_features(
            df, self.base_features, self.cyclical_features
        )
        return df


# ============================================================================
# Other helpers (data-loading-time computations, not part of the shared
# per-timestep API).
# ============================================================================


def calc_t_survey(survey_night_indices, survey_nights_max):
    t_survey = survey_night_indices / survey_nights_max
    if type(t_survey) == torch.Tensor or type(t_survey) == np.ndarray:
        assert t_survey.min() >= 0 and t_survey.max() <= 1, "t_survey should be between 0 and 1"
    return t_survey

def calc_urgency(filter_counts_arr, filter_counts_max, survey_night_indices, survey_nights_max):
    survey_progress = filter_counts_arr / filter_counts_max
    t_survey = calc_t_survey(survey_night_indices, survey_nights_max)
    urgency = np.clip((1 - survey_progress) / (1 - t_survey + 1e-9), a_min=0.01, a_max=100.0)
    return urgency

def get_night_boundaries(
    anchor: Union[
    float, int, np.integer, np.floating,
    datetime, date, pd.Timestamp, str,
    pd.Series
    ],
    sun_el_limit: float = -10.5,
    observer_lon_rad: float = -70.8065,
) -> tuple[float, float]:
    """Compute (sunset_ts, sunrise_ts) UTC unix timestamps for the
    Blanco observing night identified by `anchor`.

    Args
    ----
    anchor : float | int | np.integer | np.floating | datetime | date | pd.Timestamp | str | pd.Series
        `anchor` accepts whatever is most natural at the call site:
        - **Unix timestamp** (numeric, including numpy scalars): a moment
            in time. The night is determined with a local-noon cutover —
            moments before local solar noon resolve to the night just
            ended, after to the upcoming. References inside an observing
            night always land on that night.
        - **Datetime / Timestamp / ISO string**: same — interpreted as a
            moment, reduced via the unix-timestamp path. (For
            `first_row["night"]` at 23:59:59 UTC, the moment is well
            inside the local-noon cutover region for its night, so this
            Just Works.)
        - **`date`**: treated as the canonical evening-date of the
            observing night and used directly, no cutover needed.
        - **`pd.Series`**: uses the first element. Convenient for
            `df.groupby('night').transform(...)` callbacks where every
            element of the group belongs to the same night by construction.
    Returns
    -------
    (sunset_ts, sunrise_ts)
    """
    noon_ts = _noon_anchor_from_input(anchor, observer_lon_rad)
    sunset_ts = calc_twilight(noon_ts, "set", sun_el_limit)
    sunrise_ts = calc_twilight(noon_ts, "rise", sun_el_limit)

    if not (sunset_ts < sunrise_ts):
        raise RuntimeError(
            f"Computed sunset ({sunset_ts}) does not precede sunrise "
            f"({sunrise_ts}) for sun_el_limit={sun_el_limit}. Check that "
            f"sun_el_limit isn't above the local horizon for this date "
            f"(no twilight crossing) and that calc_twilight returns "
            f"`next_*`-style boundaries."
        )
    return sunset_ts, sunrise_ts

def _noon_anchor_from_input(
    anchor: Union[
        float, int, np.integer, np.floating,
        datetime, date, pd.Timestamp, str,
        pd.Series,
        ],
    observer_lon_deg: float = -70.8065,
) -> float:
    """Resolve a flexible `anchor` to the unix timestamp of local solar
    noon on the corresponding date."""
    # Approximate UTC offset from longitude (15° per hour). Solar time,
    # not civil — accurate to within the equation of time (~16 min) and
    # free of timezone/DST handling. Sufficient because noon is ~6 h
    # from any twilight boundary.
    utc_offset_sec = observer_lon_deg * (3600.0 / 15.0)  # degrees to seconds

    # pd.Series: every element belongs to the same night by groupby
    # construction at all current call sites; use the first.
    if isinstance(anchor, pd.Series):
        if len(anchor) == 0:
            raise ValueError(
                "Cannot derive a night anchor from an empty Series."
            )
        anchor = anchor.iloc[0]

    # `date` but NOT `datetime` (datetime is a subclass of date in
    # the stdlib): treat as canonical evening-date label, skip cutover.
    if isinstance(anchor, date) and not isinstance(anchor, datetime):
        anchor_date = anchor
    else:
        # Everything else reduces to a unix timestamp.
        if isinstance(anchor, (int, float, np.integer, np.floating)):
            ts_unix = float(anchor)
        elif isinstance(anchor, pd.Timestamp):
            ts = (
                anchor if anchor.tzinfo is not None
                else anchor.tz_localize("UTC")
            )
            ts_unix = ts.timestamp()
        elif isinstance(anchor, datetime):
            ts = (
                anchor if anchor.tzinfo is not None
                else anchor.replace(tzinfo=timezone.utc)
            )
            ts_unix = ts.timestamp()
        elif isinstance(anchor, str):
            ts = pd.Timestamp(anchor)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts_unix = ts.timestamp()
        else:
            raise TypeError(
                f"Unsupported anchor type {type(anchor).__name__}; "
                f"expected unix timestamp, datetime, date, pd.Timestamp, "
                f"ISO string, or pd.Series of timestamps."
            )

        # Local-noon cutover: shift to local time, back 12 h, take date.
        local_ref = datetime.fromtimestamp(
            ts_unix + utc_offset_sec, tz=timezone.utc
        )
        anchor_date = (local_ref - timedelta(hours=12)).date()

    return (
        datetime(
            anchor_date.year, anchor_date.month, anchor_date.day,
            hour=12, tzinfo=timezone.utc,
        ).timestamp()
        - utc_offset_sec
    )

def calc_twilight(ts, event_type='set', horizon='-14', buffer_in_seconds=10):
    obs = ephemerides.blanco_observer(time=ts)
    obs.horizon = str(horizon)
    sun = ephem.Sun()
    sun.compute(obs)

    if event_type == 'rise':
        ephem_date = obs.next_rising(sun)
    elif event_type == 'set':
        ephem_date = obs.next_setting(sun)
    else:
        raise NotImplementedError(f"Unsupported event_type: {event_type}")

    dt_utc = ephem_date.datetime().replace(tzinfo=timezone.utc)
    if event_type == 'rise':
        dt_utc -= timedelta(seconds=buffer_in_seconds)
    else:
        dt_utc += timedelta(seconds=buffer_in_seconds)
    return dt_utc.timestamp()

def calc_sun_rise_and_set_times(df):
    rise_times = df.groupby('night').apply(calc_twilight, event_type='rise').values
    set_times = df.groupby('night').apply(calc_twilight, event_type='set').values
    return rise_times, set_times

def calc_sun_and_moon_positions(time):
    sun_radec = ephemerides.get_source_ra_dec('sun', time=time)
    sun_azel = ephemerides.equatorial_to_topographic(ra=sun_radec[0], dec=sun_radec[1], time=time)
    moon_radec = ephemerides.get_source_ra_dec('moon', time=time)
    moon_azel = ephemerides.equatorial_to_topographic(ra=moon_radec[0], dec=moon_radec[1], time=time)
    return sun_radec, sun_azel, moon_radec, moon_azel

def calc_distance_to_moon(moon_radec, pointing_radec):
    return angular_separation(moon_radec, pointing_radec)

def calc_moon_phase(time):
    observer = ephemerides.blanco_observer(time=time)
    moon = ephem.Moon()
    moon.compute(observer)
    moon_phase = moon.phase / 100
    return np.float32(moon_phase)

def calc_lst(datetime_np64):
    t_arr = Time(datetime_np64, format='datetime64', scale='utc')
    lst_obj = t_arr.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
    return lst_obj.radian, lst_obj.hour # for debugging

def merge_zenith_df(df, zenith_df):
    df = pd.concat([df, zenith_df], ignore_index=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

def get_zenith_features(original_df):
    """
    Constructs dataframe with zenith features for each night in the original_df.
    Assumes zenith starts 10 seconds before the first observation.
    """
    zenith_datetimes = original_df.groupby('night').head(1).datetime - pd.Timedelta(seconds=20)
    zenith_timestamps = (zenith_datetimes - pd.Timestamp("1970-01-01", tz='utc')) // pd.Timedelta("1s")
    zenith_datetimes = zenith_datetimes.values
    # zenith_timestamps = zenith_datetimes.astype(np.int64) // 10 ** 9
    # df['timestamp'] = timestamps
    zenith_rows = []
    nights = original_df.night.unique()
    for i_row, time in tqdm(enumerate(zenith_timestamps), total=len(zenith_timestamps), desc='Calculating zenith states'):
        row_dict = {}
        row_dict['timestamp'] = time
        row_dict['night'] = nights[i_row]
        row_dict['datetime'] = zenith_datetimes[i_row]
        blanco = ephemerides.blanco_observer(time=time)
        row_dict['ra'], row_dict['dec'] = np.array(blanco.radec_of('0',  '90'))
        zenith_rows.append(row_dict)

    zenith_df = pd.DataFrame(zenith_rows)
    zenith_df['az'] = ZENITH_AZ * units.deg
    zenith_df['el'] = ZENITH_EL * units.deg
    zenith_df['airmass'] = ZENITH_AIRMASS
    zenith_df['zd'] = ZENITH_ZD * units.deg
    zenith_df['ha'] = ZENITH_HA * units.deg
    zenith_df['field'] = ZENITH_OBJECT
    zenith_df['field_id'] = ZENITH_FIELD_ID
    zenith_df['filter'] = ZENITH_FILTER
    zenith_df['datetime'] = pd.to_datetime(zenith_df['datetime'], utc=True)
    # zenith_df['night'] = pd.to_datetime(zenith_df['night'], utc=True)

    return zenith_df

def _backfill_zenith_states(df):
    """Back fills zenith state for relevant features"""
    # df['fwhm'] = df.groupby('night')['fwhm'].bfill()
    # df['night_idx'] = df.groupby('night')['night_idx'].bfill()
    # df['t_survey'] = df.groupby('night')['t_survey'].bfill()
    # for f in FILTER2IDX.keys():
    #     df[f'raw_survey_progress_{f}'] = df.groupby('night')[f'raw_survey_progress_{f}'].bfill()
    #     df[f'survey_progress_{f}'] = df.groupby('night')[f'survey_progress_{f}'].bfill()
    #     df[f'urgency_{f}'] = df.groupby('night')[f'urgency_{f}'].bfill()
    return df


def normalize_times(time_series, sun_el_limit=-10):
    sunset_ts, sunrise_ts = get_night_boundaries(time_series, sun_el_limit)

    # sunset_ts = calc_twilight(time_series.median(), event_type='set')
    # sunrise_ts = calc_twilight(time_series.median(), event_type='rise')
    total_time = sunrise_ts - sunset_ts

    time_series = (time_series - sunset_ts) / total_time
    assert all(time_series.values > 0) and all(time_series.values < 1), "Time fractions should be between 0 and 1"
    return time_series


def calculate_sun_rise_and_set_azel(df):
    rise_times, set_times = calc_sun_rise_and_set_times(df)
    rise_azels = np.empty(shape=(len(set_times), 2))
    set_azels = np.empty(shape=(len(set_times), 2))

    for i, time in enumerate(rise_times):
        ra, dec = ephemerides.get_source_ra_dec('sun', time=time)
        sun_az, sun_el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=time)
        rise_azels[i] = np.array([sun_az, sun_el])
    for i, time in enumerate(set_times):
        ra, dec = ephemerides.get_source_ra_dec('sun', time=time)
        sun_az, sun_el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=time)
        set_azels[i] = np.array([sun_az, sun_el])

    return rise_azels, set_azels

def compute_causal_fwhm(night_df, seeing_cfg) -> np.ndarray:
    """Per-row causal predicted FWHM (arcsec) for one night.

    Builds a Seeing from seeing_cfg, ingests the night's measured
    (timestamp, fwhm, band, el) rows, then predicts a canonical i-band
    zenith FWHM at each row's timestamp using strictly-past history. The
    per-row band/el enter only through ingestion; prediction is pinned to
    i-band zenith to match PredictiveSeeingModel.fwhm and the live scheduler,
    so the feature reflects time-evolving atmospheric seeing rather than the
    expert's chosen airmass. Rows with a NaN measured fwhm are not ingested;
    predictions for them still resolve from prior history (or the nominal
    fallback when none exists).

    Parameters
    ----------
        night_df : pd.DataFrame
        One night's rows with columns 'timestamp', 'fwhm', 'filter_idx',
        'el'. Row order is preserved in the output.
    seeing_cfg : SeeingConfig
        Predictor parameters.

    Returns
    -------
    np.ndarray
        Predicted FWHM in plain arcsec, shape (len(night_df),).
    """
    seeing = Seeing(
        window=seeing_cfg.window,
        retention_window=seeing_cfg.retention_window,
        from_instrument=seeing_cfg.from_instrument * units.arcsec,
        to_instrument=seeing_cfg.to_instrument * units.arcsec,
    )

    ts = night_df['timestamp'].to_numpy(dtype=float)
    fwhm_meas = night_df['fwhm'].to_numpy(dtype=float)
    filt_idx = night_df['filter_idx'].to_numpy(dtype=float)
    el = night_df['el'].to_numpy(dtype=float)

    # Zenith/wait rows carry a NaN filter_idx; fall back to the reference
    # band so int() never sees a NaN (only the airmass term then matters).
    bands = [
        FWHM_REF_FILTER if np.isnan(f) else IDX2FILTER.get(int(f), FWHM_REF_FILTER)
        for f in filt_idx
    ]

    valid = ~np.isnan(fwhm_meas)
    if valid.any():
        seeing.add(
            date=ts[valid],
            seeing=fwhm_meas[valid] * units.arcsec,
            band=[b for b, v in zip(bands, valid) if v],
            el=el[valid],
        )

    # Only warn when all rows in night are have no seeing history
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No seeing history available",
            category=RuntimeWarning,
        )
        out = np.empty(len(night_df), dtype=float)
        for i in range(len(night_df)):
            out[i] = float(
                seeing.predict(band="i", el=np.pi / 2, now=float(ts[i]))
            ) / units.arcsec

    if not valid.any():
        night_id = (
            night_df['night'].iloc[0] if 'night' in night_df.columns else "unknown"
        )
        warnings.warn(
            f"Night {night_id}: no measured seeing available; all "
            f"{len(night_df)} rows fall back to the nominal DECam median.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


