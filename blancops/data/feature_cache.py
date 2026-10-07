"""Precomputed feature cache for the offline RL pipeline.

Two dataclasses are defined here:

- ``BinFeatureCache``: stores all raw (unnormalized) HEALPix bin features for
  every observation in the training dataset, independent of experiment config.
  Computed once by ``precompute-features`` and shared across training runs.

- ``FieldFeatureCache``: the same for survey-field candidates (field_filter).

The per-run, normalized snapshot of one split is ``TransitionDatasetCache`` in ``data/dataset.py``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from blancops.configs.constants import (
    _BIN_FEATURES,
    _CYCLICAL_FEATURE_NAMES,
    _GLOBAL_FEATURES,
    ZENITH_FIELD_ID,
)
from blancops.configs.enums import AcceptanceRule
from blancops.data.features.bin_features import BinFeatureEngineer
from blancops.data.features.glob_features import GlobalFeatureEngineer
from blancops.math import geometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (mirrored from dataset.py, kept local to avoid circular
# imports — dataset.py will be the caller, not the callee)
# ---------------------------------------------------------------------------

def _get_state_indices(df: pd.DataFrame, max_time_diff_min: int = 5, label: str = ''):
    """Return transition index arrays from a timestamp-sorted DataFrame.

    Returns (state_idxs, current_state_idxs, next_state_idxs, df_idx_to_compact).
    All indices are relative to ``df`` (i.e. valid for ``df.iloc[...]``).
    """
    time_diffs = df['timestamp'].diff().values
    keep = time_diffs < max_time_diff_min * 60 + 90
    next_state_idxs = np.where(keep)[0]
    current_state_idxs = next_state_idxs - 1
    state_idxs = np.unique(np.concatenate([current_state_idxs, next_state_idxs]))
    df_idx_to_compact = {int(idx): i for i, idx in enumerate(state_idxs)}
    n_removed = int(np.sum(~keep))
    prefix = f"{label} " if label else ""
    logger.info(
        f"Removing {n_removed} {prefix}transitions with time diff > {max_time_diff_min} min. "
        f"Total {prefix}transitions: {len(next_state_idxs)}"
    )
    return state_idxs, current_state_idxs, next_state_idxs, df_idx_to_compact


def _bin_slew_distances(df, current_state_idxs, next_state_idxs, hpGrid):
    """Angular slew distance per transition between HEALPix bin centers (radians), as a float32 array."""
    from blancops.ephemerides import ephemerides as _eph

    curr_bids = df.iloc[current_state_idxs]['bin'].values.copy()
    next_bids = df.iloc[next_state_idxs]['bin'].values.copy()
    z_mask = curr_bids == -1

    if hpGrid.is_azel:
        curr_bids[z_mask] = hpGrid.ang2idx(lon=0, lat=np.pi / 2)
    else:
        z_idxs = np.where(z_mask)[0]
        z_df_idxs = current_state_idxs[z_idxs]
        z_timestamps = df.iloc[z_df_idxs]['timestamp'].values
        for i, t in zip(z_idxs, z_timestamps):
            z_ra, z_dec = _eph.topographic_to_equatorial(az=0, el=np.pi / 2, time=t)
            curr_bids[i] = hpGrid.ang2idx(lon=z_ra, lat=z_dec)

    curr_coords = np.array((hpGrid.lon[curr_bids], hpGrid.lat[curr_bids]))
    next_coords = np.array((hpGrid.lon[next_bids], hpGrid.lat[next_bids]))
    return geometry.angular_separation(curr_coords, next_coords).astype(np.float32)


def _nights_in_date_range(night_dts, start_date, end_date) -> List[str]:
    """Filter night datetimes to an inclusive date range.

    Args:
        night_dts: DatetimeIndex of unique nights.
        start_date: Inclusive lower bound (``'YYYY-MM-DD'``), or None.
        end_date: Inclusive upper bound (``'YYYY-MM-DD'``), or None.

    Returns:
        The retained nights as ``'YYYY-MM-DD'`` strings.
    """
    mask = np.ones(len(night_dts), dtype=bool)
    if start_date is not None:
        mask &= night_dts >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= night_dts <= pd.to_datetime(end_date)
    return night_dts[mask].strftime('%Y-%m-%d').tolist()


# ---------------------------------------------------------------------------
# BinFeatureCache -- HEALPix bin candidates
# ---------------------------------------------------------------------------

@dataclass
class BinFeatureCache:
    """All raw (unnormalized) HEALPix bin features for a dataset, independent of training config.

    Computed once from a FITS file by ``precompute-features`` and reused
    across training runs that differ only in normalization scheme, reward
    type, or feature subset.

    Disk layout (under ``cache_dir/``):

        metadata.json: nside, is_azel, feature name lists, n_rows, n_bins, acceptance
        global_df.parquet: enriched DataFrame with ALL global feature columns
        bin_features.npy: (n_rows, n_bins, n_bin_feats) float32; memmap-friendly
        transitions.npz: compressed arrays: state_idxs, current_state_idxs,
                              next_state_idxs, slew_distances
        interruptions.parquet: optional; survey exposures (by expnum) interrupted by other
                              archived exposures, with the interrupting pointing and filter
    """

    INTERRUPTIONS_FILE = 'interruptions.parquet'

    nside: int
    is_azel: bool

    # Enriched DataFrame: FITS columns + ALL global features (cyclical-expanded)
    global_df: pd.DataFrame
    global_feature_names: List[str]

    # (n_rows, n_bins, n_bin_feats) float32 — ALL _BIN_FEATURES
    bin_features: np.ndarray
    bin_feature_names: List[str]

    # Transition structure (timestamps → max_time_diff_min=5 filter), local indices
    state_idxs: np.ndarray
    current_state_idxs: np.ndarray
    next_state_idxs: np.ndarray
    slew_distances: np.ndarray  # (n_transitions,) float32

    # Acceptance rule of the lookups the features were computed from
    acceptance: AcceptanceRule

    # Interrupted survey exposures keyed by expnum; None when the cache has no interruptions file
    interruptions: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def compute(cls, df: pd.DataFrame, lookups, hpGrid) -> 'BinFeatureCache':
        """Build a ``BinFeatureCache`` from a raw observation DataFrame.

        Runs ``GlobalFeatureEngineer`` with *all* ``_GLOBAL_FEATURES`` and
        ``BinFeatureEngineer`` with *all* ``_BIN_FEATURES``.
        """
        from blancops.data.features.normalizations import expand_feature_set

        # Determine action_space string from hpGrid to satisfy BinFeatureEngineer
        action_space = 'azel_filter' if hpGrid.is_azel else 'radec_filter'

        logger.info("Running GlobalFeatureEngineer on all global features…")
        glob_eng = GlobalFeatureEngineer(
            lookups=lookups,
            hpGrid=hpGrid,
            base_features=_GLOBAL_FEATURES,
            cyclical_features=_CYCLICAL_FEATURE_NAMES,
            do_cyclical_norm=True,
            do_filt=True,
        )
        enriched_df = glob_eng.transform(df)

        # Collect expanded global feature names (post cyclical expansion)
        global_feature_names = expand_feature_set(
            _GLOBAL_FEATURES, _CYCLICAL_FEATURE_NAMES, do_filt=True, survey=lookups.survey
        )
        # Keep only columns that actually exist in the enriched df
        global_feature_names = [f for f in global_feature_names if f in enriched_df.columns]

        # Expand bin feature names the same way dataset.py does via setup_feature_names
        bin_feature_names = expand_feature_set(
            list(_BIN_FEATURES), _CYCLICAL_FEATURE_NAMES, do_filt=True, survey=lookups.survey
        )

        logger.info("Running BinFeatureEngineer on all bin features…")
        do_local_mean_z = any('rel_' in f for f in _BIN_FEATURES)
        bin_eng = BinFeatureEngineer(
            hpGrid=hpGrid,
            base_features=list(_BIN_FEATURES),
            cyclical_features=_CYCLICAL_FEATURE_NAMES,
            action_space=action_space,
            lookups=lookups,
            do_cyclical_norm=True,
            do_local_mean_z_score=do_local_mean_z,
        )
        # requested_features must be the expanded names (post cyclical/filter expansion)
        bin_features_raw = bin_eng.transform(enriched_df, requested_features=bin_feature_names)

        # Sanity-check alignment; the array dim is authoritative
        n_bin_feats = bin_features_raw.shape[2]
        if len(bin_feature_names) != n_bin_feats:
            logger.warning(
                f"bin_feature_names length {len(bin_feature_names)} != "
                f"array dim {n_bin_feats}; truncating to array length."
            )
            bin_feature_names = bin_feature_names[:n_bin_feats]

        logger.info("Computing transition indices…")
        state_idxs, current_state_idxs, next_state_idxs, _ = _get_state_indices(enriched_df)

        logger.info("Computing slew distances…")
        slew_distances = _bin_slew_distances(
            enriched_df, current_state_idxs, next_state_idxs, hpGrid
        )

        return cls(
            nside=hpGrid.nside,
            is_azel=hpGrid.is_azel,
            global_df=enriched_df,
            global_feature_names=global_feature_names,
            bin_features=bin_features_raw.astype(np.float32),
            bin_feature_names=bin_feature_names,
            state_idxs=state_idxs,
            current_state_idxs=current_state_idxs,
            next_state_idxs=next_state_idxs,
            slew_distances=slew_distances,
            acceptance=lookups.acceptance,
        )

    # ------------------------------------------------------------------
    # Night filtering
    # ------------------------------------------------------------------

    def log_transition_filter_stats(self, nights, label: str = '') -> None:
        """Log how many transitions would be removed by the time-diff filter for ``nights``."""
        nights_set = {str(n) for n in nights}
        mask = self.global_df['night'].astype(str).isin(nights_set)
        filtered_df = self.global_df[mask].reset_index(drop=True)
        _get_state_indices(filtered_df, label=label)

    def filter_nights(self, nights, label: str = '') -> 'BinFeatureCache':
        """Return a new ``BinFeatureCache`` restricted to ``nights``.

        All indices in the returned cache are **local** to the filtered
        DataFrame (0-based), so downstream code sees a self-contained,
        smaller dataset.
        """
        nights_set = set(str(n) for n in nights)
        mask = self.global_df['night'].astype(str).isin(nights_set)
        filtered_df = self.global_df[mask].reset_index(drop=True)

        if len(filtered_df) == 0:
            raise ValueError(f"filter_nights: no rows matched nights {nights_set}")

        # Original row positions (into self.global_df) for the filtered rows
        orig_positions = np.where(mask.values)[0]

        # Re-derive transition indices from the filtered df's timestamps
        state_idxs, current_state_idxs, next_state_idxs, _ = _get_state_indices(filtered_df, label=label)

        # Slice bin_features using original positions then re-index to state_idxs
        bin_subset = self.bin_features[orig_positions]  # (n_filtered, n_bins, n_feats)

        from blancops.ephemerides import ephemerides as _eph
        hpGrid = _eph.HealpixGrid(nside=self.nside, is_azel=self.is_azel)
        slew_distances = _bin_slew_distances(
            filtered_df, current_state_idxs, next_state_idxs, hpGrid
        )

        return BinFeatureCache(
            nside=self.nside,
            is_azel=self.is_azel,
            global_df=filtered_df,
            global_feature_names=self.global_feature_names,
            bin_features=bin_subset,
            bin_feature_names=self.bin_feature_names,
            state_idxs=state_idxs,
            current_state_idxs=current_state_idxs,
            next_state_idxs=next_state_idxs,
            slew_distances=slew_distances,
            acceptance=self.acceptance,
            interruptions=self.interruptions,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. metadata.json
        meta = {
            'nside': self.nside,
            'is_azel': self.is_azel,
            'global_feature_names': self.global_feature_names,
            'bin_feature_names': self.bin_feature_names,
            'n_rows': int(len(self.global_df)),
            'n_bins': int(self.bin_features.shape[1]),
            'acceptance': self.acceptance.value,
        }
        with open(cache_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        # 2. global_df.parquet (columnar, snappy-compressed)
        self.global_df.to_parquet(cache_dir / 'global_df.parquet', index=True)

        # 3. bin_features.npy (uncompressed; enables np.load(mmap_mode='r'))
        np.save(cache_dir / 'bin_features.npy', self.bin_features)

        # 4. transitions.npz (compressed; small index arrays)
        np.savez_compressed(
            cache_dir / 'transitions.npz',
            state_idxs=self.state_idxs,
            current_state_idxs=self.current_state_idxs,
            next_state_idxs=self.next_state_idxs,
            slew_distances=self.slew_distances,
        )
        if self.interruptions is not None:
            self.save_interruptions(cache_dir, self.interruptions)
        logger.info(f"BinFeatureCache saved to {cache_dir}")

    @classmethod
    def save_interruptions(cls, cache_dir: Path, interruptions: pd.DataFrame) -> None:
        """Write the interruptions file into a cache directory, alongside an existing cache.

        Parameters
        ----------
        cache_dir : Path
            Cache directory.
        interruptions : pd.DataFrame
            Output of ``preprocessing.find_interruptions``.
        """
        interruptions.to_parquet(Path(cache_dir) / cls.INTERRUPTIONS_FILE, index=False)

    @classmethod
    def load(cls, cache_dir: Path, mmap_bin: bool = False,
             start_date: str | None = None,
             end_date: str | None = None,
             acceptance: AcceptanceRule | str | None = None) -> 'BinFeatureCache':
        """Load from disk.

        Args:
            cache_dir:   Directory written by ``save()``.
            mmap_bin:    If True, ``bin_features`` is memory-mapped (read-only).
            start_date:  Inclusive lower bound on night (``'YYYY-MM-DD'``).
            end_date:    Inclusive upper bound on night (``'YYYY-MM-DD'``).
            acceptance:  Rule the caller expects the cache to be built with; None skips the check.
        """
        cache_dir = Path(cache_dir)

        with open(cache_dir / 'metadata.json') as f:
            meta = json.load(f)
        accept_rule = AcceptanceRule(meta['acceptance'])
        if acceptance is not None:
            accept_rule.require(acceptance, cache_dir)
        logger.info(f"Loading BinFeatureCache metadata from {cache_dir}")

        global_df = pd.read_parquet(cache_dir / 'global_df.parquet')

        mmap_mode = 'r' if mmap_bin else None
        bin_features = np.load(cache_dir / 'bin_features.npy', mmap_mode=mmap_mode)
        logger.info(f"Loading BinFeatureCache bin_features from {cache_dir}")

        t = np.load(cache_dir / 'transitions.npz')
        logger.info(f"Loading BinFeatureCache transitions from {cache_dir}")

        cache = cls(
            nside=meta['nside'],
            is_azel=meta['is_azel'],
            global_df=global_df,
            global_feature_names=meta['global_feature_names'],
            bin_features=bin_features,
            bin_feature_names=meta['bin_feature_names'],
            state_idxs=t['state_idxs'],
            current_state_idxs=t['current_state_idxs'],
            next_state_idxs=t['next_state_idxs'],
            slew_distances=t['slew_distances'],
            acceptance=accept_rule,
            interruptions=(pd.read_parquet(cache_dir / cls.INTERRUPTIONS_FILE)
                           if (cache_dir / cls.INTERRUPTIONS_FILE).exists() else None),
        )

        if start_date is not None or end_date is not None:
            all_nights = pd.to_datetime(cache.global_df['night'].unique())
            filtered_nights = _nights_in_date_range(all_nights, start_date, end_date)
            # Filtering copies bin_features out of the memmap, so skip it when
            # the range already covers every cached night.
            if len(filtered_nights) < len(all_nights):
                logger.info(
                    f"Filtering cache to {len(filtered_nights)} nights "
                    f"({start_date} to {end_date})"
                )
                cache = cache.filter_nights(filtered_nights, label='date range')
            else:
                logger.info(
                    f"Date range ({start_date} to {end_date}) covers all "
                    f"{len(all_nights)} cached nights; skipping filter."
                )

        return cache

    @classmethod
    def nights_in_range(cls, cache_dir: Path,
                        start_date: str | None = None,
                        end_date: str | None = None) -> List[str]:
        """Night strings held by a cache, reading only the night column.

        Skips bin_features and transitions entirely so callers that just need
        the night list do not materialize the feature arrays.

        Args:
            cache_dir: Directory written by ``save()``.
            start_date: Inclusive lower bound on night (``'YYYY-MM-DD'``).
            end_date: Inclusive upper bound on night (``'YYYY-MM-DD'``).

        Returns:
            The night strings within the date range, in ascending order.
        """
        nights = pd.read_parquet(
            Path(cache_dir) / 'global_df.parquet', columns=['night']
        )['night']
        return _nights_in_date_range(pd.to_datetime(nights.unique()), start_date, end_date)

    @classmethod
    def exists(cls, cache_dir: Path) -> bool:
        cache_dir = Path(cache_dir)
        return all(
            (cache_dir / f).exists()
            for f in ('metadata.json', 'global_df.parquet', 'bin_features.npy', 'transitions.npz')
        )


# ---------------------------------------------------------------------------
# FieldFeatureCache -- field-level candidates
# ---------------------------------------------------------------------------

def _field_slew_distances(df: pd.DataFrame, current_state_idxs, next_state_idxs, ra, dec) -> np.ndarray:
    """Angular slew distance per transition between field centers (radians); zenith rows use the zenith then.

    Parameters
    ----------
    df : pd.DataFrame
        Global frame with field_id and timestamp.
    current_state_idxs, next_state_idxs : np.ndarray
        Transition row indices.
    ra, dec : np.ndarray
        Field centers in radians, indexed by field_id.

    Returns
    -------
    np.ndarray
        float32 distances.
    """
    from blancops.ephemerides import ephemerides as _eph

    def centers(rows):
        fids = df['field_id'].to_numpy()[rows].astype(int)
        lon, lat = ra[np.clip(fids, 0, None)].copy(), dec[np.clip(fids, 0, None)].copy()
        for i in np.where(fids == ZENITH_FIELD_ID)[0]:
            lon[i], lat[i] = _eph.blanco_observer(time=float(df['timestamp'].iloc[rows[i]])).radec_of('0', '90')
        return np.array((lon, lat))

    return geometry.angular_separation(centers(current_state_idxs), centers(next_state_idxs)).astype(np.float32)


@dataclass
class FieldFeatureCache:
    """Raw (unnormalized) field-level features for the field_filter action space, independent of any HEALPix grid.

    Disk layout (under ``cache_dir/``):

        metadata.json: global and field feature names, n_rows, n_fields, acceptance
        global_df.parquet: enriched DataFrame with all global feature columns (no bin column)
        field_features.npy: (n_rows, n_fields, n_field_feats) float32; memmap-friendly
        field_tiling.npz: per-row global_mean_tiling including the row's own exposure (overall and per filter)
        transitions.npz: state_idxs, current_state_idxs, next_state_idxs, slew_distances (field centers)
        interruptions.parquet: survey exposures (by expnum) interrupted by other archived exposures
    """

    FIELD_FEATURES_FILE = 'field_features.npy'
    FIELD_TILING_FILE = 'field_tiling.npz'
    INTERRUPTIONS_FILE = 'interruptions.parquet'
    _FILES = ('metadata.json', 'global_df.parquet', FIELD_FEATURES_FILE, FIELD_TILING_FILE, 'transitions.npz')

    global_df: pd.DataFrame
    global_feature_names: List[str]
    field_features: np.ndarray
    field_feature_names: List[str]
    field_tiling: dict
    state_idxs: np.ndarray
    current_state_idxs: np.ndarray
    next_state_idxs: np.ndarray
    slew_distances: np.ndarray
    acceptance: AcceptanceRule
    interruptions: Optional[pd.DataFrame] = None

    @classmethod
    def compute(cls, cache_dir: Path, df: pd.DataFrame, lookups, base_features: list[str],
                interruptions: Optional[pd.DataFrame] = None) -> None:
        """Build the cache from a processed survey DataFrame and write it to ``cache_dir``.

        Field features are filled into a temporary disk memmap that gets its real name only when complete.

        Parameters
        ----------
        cache_dir : Path
            Output directory.
        df : pd.DataFrame
            Survey exposures from ``load_and_process_historic_data``.
        lookups : LookupTables
            Survey lookups.
        base_features : list of str
            Field feature base names.
        interruptions : pd.DataFrame or None
            Output of ``preprocessing.find_interruptions``.
        """
        from blancops.data.features.field_features import FieldFeatureEngineer
        from blancops.data.features.normalizations import expand_feature_set

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        glob_eng = GlobalFeatureEngineer(
            lookups=lookups, hpGrid=None, base_features=_GLOBAL_FEATURES,
            cyclical_features=_CYCLICAL_FEATURE_NAMES, do_cyclical_norm=True, do_filt=True,
        )
        global_df = glob_eng.transform(df)
        global_feature_names = [f for f in expand_feature_set(_GLOBAL_FEATURES, _CYCLICAL_FEATURE_NAMES, do_filt=True,
                                                             survey=lookups.survey)
                                if f in global_df.columns]

        eng = FieldFeatureEngineer(lookups, base_features)
        tmp_path = cache_dir / f"{cls.FIELD_FEATURES_FILE}.tmp"
        out = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.float32,
                                        shape=(len(global_df), len(eng.grid.lon), len(eng.feature_names)))
        _, tiling = eng.transform(global_df, out=out)
        out.flush()
        del out
        tmp_path.replace(cache_dir / cls.FIELD_FEATURES_FILE)

        state_idxs, current_state_idxs, next_state_idxs, _ = _get_state_indices(global_df)
        slew = _field_slew_distances(global_df, current_state_idxs, next_state_idxs, eng.grid.lon, eng.grid.lat)
        global_df.to_parquet(cache_dir / 'global_df.parquet', index=True)
        np.savez_compressed(cache_dir / cls.FIELD_TILING_FILE, **tiling)
        np.savez_compressed(cache_dir / 'transitions.npz', state_idxs=state_idxs,
                            current_state_idxs=current_state_idxs, next_state_idxs=next_state_idxs,
                            slew_distances=slew)
        if interruptions is not None:
            interruptions.to_parquet(cache_dir / cls.INTERRUPTIONS_FILE, index=False)
        with open(cache_dir / 'metadata.json', 'w') as f:
            json.dump({'global_feature_names': global_feature_names, 'field_feature_names': eng.feature_names,
                       'n_rows': int(len(global_df)), 'n_fields': int(len(eng.grid.lon)),
                       'acceptance': lookups.acceptance.value}, f, indent=2)
        logger.info(f"FieldFeatureCache saved to {cache_dir}")

    @classmethod
    def exists(cls, cache_dir: Path) -> bool:
        """Whether ``cache_dir`` holds a complete field feature cache."""
        return all((Path(cache_dir) / f).exists() for f in cls._FILES)

    @classmethod
    def load(cls, cache_dir: Path, mmap: bool = True, start_date: str | None = None,
             end_date: str | None = None,
             acceptance: AcceptanceRule | str | None = None) -> 'FieldFeatureCache':
        """Load from disk, optionally restricted to a date range.

        Parameters
        ----------
        cache_dir : Path
            Directory written by ``compute``.
        mmap : bool
            Memory-map the field features (read-only).
        start_date, end_date : str or None
            Inclusive night bounds ('YYYY-MM-DD').
        acceptance : AcceptanceRule, str, or None
            Rule the caller expects the cache to be built with; None skips the check.

        Returns
        -------
        FieldFeatureCache
            The cache.
        """
        cache_dir = Path(cache_dir)
        with open(cache_dir / 'metadata.json') as f:
            meta = json.load(f)
        accept_rule = AcceptanceRule(meta['acceptance'])
        if acceptance is not None:
            accept_rule.require(acceptance, cache_dir)
        t = np.load(cache_dir / 'transitions.npz')
        interruptions_path = cache_dir / cls.INTERRUPTIONS_FILE
        cache = cls(
            global_df=pd.read_parquet(cache_dir / 'global_df.parquet'),
            global_feature_names=meta['global_feature_names'],
            field_features=np.load(cache_dir / cls.FIELD_FEATURES_FILE, mmap_mode='r' if mmap else None),
            field_feature_names=meta['field_feature_names'],
            field_tiling=dict(np.load(cache_dir / cls.FIELD_TILING_FILE)),
            state_idxs=t['state_idxs'], current_state_idxs=t['current_state_idxs'],
            next_state_idxs=t['next_state_idxs'], slew_distances=t['slew_distances'],
            acceptance=accept_rule, interruptions=pd.read_parquet(interruptions_path) if interruptions_path.exists() else None,
        )
        if start_date is not None or end_date is not None:
            all_nights = pd.to_datetime(cache.global_df['night'].unique())
            nights = _nights_in_date_range(all_nights, start_date, end_date)
            if len(nights) < len(all_nights):
                cache = cache.filter_nights(nights, label='date range')
        return cache

    nights_in_range = BinFeatureCache.nights_in_range

    def log_transition_filter_stats(self, nights, label: str = '') -> None:
        """Log how many transitions the time-diff filter removes for ``nights``."""
        mask = self.global_df['night'].astype(str).isin({str(n) for n in nights})
        _get_state_indices(self.global_df[mask].reset_index(drop=True), label=label)

    def filter_nights(self, nights, label: str = '') -> 'FieldFeatureCache':
        """A new cache restricted to ``nights``, with indices local to the filtered frame.

        Parameters
        ----------
        nights : iterable
            Night strings to keep.
        label : str
            Log label.

        Returns
        -------
        FieldFeatureCache
            The filtered cache.
        """
        mask = self.global_df['night'].astype(str).isin({str(n) for n in nights})
        filtered_df = self.global_df[mask].reset_index(drop=True)
        if len(filtered_df) == 0:
            raise ValueError(f"filter_nights: no rows matched nights {set(nights)}")
        pos = np.where(mask.values)[0]
        state_idxs, current_state_idxs, next_state_idxs, _ = _get_state_indices(filtered_df, label=label)
        # Slews are a function of the two rows, so recompute them on the filtered transitions.
        full_next = dict(zip(self.next_state_idxs.tolist(), self.slew_distances.tolist()))
        slew = np.array([full_next[int(pos[i])] for i in next_state_idxs], dtype=np.float32)
        return FieldFeatureCache(
            global_df=filtered_df, global_feature_names=self.global_feature_names,
            field_features=self.field_features[pos], field_feature_names=self.field_feature_names,
            field_tiling={k: v[pos] for k, v in self.field_tiling.items()},
            state_idxs=state_idxs, current_state_idxs=current_state_idxs, next_state_idxs=next_state_idxs,
            slew_distances=slew, acceptance=self.acceptance, interruptions=self.interruptions,
        )
