"""Precomputed feature cache for the offline RL pipeline.

Two dataclasses are defined here:

- ``RawFeatureCache``: stores all raw (unnormalized) features for every
  observation in the training dataset, independent of experiment config.
  Computed once by ``precompute-features`` and shared across training runs.

- ``DatasetCache``: stores normalized tensors for one split's nights (val or
  test), built after a training run has fixed the split and normalization
  stats. Loaded by the evaluation pipeline to avoid re-processing on repeated
  runs. ``ValDatasetCache`` remains as a backwards-compatible alias.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from blancops.configs.constants import (
    _BIN_FEATURES,
    _CYCLICAL_FEATURE_NAMES,
    _GLOBAL_FEATURES,
)
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


def _compute_slew_distances(df, current_state_idxs, next_state_idxs, hpGrid):
    """Angular slew distance for each transition (radians), as a float32 array."""
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
# RawFeatureCache
# ---------------------------------------------------------------------------

@dataclass
class RawFeatureCache:
    """All raw (unnormalized) features for a dataset, independent of training config.

    Computed once from a FITS file by ``precompute-features`` and reused
    across training runs that differ only in normalization scheme, reward
    type, or feature subset.

    Disk layout (under ``cache_dir/``):

        metadata.json: nside, is_azel, feature name lists, n_rows, n_bins
        global_df.parquet: enriched DataFrame with ALL global feature columns
        bin_features.npy: (n_rows, n_bins, n_bin_feats) float32; memmap-friendly
        transitions.npz: compressed arrays: state_idxs, current_state_idxs,
                              next_state_idxs, slew_distances
    """

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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def compute(cls, df: pd.DataFrame, lookups, hpGrid) -> 'RawFeatureCache':
        """Build a ``RawFeatureCache`` from a raw observation DataFrame.

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
            _GLOBAL_FEATURES, _CYCLICAL_FEATURE_NAMES, do_filt=True
        )
        # Keep only columns that actually exist in the enriched df
        global_feature_names = [f for f in global_feature_names if f in enriched_df.columns]

        # Expand bin feature names the same way dataset.py does via setup_feature_names
        bin_feature_names = expand_feature_set(
            list(_BIN_FEATURES), _CYCLICAL_FEATURE_NAMES, do_filt=True
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
        slew_distances = _compute_slew_distances(
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

    def filter_nights(self, nights, label: str = '') -> 'RawFeatureCache':
        """Return a new ``RawFeatureCache`` restricted to ``nights``.

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
        slew_distances = _compute_slew_distances(
            filtered_df, current_state_idxs, next_state_idxs, hpGrid
        )

        return RawFeatureCache(
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
        logger.info(f"RawFeatureCache saved to {cache_dir}")

    @classmethod
    def load(cls, cache_dir: Path, mmap_bin: bool = False,
             start_date: str | None = None,
             end_date: str | None = None) -> 'RawFeatureCache':
        """Load from disk.

        Args:
            cache_dir:   Directory written by ``save()``.
            mmap_bin:    If True, ``bin_features`` is memory-mapped (read-only).
            start_date:  Inclusive lower bound on night (``'YYYY-MM-DD'``).
            end_date:    Inclusive upper bound on night (``'YYYY-MM-DD'``).
        """
        cache_dir = Path(cache_dir)

        with open(cache_dir / 'metadata.json') as f:
            meta = json.load(f)
        logger.info(f"Loading RawFeatureCache metadata from {cache_dir}")

        global_df = pd.read_parquet(cache_dir / 'global_df.parquet')

        mmap_mode = 'r' if mmap_bin else None
        bin_features = np.load(cache_dir / 'bin_features.npy', mmap_mode=mmap_mode)
        logger.info(f"Loading RawFeatureCache bin_features from {cache_dir}")

        t = np.load(cache_dir / 'transitions.npz')
        logger.info(f"Loading RawFeatureCache transitions from {cache_dir}")

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
# DatasetCache
# ---------------------------------------------------------------------------

@dataclass
class DatasetCache:
    """Post-normalization tensors for one split's nights.

    Built after a training run fixes the night split and normalization stats.
    Saved as ``outdir/checkpoints/<split>_dataset_cache.pt`` (``torch.save``).

    Exposes the same attributes queried by the evaluator infrastructure
    (``DataContainer``, ``SingleStepEvaluator``) so it can be used as a
    drop-in replacement for ``TransitionDataset`` in those paths.
    """

    # Normalized state tensors (val states only)
    states: torch.Tensor
    bin_states: Optional[torch.Tensor]  # None when no bin features
    action_masks: torch.Tensor
    active_bin_mask: Optional[torch.Tensor]  # None when no bin features

    # Per-transition tensors
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    slew_distances: torch.Tensor

    # Compact indices into val-state tensors
    curr_compact_idxs: np.ndarray
    next_compact_idxs: np.ndarray

    # Original (local-to-val-df) state indices — needed by DataContainer for
    # night-boundary detection and _df.iloc[] access
    current_state_idxs: np.ndarray
    next_state_idxs: np.ndarray
    state_idxs: np.ndarray

    # Split-night DataFrame (all enriched columns, split nights only, local index)
    split_df: pd.DataFrame

    # Metadata
    global_feature_names: List[str]
    bin_feature_names: List[str]
    dataset_dims: dict
    split_nights: List[str]
    nside: int
    is_azel: bool
    split: str = 'val'

    # ------------------------------------------------------------------
    # Properties for evaluator compatibility
    # ------------------------------------------------------------------

    @property
    def _df(self) -> pd.DataFrame:
        return self.split_df

    @property
    def val_df(self) -> pd.DataFrame:
        return self.split_df

    @property
    def val_nights(self) -> List[str]:
        return self.split_nights

    @property
    def unique_nights(self):
        return self.split_df['night'].unique()

    @property
    def _prenorm_bin_states(self) -> Optional[torch.Tensor]:
        # In TransitionDataset the prenorm array is normalized in-place, so
        # _prenorm_bin_states IS the normalized bin_states after __init__.
        return self.bin_states

    @property
    def nbins(self) -> int:
        return self.dataset_dims['num_bins']

    @property
    def include_bin_features(self) -> bool:
        return self.bin_states is not None

    @property
    def hpGrid(self):
        from blancops.ephemerides import ephemerides as _eph
        return _eph.HealpixGrid(nside=self.nside, is_azel=self.is_azel)

    # ------------------------------------------------------------------
    # Construction from TransitionDataset
    # ------------------------------------------------------------------

    @classmethod
    def from_transition_dataset(cls, dataset, split: str = 'val') -> 'DatasetCache':
        """Build from a ``TransitionDataset`` constructed on a single split's
        ``RawFeatureCache``. All transitions in the source dataset belong to
        that split.

        Args:
            dataset: The source TransitionDataset.
            split: Split name, 'val' or 'test'.

        Returns:
            The populated DatasetCache.
        """
        split_nights = dataset.night_split.nights_for(split)
        return cls(
            states=dataset.states,
            bin_states=dataset.bin_states,
            action_masks=dataset.action_masks,
            active_bin_mask=getattr(dataset, 'active_bin_mask', None),
            actions=dataset.actions,
            rewards=dataset.rewards,
            dones=dataset.dones,
            slew_distances=dataset.slew_distances,
            curr_compact_idxs=dataset.curr_compact_idxs,
            next_compact_idxs=dataset.next_compact_idxs,
            current_state_idxs=dataset.current_state_idxs,
            next_state_idxs=dataset.next_state_idxs,
            state_idxs=dataset.state_idxs,
            split_df=dataset._df,
            global_feature_names=dataset.global_feature_names,
            bin_feature_names=dataset.bin_feature_names,
            dataset_dims=dataset.dataset_dims,
            split_nights=list(split_nights),
            nside=dataset.hpGrid.nside,
            is_azel=dataset.hpGrid.is_azel,
            split=split,
        )

    # ------------------------------------------------------------------
    # Transition alignment
    # ------------------------------------------------------------------

    def to_transition_tensors(
        self,
        idxs: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Expand the compact per-state tensors into per-transition rows. Used only
        in run_explain.py for now.

        Args:
            idxs: transition indices to expand; None expands all. Subsetting
                here, before the gather, keeps peak memory proportional to the
                sample instead of the full transition set.

        Returns:
            global_obs     [n_transitions, n_global]
            bin_obs        [n_transitions, n_bins, n_bin_feats]
            expert_actions [n_transitions]
            valid_mask     [n_transitions, n_bins * n_filters] bool
        """
        curr = torch.as_tensor(self.curr_compact_idxs, dtype=torch.long)
        actions = self.actions.long()
        if idxs is not None:
            idxs = torch.as_tensor(idxs, dtype=torch.long)
            curr = curr[idxs]
            actions = actions[idxs]
        return (
            self.states[curr],               # [n_transitions, n_global]
            self.bin_states[curr] if self.bin_states is not None else None,           # [n_transitions, n_bins, n_bin_feats]
            actions,                         # [n_transitions]
            self.action_masks[curr].bool(),  # [n_transitions, n_actions]
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # torch.save pickles non-tensor fields (DataFrame, lists, dicts)
        torch.save(dataclasses.asdict(self), path)
        logger.info(f"DatasetCache ({self.split}) saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'DatasetCache':
        d = torch.load(path, weights_only=False)
        # Restore numpy arrays from any tensors that torch.save may have converted
        for key in ('curr_compact_idxs', 'next_compact_idxs',
                    'current_state_idxs', 'next_state_idxs', 'state_idxs'):
            if isinstance(d[key], torch.Tensor):
                d[key] = d[key].numpy()
        # Migrate caches written before the split rename
        if 'val_df' in d:
            d['split_df'] = d.pop('val_df')
        if 'val_nights' in d:
            d['split_nights'] = d.pop('val_nights')
        d.setdefault('split', 'val')
        return cls(**d)

    @classmethod
    def exists(cls, path: Path) -> bool:
        return Path(path).exists()


# Backwards-compatible alias for code and pickles predating the split rename
ValDatasetCache = DatasetCache
