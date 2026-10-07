import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset, RandomSampler

from blancops.configs.enums import RewardTerm
from blancops.configs.experiment_schema import ExperimentConfig, RLAlgConfig
from blancops.data.feature_cache import BinFeatureCache, FieldFeatureCache
from blancops.data.lookup_tables import LookupTables
from blancops.data.norm_stats import NormStats
from blancops.data.rewards import combine_rewards, normalize_rewards, reward_norm_stats
from blancops.ephemerides import ephemerides
from blancops.math import geometry, units

from blancops.configs.constants import _CYCLICAL_FEATURE_NAMES, ZENITH_BIN_NUM, ZENITH_FIELD_ID, ZENITH_FILTER

from blancops.data.features.normalizations import StateNormalizer, build_normalizer_kwargs, setup_feature_names
from blancops.data.splits import NightSplit, resolve_night_split
from blancops.survey.profiles import DES, SurveyProfile
from blancops.telescope.base import TelescopeProfile
from blancops.configs.enums import grid_is_azel, has_filter, is_field_level
from blancops.configs.experiment_schema import ActionConstraints
from blancops.data.features.field_features import FieldGrid, build_field_normalizer, expand_field_feature_names, field_norm_stats

logger = logging.getLogger(__name__)


# Chunk size for OOM-safe candidate feature gathering
_GATHER_CHUNK = 1024


def _gather_candidate_features(candidate_features, rows, cols):
    """Read selected rows and feature columns of a candidate-feature array in chunks.

    Args:
        candidate_features: (n_rows, n_candidates, n_all_feats) array or memmap.
        rows: Row indices to gather.
        cols: Feature-column indices to keep.

    Returns:
        (len(rows), n_candidates, len(cols)) float32 array.
    """
    rows = np.asarray(rows)
    n_candidates = candidate_features.shape[1]
    out = np.empty((len(rows), n_candidates, len(cols)), dtype=np.float32)
    for start in range(0, len(rows), _GATHER_CHUNK):
        chunk_rows = rows[start:start + _GATHER_CHUNK]
        out[start:start + len(chunk_rows)] = candidate_features[chunk_rows][:, :, cols]
    return out


def _overwrite_fwhm_with_causal(df, seeing_cfg, survey: SurveyProfile = DES):
    """Replace the raw measured 'fwhm' column with the causal prediction.

    Groups by night and applies compute_causal_fwhm so each row's fwhm is
    the strictly-past Seeing prediction at that row's own pointing. Matches
    the historic-eval rollout and live inference fwhm, removing train/serve
    skew. No-op if 'fwhm' is absent.
    """
    from blancops.data.features.glob_features import compute_causal_fwhm
    if 'fwhm' not in df.columns:
        return df
    out = np.empty(len(df), dtype=float)
    for _, idx in df.groupby('night', sort=False).groups.items():
        positions = df.index.get_indexer(idx)
        night_df = df.loc[idx]
        out[positions] = compute_causal_fwhm(night_df, seeing_cfg, survey=survey)
    df = df.copy()
    df['fwhm'] = out
    return df


def _collapse_cyclical_expansions(feature_names, cyclical_names):
    """Collapse ``<name>_cos`` / ``<name>_sin`` pairs back to ``<name>``.

    Idempotent on already-collapsed lists. # RH: probably a better way to manage features/mappings...
    """
    def _is_cyclical(name):
        return any(
            name == cyc or name.endswith(f"_{cyc}")
            for cyc in cyclical_names
        )

    result = []
    seen = set()
    for name in feature_names:
        base = name
        for suffix in ("_cos", "_sin"):
            if name.endswith(suffix):
                candidate = name[:-len(suffix)]
                if _is_cyclical(candidate):
                    base = candidate
                    break
        if base not in seen:
            result.append(base)
            seen.add(base)
    return result




# ---------------------------------------------------------------------------
# OfflineDataset — light DataLoader wrapper
# ---------------------------------------------------------------------------

class OfflineDataset:
    """Thin wrapper that creates train/val DataLoaders from a ``TransitionDataset``."""

    def __init__(
        self,
        dataset: "TransitionDataset",
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        generator = torch.Generator().manual_seed(seed)

        train_subset = Subset(dataset, dataset.train_transition_idxs.tolist())
        val_subset = Subset(dataset, dataset.val_transition_idxs.tolist())

        self.train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=RandomSampler(
                train_subset, replacement=True, num_samples=10 ** 10, generator=generator
            ),
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


# ---------------------------------------------------------------------------
# TransitionDatasetCache -- frozen, normalized TransitionDataset for one split
# ---------------------------------------------------------------------------

@dataclass
class TransitionDatasetCache:
    """Post-normalization tensors for one split's nights.

    Built after a training run fixes the night split and normalization stats.
    Saved as ``outdir/checkpoints/<split>_dataset_cache.pt`` (``torch.save``).

    Exposes the same attributes queried by the evaluator infrastructure
    (``DataContainer``, ``SingleStepEvaluator``) so it can be used as a
    drop-in replacement for ``TransitionDataset`` in those paths.
    """

    # Normalized state tensors (val states only)
    states: torch.Tensor
    candidate_states: Optional[torch.Tensor]  # None when no candidate features
    action_masks: torch.Tensor
    active_candidate_mask: Optional[torch.Tensor]  # None when no candidate features

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
    candidate_feature_names: List[str]
    dataset_dims: dict
    split_nights: List[str]
    nside: int
    is_azel: bool
    split: str = 'val'
    # Field centers (RA, Dec) for field_filter datasets, whose candidates are fields; None for bin datasets
    field_radec: Optional[np.ndarray] = None

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
    def _prenorm_candidate_states(self) -> Optional[torch.Tensor]:
        # In TransitionDataset the prenorm array is normalized in-place, so
        # _prenorm_candidate_states IS the normalized candidate_states after __init__.
        return self.candidate_states

    @property
    def n_candidates(self) -> int:
        return self.dataset_dims['num_candidates']

    @property
    def include_candidate_features(self) -> bool:
        return self.candidate_states is not None

    @property
    def candidate_grid(self):
        if self.field_radec is not None:
            return FieldGrid(self.field_radec[0], self.field_radec[1])
        return ephemerides.HealpixGrid(nside=self.nside, is_azel=self.is_azel)

    # ------------------------------------------------------------------
    # Construction from TransitionDataset
    # ------------------------------------------------------------------

    @classmethod
    def from_transition_dataset(cls, dataset, split: str = 'val') -> 'TransitionDatasetCache':
        """Build from a ``TransitionDataset`` constructed on a single split's
        feature cache. All transitions in the source dataset belong to
        that split.

        Args:
            dataset: The source TransitionDataset.
            split: Split name, 'val' or 'test'.

        Returns:
            The populated TransitionDatasetCache.
        """
        split_nights = dataset.night_split.nights_for(split)
        return cls(
            states=dataset.states,
            candidate_states=dataset.candidate_states,
            action_masks=dataset.action_masks,
            active_candidate_mask=getattr(dataset, 'active_candidate_mask', None),
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
            candidate_feature_names=dataset.candidate_feature_names,
            dataset_dims=dataset.dataset_dims,
            split_nights=list(split_nights),
            nside=dataset.candidate_grid.nside,
            is_azel=dataset.candidate_grid.is_azel,
            split=split,
            field_radec=(np.array([dataset.candidate_grid.lon, dataset.candidate_grid.lat])
                         if getattr(dataset, 'field_level', False) else None),
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
            candidate_obs  [n_transitions, n_candidates, n_candidate_feats]
            expert_actions [n_transitions]
            valid_mask     [n_transitions, n_candidates * n_filters] bool
        """
        curr = torch.as_tensor(self.curr_compact_idxs, dtype=torch.long)
        actions = self.actions.long()
        if idxs is not None:
            idxs = torch.as_tensor(idxs, dtype=torch.long)
            curr = curr[idxs]
            actions = actions[idxs]
        return (
            self.states[curr],               # [n_transitions, n_global]
            self.candidate_states[curr] if self.candidate_states is not None else None,  # [n_transitions, n_candidates, n_candidate_feats]
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
        logger.info(f"TransitionDatasetCache ({self.split}) saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'TransitionDatasetCache':
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
        # Migrate caches written before the bin -> candidate rename
        for old, new in (('bin_states', 'candidate_states'), ('active_bin_mask', 'active_candidate_mask'),
                         ('bin_feature_names', 'candidate_feature_names')):
            if old in d:
                d[new] = d.pop(old)
        for old, new in (('bin_state_dim', 'candidate_state_dim'), ('num_bins', 'num_candidates')):
            if old in d['dataset_dims']:
                d['dataset_dims'][new] = d['dataset_dims'].pop(old)
        if 'candidate_idx' not in d['split_df'].columns:
            d['split_df']['candidate_idx'] = d['split_df']['bin']
        return cls(**d)

    @classmethod
    def exists(cls, path: Path) -> bool:
        return Path(path).exists()



# ---------------------------------------------------------------------------
# TransitionDataset — all heavy logic
# ---------------------------------------------------------------------------

class TransitionDataset(torch.utils.data.Dataset):
    """Constructs and stores all RL transitions from a ``BinFeatureCache`` or ``FieldFeatureCache``.

    Pre-computing performs heavy data-processing so various models can be run on a one-time-compute
    dataset. Only normalization, reward/action/mask construction, and train/val/test splitting happen here.
    """

    def __init__(
        self,
        cache : BinFeatureCache | FieldFeatureCache,
        cfg : ExperimentConfig = None,
        lookups : LookupTables = None,
        norm_stats: NormStats | None = None,     # None: fit on this dataset's training transitions
        split_role : str | None = None,
        telescope: TelescopeProfile | None = None,
        survey: SurveyProfile = DES,
    ):
        self._given_norm_stats = norm_stats
        self._telescope = telescope or survey.telescope
        self._survey = survey
        survey.check_telescope(self._telescope)
        if lookups is not None:
            survey.check_lookups(lookups)
        norm_kwargs = build_normalizer_kwargs(cfg.data.norm, survey=survey)
        self._split_role = split_role
        self._setup_configuration(cfg, norm_kwargs)
        self.lookups = lookups
        if self.field_level:
            self.candidate_grid = FieldGrid(lookups.fields['ra'].to_numpy(), lookups.fields['dec'].to_numpy())
        else:
            self.candidate_grid = ephemerides.HealpixGrid(
                nside=cfg.data.nside,
                is_azel=grid_is_azel(cfg.data.action_space),
            )

        self._load_from_cache(cache)
        self._build_transitions(cfg.data.action_space)
        self._apply_min_teff()
        self._drop_interrupted()
        self._split_data(cfg)
        self._normalize_rewards()
        self._normalize_states(norm_kwargs)
        self._format_tensors_for_network(cfg.model.network)
        self._validate_dataset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_configuration(self, cfg, norm_kwargs):
        self._seeing_cfg = cfg.data.seeing
        self._min_teff = cfg.data.min_teff
        self._band_threshold = self._survey.acceptance_thresholds(cfg.data.acceptance)  # [n_filters] minimum accepted teff
        self._drop_interrupted_transitions = getattr(cfg.data, 'drop_interrupted', False)
        self.reward_cfg = cfg.model.reward if isinstance(cfg.model, RLAlgConfig) else None
        self._calculate_action_mask = cfg.model.algorithm != 'bc' # expensive and not needed for bc
        self.include_candidate_features = len(cfg.data.bin_features) > 0

        action_space = cfg.data.action_space
        self.num_filters = self._survey.num_filters if has_filter(action_space) else 1

        # num_actions resolved after n_candidates is known from cache
        self._action_space_str = action_space
        if action_space == 'filter':
            self.num_actions = self.num_filters
        else:
            self.num_actions = None  # filled in _load_from_cache

        # Collapse any post-expansion feature names (from resolved_config.yaml)
        base_global = _collapse_cyclical_expansions(
            list(cfg.data.global_features), _CYCLICAL_FEATURE_NAMES
        )
        base_bin = _collapse_cyclical_expansions(
            list(cfg.data.bin_features), _CYCLICAL_FEATURE_NAMES
        )
        self.base_global_feature_names = base_global
        self.base_bin_feature_names = base_bin
        self.global_feature_names, self.candidate_feature_names = setup_feature_names(
            base_global,
            base_bin,
            norm_kwargs['cyclical_feature_names'],
            norm_kwargs['do_cyclical_norm'],
            do_filt=has_filter(action_space),
            survey=self._survey,
        )
        self.do_local_mean_z_score = any('rel_' in name for name in self.candidate_feature_names)

        # field_filter
        self.field_level = is_field_level(action_space)
        if self.field_level:
            self.candidate_feature_names = expand_field_feature_names(list(cfg.data.field_features), survey=self._survey)
            self.include_candidate_features = True
            self.do_local_mean_z_score = False
            constraints = ActionConstraints()
            self._airmass_limit = min(constraints.airmass_limit, constraints.airmass_failsafe)

    # ------------------------------------------------------------------
    # Load from cache
    # ------------------------------------------------------------------

    def _load_from_cache(self, cache):
        missing_global = set(self.global_feature_names) - set(cache.global_feature_names)
        assert not missing_global, (
            f"Global features missing from cache: {missing_global}. "
            "Re-run precompute-features."
        )
        if self.field_level:
            if cache.field_features is None:
                raise FileNotFoundError("field_filter needs field features in the cache; run "
                                        "`precompute-train-features --field_features`.")
            missing_field = set(self.candidate_feature_names) - set(cache.field_feature_names)
            assert not missing_field, f"Field features missing from cache: {missing_field}."
        elif self.include_candidate_features:
            missing_bin = set(self.candidate_feature_names) - set(cache.bin_feature_names)
            assert not missing_bin, (
                f"Bin features missing from cache: {missing_bin}. "
                "Re-run precompute-features."
            )

        self._df = cache.global_df
        self._interruptions = getattr(cache, 'interruptions', None)

        if 'fwhm' in self.global_feature_names:
            self._df = _overwrite_fwhm_with_causal(self._df, self._seeing_cfg, self._survey)

        if self.field_level:
            field_indices = [cache.field_feature_names.index(f) for f in self.candidate_feature_names]
            self._prenorm_candidate_states = _gather_candidate_features(
                cache.field_features, cache.state_idxs, field_indices
            )
            self._field_cache = cache
            self._df = self._df.copy()
            for name, values in cache.field_tiling.items():
                if name in self._df.columns:
                    self._df[name] = values
            # Field candidates are indexed by field id (zenith rows keep the zenith sentinel).
            fids = self._df['field_id'].to_numpy()
            self._df['candidate_idx'] = np.where(fids == ZENITH_FIELD_ID, ZENITH_BIN_NUM, fids).astype(np.int64)
        else:
            self._df['candidate_idx'] = self._df['bin']
            if self.include_candidate_features:
                bin_indices = [cache.bin_feature_names.index(f) for f in self.candidate_feature_names]
                self._prenorm_candidate_states = _gather_candidate_features(
                    cache.bin_features, cache.state_idxs, bin_indices
                )
            else:
                self._prenorm_candidate_states = None

        self.state_idxs = cache.state_idxs
        self.current_state_idxs = cache.current_state_idxs
        self.next_state_idxs = cache.next_state_idxs
        self.df_idx_to_compact = {int(idx): i for i, idx in enumerate(cache.state_idxs)}
        self.curr_compact_idxs = np.array(
            [self.df_idx_to_compact[i] for i in cache.current_state_idxs]
        )
        self.next_compact_idxs = np.array(
            [self.df_idx_to_compact[i] for i in cache.next_state_idxs]
        )
        self.slew_distances = torch.as_tensor(cache.slew_distances, dtype=torch.float32)

        self.n_candidates = len(self.candidate_grid.lon)
        self.unique_nights = self._df['night'].unique()
        self.n_nights = self._df.groupby('night').ngroups

        if self.num_actions is None:
            action_space = self._action_space_str
            if action_space in ['radec', 'azel']:
                self.num_actions = self.n_candidates
            else:
                self.num_actions = self.n_candidates * self.num_filters

    # ------------------------------------------------------------------
    # Transition construction
    # ------------------------------------------------------------------

    def _build_transitions(self, action_space):
        states, candidate_states = self._construct_states(
            df=self._df,
            candidate_states=self._prenorm_candidate_states,
            include_candidate_features=self.include_candidate_features,
            state_idxs=self.state_idxs,
        )
        num_transitions = len(self.next_state_idxs)

        actions = self._construct_actions(self._df, action_space=action_space,
                                          next_state_idxs=self.next_state_idxs)
        rewards = self._construct_rewards(self._df, next_state_idxs=self.next_state_idxs)
        dones = self._construct_dones(num_transitions=num_transitions,
                                      next_state_idxs=self.next_state_idxs,
                                      current_state_idxs=self.current_state_idxs)
        action_masks = self._construct_action_masks(
            state_df=self._df, action_space=action_space,
            num_states=len(self.state_idxs), state_idxs=self.state_idxs,
        )

        if self.field_level and self._calculate_action_mask:
            # Let each state allow the chosen expert action.
            action_masks[self.curr_compact_idxs, actions] = True

        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.int32)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)
        self.dones = torch.as_tensor(dones, dtype=torch.bool)
        self.action_masks = torch.as_tensor(action_masks, dtype=torch.bool)
        self.num_transitions = num_transitions

        if self.include_candidate_features:
            self._prenorm_candidate_states = torch.as_tensor(candidate_states, dtype=torch.float32)
        else:
            self._prenorm_candidate_states = None

    def _apply_min_teff(self) -> None:
        """Drop transitions whose exposure has teff <= min_teff; no-op when min_teff is None.

        Runs after _build_transitions, so dones (night ends) are computed on the full sequence and a
        dropped mid-night exposure does not end an episode early.
        """
        if self._min_teff is None:
            return
        keep = self._df['teff'].to_numpy()[self.next_state_idxs] > self._min_teff
        logger.info(f"min_teff={self._min_teff}: keeping {int(keep.sum())} of {len(keep)} transitions")
        self._keep_transitions(keep)

    def _drop_interrupted(self) -> None:
        """Drop transitions whose exposure followed another archived exposure since the previous survey exposure.

        The DES wide-field survey had program interruptions. When calculating transition quantities, need to drop these
        observations. No-op when disabled or when the cache has no interruptions file;
        dones are computed before dropping, as for min_teff.
        """
        if not self._drop_interrupted_transitions or self._interruptions is None:
            return
        keep = ~np.isin(self._df['expnum'].to_numpy()[self.next_state_idxs], self._interruptions['expnum'].to_numpy())
        logger.info(f"drop_interrupted: keeping {int(keep.sum())} of {len(keep)} transitions")
        self._keep_transitions(keep)

    def _keep_transitions(self, keep: np.ndarray) -> None:
        """Keep only the transitions where ``keep`` is True, across every per-transition array.

        Parameters
        ----------
        keep : np.ndarray
            Boolean mask over transitions.
        """
        self.current_state_idxs = self.current_state_idxs[keep]
        self.next_state_idxs = self.next_state_idxs[keep]
        self.curr_compact_idxs = self.curr_compact_idxs[keep]
        self.next_compact_idxs = self.next_compact_idxs[keep]
        keep_t = torch.as_tensor(keep)
        self.slew_distances = self.slew_distances[keep_t]
        self.actions = self.actions[keep_t]
        self.rewards = self.rewards[keep_t]
        self.dones = self.dones[keep_t]
        self.num_transitions = int(keep.sum())

    def _construct_dones(self, num_transitions, next_state_idxs, current_state_idxs):
        dones = ~np.isin(next_state_idxs, current_state_idxs)
        dones[-1] = True
        return dones

    def _construct_states(self, df, candidate_states, include_candidate_features, state_idxs):
        global_states = self._construct_global_features(df=df, state_idxs=state_idxs)
        if not include_candidate_features:
            candidate_states = None
        return global_states, candidate_states

    def _construct_global_features(self, df, state_idxs):
        missing_cols = set(self.global_feature_names) - set(df.columns)
        assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe.'
        return df.iloc[state_idxs][self.global_feature_names].to_numpy()

    def _construct_actions(self, df, action_space, next_state_idxs):
        assert action_space in ['radec', 'azel', 'radec_filter', 'azel_filter', 'field_filter', 'filter'] # XXX check if in ActionSpace Enum, not hard-coded
        next_state_df = df.iloc[next_state_idxs]

        if is_field_level(action_space):
            assert ZENITH_FILTER not in next_state_df['filter'].values, \
                f"Invalid data: Found '{ZENITH_FILTER}' in next_state_df."
            field_ids = next_state_df['field_id'].to_numpy().astype(np.int64)
            filter_indices = next_state_df['filter'].map(self._survey.filter2idx).values.astype(np.int64)
            return field_ids * self._survey.num_filters + filter_indices

        if self.candidate_grid.is_azel:
            lonlat = next_state_df[['az', 'el']].values
        else:
            lonlat = next_state_df[['ra', 'dec']].values

        bin_indices = self.candidate_grid.ang2idx(lon=lonlat[:, 0], lat=lonlat[:, 1])

        if 'filter' not in action_space:
            return bin_indices
        elif ('radec' not in action_space) and ('azel' not in action_space):
            return df.iloc[next_state_idxs]['filter'].map(self._survey.filter2idx).values.astype(np.int32)
        else:
            assert ZENITH_FILTER not in next_state_df['filter'].values, \
                f"Invalid data: Found '{ZENITH_FILTER}' in next_state_df."
            filter_indices = next_state_df['filter'].map(self._survey.filter2idx).values.astype(np.int32)
            return (bin_indices * self._survey.num_filters) + filter_indices

    def _construct_rewards(self, df, next_state_idxs) -> np.ndarray:
        """Unnormalized rewards; scaled later by _normalize_rewards."""
        if self.reward_cfg is None:
            return np.zeros(len(next_state_idxs), dtype=np.float32)
        return combine_rewards(self.reward_cfg, self._reward_term_inputs(df, next_state_idxs))

    def _normalize_rewards(self) -> None:
        """Fit reward stats on the training transitions (or take the given ones) and scale all rewards."""
        self._reward_stats = None
        if self.reward_cfg is None:
            return
        R_tot = self.rewards.numpy()
        if self._given_norm_stats is None:
            self._reward_stats = reward_norm_stats(self.reward_cfg, R_tot[self.train_transition_idxs])
        else:
            self._reward_stats = self._given_norm_stats.reward
        self.rewards = torch.as_tensor(
            normalize_rewards(self.reward_cfg, R_tot, self._reward_stats), dtype=torch.float32
        )

    def _reward_term_inputs(self, df, next_state_idxs):
        next_df = df.iloc[next_state_idxs]
        # groups = df.groupby(['field_id', 'filter'])

        return {
            RewardTerm.EXPERT: lambda: dict(n_transitions=len(next_state_idxs)),
            RewardTerm.TEFF: lambda: dict(teff=next_df['teff'].values),
            RewardTerm.TEFF_ACCEPTED: lambda: self._teff_accepted_inputs(next_df),
            RewardTerm.SLEW: lambda: dict(excess_times=self._excess_dead_times(df, next_state_idxs)),
            RewardTerm.UNIFORMITY: lambda: self._uniformity_inputs(df, next_state_idxs),
        }

    def _exposure_field_filter_idxs(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exposure mask and integer field and filter indices per row; non-exposures get index 0.

        Parameters
        ----------
        df : pd.DataFrame
            Rows with field_id and filter.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            is_exposure (bool), field_ids (int), filter_idxs (int), each shape (n_rows,).
        """
        is_exposure = (df['field_id'] != ZENITH_FIELD_ID).to_numpy()
        field_ids = np.where(is_exposure, df['field_id'].to_numpy(), 0).astype(int)
        filter_idxs = df['filter'].map(self._survey.filter2idx).fillna(0).to_numpy().astype(int)
        return is_exposure, field_ids, filter_idxs

    def _teff_accepted_inputs(self, next_df: pd.DataFrame) -> dict:
        """Inputs of the accepted effective-seconds reward for each transition's exposure.

        Exposure time comes from the lookups (most common per field-filter), as in the environment,
        so offline and rollout rewards agree. A transition that is not an exposure earns 0.

        Parameters
        ----------
        next_df : pd.DataFrame
            Row of each transition's exposure, with field_id, filter, teff.

        Returns
        -------
        dict
            teff, min_teff, exptime, each shape (n_transitions,).
        """
        is_exposure, field_ids, filter_idxs = self._exposure_field_filter_idxs(next_df)
        exptime = np.where(is_exposure, self.lookups.fidfilt_exptime[field_ids, filter_idxs], 0.0)
        min_teff = np.where(is_exposure, self._band_threshold[filter_idxs], 0.0)
        return dict(teff=next_df['teff'].to_numpy(dtype=float), min_teff=min_teff, exptime=exptime)

    def _uniformity_inputs(self, df, next_state_idxs) -> dict: # XXX make independent of form of uniformity metric
        """Inputs of the uniformity reward for each transition, from survey counts before its exposure.

        count_before = count at night start (lookups.night2fidfilt_visit_hist) + earlier exposures of the
        field-filter that night; filter_mean m_b = (night-start sum of counts over in-plan fields + earlier
        in-plan exposures in the filter that night) / W_b, with W_b the filter's total target. Every exposure
        of the night earns its pass size and advances the counts whatever its teff; night-start counts hold
        accepted exposures only. All rows of the night are scanned, including rows whose transitions were
        dropped.

        Parameters
        ----------
        df : pd.DataFrame
            Global DataFrame with night, field_id, filter columns, in time order within each night.
        next_state_idxs : np.ndarray
            Row index of each transition's exposure.

        Returns
        -------
        dict
            completion, filter_mean, total_target, pass_size, each shape (n_transitions,).
        """
        visit_hist = getattr(self.lookups, 'night2fidfilt_visit_hist', None)
        if visit_hist is None:
            raise ValueError("The uniformity reward needs lookups.night2fidfilt_visit_hist; rebuild lookups.")
        if 'filter' not in self._action_space_str:
            raise ValueError("The uniformity reward needs a filter action space (azel_filter or radec_filter).")

        targets = self.lookups.target_fidfilt_counts.astype(float)                      # [n_fields, n_filters]
        in_plan = targets > 0
        total_target = targets.sum(axis=0)                                              # [n_filters]
        inv_target = np.divide(1.0, targets, out=np.zeros_like(targets), where=in_plan)

        is_exposure, field_ids, filter_idxs = self._exposure_field_filter_idxs(df)
        pass_size = np.where(is_exposure, inv_target[field_ids, filter_idxs], 0.0)

        count_start = np.zeros(len(df))
        sum_n_start = np.zeros(len(df))
        for night, rows in df.groupby('night').indices.items():
            night_counts = visit_hist[night].astype(float)
            count_start[rows] = night_counts[field_ids[rows], filter_idxs[rows]]
            sum_n_start[rows] = (night_counts * in_plan).sum(axis=0)[filter_idxs[rows]]

        exposures = pd.DataFrame({
            'night': df['night'].to_numpy(), 'field_id': field_ids,
            'filter_idx': filter_idxs, 'in_plan': (pass_size > 0).astype(float),
        })[is_exposure]
        earlier_visits = np.zeros(len(df))
        earlier_in_plan = np.zeros(len(df))
        earlier_visits[is_exposure] = exposures.groupby(['night', 'field_id', 'filter_idx']).cumcount().to_numpy()
        earlier_in_plan[is_exposure] = (
            exposures.groupby(['night', 'filter_idx'])['in_plan'].cumsum() - exposures['in_plan']
        ).to_numpy()

        idx = next_state_idxs
        return dict(
            completion=(count_start[idx] + earlier_visits[idx]) * pass_size[idx],
            filter_mean=(sum_n_start[idx] + earlier_in_plan[idx]) / total_target[filter_idxs[idx]],
            total_target=total_target[filter_idxs[idx]],
            pass_size=pass_size[idx],
        )

    def _excess_dead_times(self, df, next_state_idxs) -> np.ndarray:
        """Excess dead time beyond the per-visit overhead.
        """
        curr_df = df.iloc[self.current_state_idxs]
        next_df = df.iloc[next_state_idxs]
        distances = geometry.angular_separation(
            self._field_center_radec(curr_df), self._field_center_radec(next_df)
        )
        curr_filters = curr_df['filter'].values
        filter_change = (curr_filters != next_df['filter'].values) & (curr_filters != ZENITH_FILTER)

        params = self._telescope.parameters
        return params.dead_time(distances / units.deg, filter_change) - params.visit_overhead(filter_change)

    def _field_center_radec(self, rows) -> tuple[np.ndarray, np.ndarray]:
        """Lookup field-center (and zenith) RA/Dec per row.

        Parameters
        ----------
        rows : pd.DataFrame
            Rows with field_id, ra, dec (radians).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            RA and Dec in radians, each shape (n_rows,).
        """
        field_ids = rows['field_id'].to_numpy()
        is_zenith = field_ids == ZENITH_FIELD_ID
        safe_ids = np.where(is_zenith, 0, field_ids).astype(int)
        zen_ra, zen_dec = rows['ra'].to_numpy(dtype=float), rows['dec'].to_numpy(dtype=float)
        if self.field_level and is_zenith.any():
            # Field runs slew from the zenith computed as the environment does (float64 at the row's time).
            ts = rows['timestamp'].to_numpy()
            for i in np.where(is_zenith)[0]:
                zen_ra[i], zen_dec[i] = ephemerides.blanco_observer(time=float(ts[i])).radec_of('0', '90')
        ra = np.where(is_zenith, zen_ra if self.field_level else rows['ra'].to_numpy(),
                      self.lookups.fields['ra'].to_numpy()[safe_ids])
        dec = np.where(is_zenith, zen_dec if self.field_level else rows['dec'].to_numpy(),
                       self.lookups.fields['dec'].to_numpy()[safe_ids])
        return ra, dec



    # def _construct_rewards(self, df, next_state_idxs, reward):
    #     if reward == RewardStructure.TEFF:
    #         R_tot = df.iloc[next_state_idxs]['teff'].fillna(0).values
    #     elif reward == RewardStructure.EXPERT_ACTION:
    #         R_tot = np.ones(len(next_state_idxs), dtype=np.float32)
    #     elif reward == RewardStructure.COMPOSITE:
    #         rw = self.reward_weights
    #         R_slew = self._construct_slew_reward()
    #         # R_airmass = self._construct_airmass_reward(df, next_state_idxs, rw)
    #         R_tsince = self._construct_t_since_reward(df, next_state_idxs, rw)
    #         R_tiling = self._construct_min_tiling_reward(df, next_state_idxs, rw)
    #         R_tot = (rw.w_slew * R_slew
    #                 #  + rw.w_airmass * R_airmass
    #                 #  + rw.w_t_last_visit * R_tsince
    #                  + rw.w_min_tiling * R_tiling).astype(np.float32)
    #     elif reward is None:
    #         return np.zeros(len(next_state_idxs), dtype=np.float32)
    #     else:
    #         raise NotImplementedError

    #     if self.reward_norm == 'minmax':
    #         R_tot = (R_tot - R_tot.min()) / (R_tot.max() - R_tot.min())
    #     elif self.reward_norm is not None:
    #         logger.warning(f"Unknown reward norm: {self.reward_norm}")
    #     return R_tot

    # def _construct_airmass_reward(self, df, next_state_idxs, rw):
    #     airmass = df.iloc[next_state_idxs]['airmass'].values
    #     return np.clip(
    #         (rw.airmass_limit - airmass) / (rw.airmass_limit - 1.0), 0.0, 1.0
    #     )

    # def _construct_slew_reward(self):
    #     return 1.0 - self.slew_distances.numpy() / np.pi

    # def _construct_t_since_reward(self, df, next_state_idxs, rw):
    #     t_diff = df.groupby(['field_id', 'filter'])['timestamp'].diff()
    #     t_since = t_diff.iloc[next_state_idxs].fillna(rw.t_ref_seconds).values
    #     t_min, t_max = t_since.min(), t_since.max()
    #     return (t_since - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(t_since)

    # def _construct_min_tiling_reward(self, df, next_state_idxs, rw):
    #     field_ids = df.iloc[next_state_idxs]['field_id'].values.astype(int)
    #     filter_idxs = df.iloc[next_state_idxs]['filter'].map(self._survey.filter2idx).values.astype(int)
    #     visits_before = df.groupby(['field_id', 'filter']).cumcount().iloc[next_state_idxs].values
    #     target_visits = self.lookups.target_fidfilt_counts[field_ids, filter_idxs]
    #     safe_target = np.where(target_visits > 0, target_visits, 1)
    #     assert ZENITH_FILTER not in df.iloc[next_state_idxs]['filter'].values
    #     return np.where(
    #         target_visits > 0,
    #         np.clip(1.0 - visits_before / safe_target, 0.0, 1.0),
    #         0.0,
    #     )

    def _field_action_masks(self, state_idxs) -> np.ndarray:
        """Field-level masks per state: visible (airmass, telescope mount envelope) and incomplete fields in plan.

        Parameters
        ----------
        state_idxs : np.ndarray
            Row indices of the states.

        Returns
        -------
        np.ndarray
            Boolean (n_states, n_fields * n_filters).
        """
        cache = self._field_cache
        names = cache.field_feature_names
        el_col, ha_col = names.index('el'), names.index('ha')
        comp_cols = [names.index(f"completion_{f}") for f in self._survey.filters]
        dec = self.lookups.fields['dec'].to_numpy()
        rows = np.asarray(state_idxs)
        masks = np.empty((len(rows), self.n_candidates * self.num_filters), dtype=np.bool_)
        for start in range(0, len(rows), _GATHER_CHUNK):
            chunk = cache.field_features[rows[start:start + _GATHER_CHUNK]]
            visible = self._telescope.visible(chunk[..., el_col], chunk[..., ha_col], dec, self._airmass_limit)
            completion = chunk[..., comp_cols]
            incomplete = completion < 1.0          # NaN (out of plan) compares False
            masks[start:start + len(chunk)] = (incomplete & visible[..., None]).reshape(len(chunk), -1)
        return masks

    def _construct_action_masks(self, state_df, action_space, num_states, state_idxs):
        if self.field_level:
            if not self._calculate_action_mask:
                return np.ones((num_states, self.num_actions), dtype=np.bool_)
            return self._field_action_masks(state_idxs)
        state_df = state_df.iloc[state_idxs]
        els = np.empty((num_states, self.n_candidates), dtype=np.float32)

        if action_space == 'filter':
            return np.ones((num_states, self.num_filters), dtype=np.bool_)

        if self._calculate_action_mask:
            logger.info("Calculating action masks based on horizon…")
            if not self.candidate_grid.is_azel:
                lon, lat = self.candidate_grid.lon, self.candidate_grid.lat
                for i, time in tqdm(
                    enumerate(state_df['timestamp'].values),
                    total=len(state_df['timestamp'].values),
                    desc="Calculating action mask",
                ):
                    _, els[i] = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)
                self._els = els
                action_mask = els > 0
            else:
                els = np.tile(
                    self.candidate_grid.lat[:, np.newaxis],
                    reps=len(state_df['timestamp'].values),
                ).T
                action_mask = els > 0
            if 'filter' in action_space:
                action_mask = np.repeat(action_mask, self.num_filters, axis=1)
        else:
            action_mask = np.ones((num_states, self.num_actions), dtype=np.bool_)
        return action_mask

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------

    def _split_data(self, cfg):
        """Assign transitions to train, val, and test.

        When ``split_role`` was passed to the constructor the cache already
        holds a single split's nights, so resolution is skipped and every
        transition is assigned to that split.

        Args:
            cfg: The experiment config.
        """
        n_transitions = len(self.next_state_idxs)
        all_idxs = np.arange(n_transitions)
        empty = np.array([], dtype=int)
        nights = [str(n) for n in self.unique_nights]

        if self._split_role is not None:
            if self._split_role not in ('train', 'val', 'test'):
                raise ValueError(f"Unknown split_role '{self._split_role}'.")
            self.night_split = NightSplit(
                train=sorted(nights) if self._split_role == 'train' else [],
                val=sorted(nights) if self._split_role == 'val' else [],
                test=sorted(nights) if self._split_role == 'test' else [],
                seed=int(cfg.train.seed),
            )
            self.train_transition_idxs = all_idxs if self._split_role == 'train' else empty
            self.val_transition_idxs = all_idxs if self._split_role == 'val' else empty
            self.test_transition_idxs = all_idxs if self._split_role == 'test' else empty
            train_c = self.curr_compact_idxs
            train_n = self.next_compact_idxs
        else:
            self.night_split = resolve_night_split(
                unique_nights=nights,
                seed=int(cfg.train.seed),
                val_nights=cfg.data.val_nights,
                val_frac=cfg.data.effective_val_frac,
                test_nights=cfg.data.test_nights,
                test_frac=cfg.data.test_frac,
            )
            transition_nights = self._df.iloc[self.next_state_idxs - 1]['night'].astype(str).to_numpy()
            self.train_transition_idxs = np.where(np.isin(transition_nights, self.night_split.train))[0]
            self.val_transition_idxs = np.where(np.isin(transition_nights, self.night_split.val))[0]
            self.test_transition_idxs = np.where(np.isin(transition_nights, self.night_split.test))[0]
            train_c = self.curr_compact_idxs[self.train_transition_idxs]
            train_n = self.next_compact_idxs[self.train_transition_idxs]

        self.train_nights = self.night_split.train
        self.val_nights = self.night_split.val
        self.test_nights = self.night_split.test
        self.train_state_idxs = np.unique(np.concatenate([train_c, train_n]))

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_states(self, norm_kwargs):
        """Fit feature stats on the training states (or apply the given ones) and set self.norm_stats."""
        fit = self._given_norm_stats is None
        global_normalizer = StateNormalizer(
            state_feature_names=self.global_feature_names, survey=self._survey, **norm_kwargs
        )
        candidate_normalizer = StateNormalizer(
            state_feature_names=self.candidate_feature_names, survey=self._survey, **norm_kwargs
        )
        if self.field_level:
            candidate_normalizer = build_field_normalizer(self.candidate_feature_names)
            field_stats = field_norm_stats(self.candidate_feature_names)

        if fit:
            self.states, glob_z, glob_rel, self.global_sentinel_mask = \
                global_normalizer.fit_transform(
                    state=self.states, train_state_idxs=self.train_state_idxs
                )
        else:
            self.states, self.global_sentinel_mask = global_normalizer.transform(
                state=self.states, **self._given_norm_stats.normalizer_kwargs('global_features')
            )

        bin_z, bin_rel = None, None
        if self.field_level:
            self._prenorm_candidate_states, self.candidate_sentinel_mask = candidate_normalizer.transform(
                state=torch.as_tensor(self._prenorm_candidate_states), z_stats_dict=field_stats, rel_stats_dict={}
            )
        elif self.include_candidate_features and self._prenorm_candidate_states is not None:
            bin_tensor = torch.as_tensor(self._prenorm_candidate_states)
            if fit:
                self._prenorm_candidate_states, bin_z, bin_rel, self.candidate_sentinel_mask = \
                    candidate_normalizer.fit_transform(
                        state=bin_tensor, train_state_idxs=self.train_state_idxs
                    )
            else:
                self._prenorm_candidate_states, self.candidate_sentinel_mask = candidate_normalizer.transform(
                    state=bin_tensor, **self._given_norm_stats.normalizer_kwargs('bin_features')
                )
        else:
            self.candidate_sentinel_mask = None

        # (n_states, n_candidates): True where the candidate has no sentinel values at this timestep
        if self.candidate_sentinel_mask is not None:
            self.active_candidate_mask = ~self.candidate_sentinel_mask.any(dim=-1)
        else:
            self.active_candidate_mask = None

        self.norm_stats = self._given_norm_stats if not fit else NormStats(
            z_score={'global_features': glob_z, 'bin_features': bin_z},
            rel_norm={'global_features': glob_rel, 'bin_features': bin_rel},
            reward=self._reward_stats,
        )
        if self.field_level and fit:
            self.norm_stats.z_score['field_features'] = field_stats

    # ------------------------------------------------------------------
    # Tensor formatting & validation
    # ------------------------------------------------------------------

    def _format_tensors_for_network(self, network_type):
        if network_type == 'mlp':
            if self.include_candidate_features and self._prenorm_candidate_states is not None:
                if not isinstance(self.states, torch.Tensor):
                    self.states = torch.as_tensor(self.states, dtype=torch.float32)
                if not isinstance(self._prenorm_candidate_states, torch.Tensor):
                    self._prenorm_candidate_states = torch.as_tensor(
                        self._prenorm_candidate_states, dtype=torch.float32
                    )
                cs_flat = self._prenorm_candidate_states.reshape(
                    self._prenorm_candidate_states.shape[0], -1
                )
                self.states = torch.cat([self.states, cs_flat], dim=1)
                self._prenorm_candidate_states = None
                self.candidate_states = None
            self.candidate_state_dim = 0
            self.state_dim = self.states.shape[-1]
        else:
            self.state_dim = self.states.shape[-1]
            self.candidate_states = self._prenorm_candidate_states
            self.candidate_state_dim = (
                self.candidate_states.shape[-1]
                if self.include_candidate_features and self.candidate_states is not None else 0
            )

        self.dataset_dims = {
            'state_dim': self.state_dim,
            'candidate_state_dim': self.candidate_state_dim,
            'num_candidates': self.n_candidates,
            'num_filters': self.num_filters,
            'num_actions': self.num_actions,
        }
        self.dataset_feature_names = {
            'global_features': self.global_feature_names,
            'bin_features': [] if self.field_level else self.candidate_feature_names,
            'field_features': self.candidate_feature_names if self.field_level else [],
        }

    def _validate_dataset(self):
        assert self.states.shape[0] == self.action_masks.shape[0], \
            "States and masks must be 1:1"
        assert (self.actions.shape[0] == self.rewards.shape[0]
                == self.dones.shape[0] == self.num_transitions), \
            (f"Transition mismatch: actions {self.actions.shape[0]}, "
             f"rewards {self.rewards.shape[0]}, dones {self.dones.shape[0]}")
        if self.include_candidate_features and self.candidate_states is not None:
            assert self.states.shape[0] == self.candidate_states.shape[0], \
                f"State mismatch: global {self.states.shape[0]}, candidate {self.candidate_states.shape[0]}"

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        c_idx = self.curr_compact_idxs[idx]
        n_idx = self.next_compact_idxs[idx]
        is_done = self.dones[idx].item()

        has_candidates = self.include_candidate_features and self.candidate_states is not None
        _zero_cand = torch.zeros_like(self.candidate_states[0]) if has_candidates else torch.as_tensor(0)
        cand_c = self.candidate_states[c_idx] if has_candidates else torch.as_tensor(0)
        cand_n = (self.candidate_states[n_idx] if not is_done else _zero_cand) if has_candidates else torch.as_tensor(0)

        return (
            self.states[c_idx],
            self.actions[idx],
            self.rewards[idx],
            self.states[n_idx] if not is_done else torch.zeros_like(self.states[0]),
            self.dones[idx],
            self.action_masks[c_idx],
            self.action_masks[n_idx] if not is_done else torch.zeros_like(self.action_masks[0]),
            cand_c,
            cand_n,
            self.slew_distances[idx],
        )
