import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset, RandomSampler

from blancops.configs.enums import RewardStructure
from blancops.configs.experiment_schema import RewardWeights
from blancops.ephemerides import ephemerides
from blancops.math import geometry

from blancops.configs.constants import _CYCLICAL_FEATURE_NAMES, _NUM_FILTERS, FILTER2IDX, ZENITH_FILTER

from blancops.data.features.normalizations import StateNormalizer, build_normalizer_kwargs, setup_feature_names
from blancops.data.splits import NightSplit, resolve_night_split

logger = logging.getLogger(__name__)


# Chunk size for OOM-safe bin feature gathering
_BIN_GATHER_CHUNK = 1024


def _gather_bin_features(bin_features, rows, cols):
    """Read selected rows and feature columns of a bin-feature array in chunks.

    Args:
        bin_features: (n_rows, n_bins, n_all_feats) array or memmap.
        rows: Row indices to gather.
        cols: Feature-column indices to keep.

    Returns:
        (len(rows), n_bins, len(cols)) float32 array.
    """
    rows = np.asarray(rows)
    n_bins = bin_features.shape[1]
    out = np.empty((len(rows), n_bins, len(cols)), dtype=np.float32)
    for start in range(0, len(rows), _BIN_GATHER_CHUNK):
        chunk_rows = rows[start:start + _BIN_GATHER_CHUNK]
        out[start:start + len(chunk_rows)] = bin_features[chunk_rows][:, :, cols]
    return out


def _overwrite_fwhm_with_causal(df, seeing_cfg):
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
        out[positions] = compute_causal_fwhm(night_df, seeing_cfg)
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
# TransitionDataset — all heavy logic
# ---------------------------------------------------------------------------

class TransitionDataset(torch.utils.data.Dataset):
    """Constructs and stores all RL transitions from a ``RawFeatureCache``.

    Accepts a pre-computed ``RawFeatureCache`` instead of a raw DataFrame so
    feature engineering (i.e., heavy computation) is skipped.
    Only normalization, reward/action/mask construction, and train/val/test splitting happen here.
    """

    def __init__(
        self,
        mode: str,
        cache,                  # RawFeatureCache; XXX why no type? circular import or forgot?
        cfg=None,
        lookups=None,
        z_score_stats=None,
        rel_norm_stats=None,
        split_role=None,
    ):
        norm_kwargs = build_normalizer_kwargs(cfg.data.norm)
        self._split_role = split_role
        self._setup_configuration(cfg, norm_kwargs)
        self.lookups = lookups
        self.hpGrid = ephemerides.HealpixGrid(
            nside=cfg.data.nside,
            is_azel=('azel' in cfg.data.action_space),
        )

        self._load_from_cache(cache)
        self._build_transitions(cfg.data.action_space)
        self._split_data(cfg)
        self._normalize_states(mode, cfg, norm_kwargs, z_score_stats, rel_norm_stats)
        self._format_tensors_for_network(cfg.model.network)
        self._validate_dataset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_configuration(self, cfg, norm_kwargs):
        self._seeing_cfg = cfg.data.seeing
        self.reward = cfg.model.reward
        self.reward_weights = getattr(cfg.model, 'reward_weights', None) or RewardWeights()
        self.reward_norm = getattr(cfg.model, 'reward_norm', 'minmax')
        self._calculate_action_mask = cfg.model.algorithm != 'bc'
        self.include_bin_features = len(cfg.data.bin_features) > 0

        action_space = cfg.data.action_space
        self.num_filters = _NUM_FILTERS if 'filter' in action_space else 1

        # num_actions resolved after nbins is known from cache
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
        self.global_feature_names, self.bin_feature_names = setup_feature_names(
            base_global,
            base_bin,
            norm_kwargs['cyclical_feature_names'],
            norm_kwargs['do_cyclical_norm'],
            do_filt='filter' in action_space,
        )
        self.do_local_mean_z_score = any('rel_' in name for name in self.bin_feature_names)

    # ------------------------------------------------------------------
    # Load from cache
    # ------------------------------------------------------------------

    def _load_from_cache(self, cache):
        missing_global = set(self.global_feature_names) - set(cache.global_feature_names)
        assert not missing_global, (
            f"Global features missing from cache: {missing_global}. "
            "Re-run precompute-features."
        )
        if self.include_bin_features:
            missing_bin = set(self.bin_feature_names) - set(cache.bin_feature_names)
            assert not missing_bin, (
                f"Bin features missing from cache: {missing_bin}. "
                "Re-run precompute-features."
            )

        self._df = cache.global_df

        if 'fwhm' in self.global_feature_names:
            self._df = _overwrite_fwhm_with_causal(self._df, self._seeing_cfg)

        if self.include_bin_features:
            bin_indices = [cache.bin_feature_names.index(f) for f in self.bin_feature_names]
            self._prenorm_bin_states = _gather_bin_features(
                cache.bin_features, cache.state_idxs, bin_indices
            )
        else:
            self._prenorm_bin_states = None

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

        self.nbins = len(self.hpGrid.lon)
        self.unique_nights = self._df['night'].unique()
        self.n_nights = self._df.groupby('night').ngroups

        if self.num_actions is None:
            action_space = self._action_space_str
            if action_space in ['radec', 'azel']:
                self.num_actions = self.nbins
            else:
                self.num_actions = self.nbins * self.num_filters

    # ------------------------------------------------------------------
    # Transition construction
    # ------------------------------------------------------------------

    def _build_transitions(self, action_space):
        states, bin_states = self._construct_states(
            df=self._df,
            bin_states=self._prenorm_bin_states,
            include_bin_features=self.include_bin_features,
            state_idxs=self.state_idxs,
        )
        num_transitions = len(self.next_state_idxs)

        actions = self._construct_actions(self._df, action_space=action_space,
                                          next_state_idxs=self.next_state_idxs)
        rewards = self._construct_rewards(self._df, next_state_idxs=self.next_state_idxs,
                                          reward=self.reward)
        dones = self._construct_dones(num_transitions=num_transitions,
                                      next_state_idxs=self.next_state_idxs,
                                      current_state_idxs=self.current_state_idxs)
        action_masks = self._construct_action_masks(
            state_df=self._df, action_space=action_space,
            num_states=len(self.state_idxs), state_idxs=self.state_idxs,
        )

        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.int32)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)
        self.dones = torch.as_tensor(dones, dtype=torch.bool)
        self.action_masks = torch.as_tensor(action_masks, dtype=torch.bool)
        self.num_transitions = num_transitions

        if self.include_bin_features:
            self._prenorm_bin_states = torch.as_tensor(bin_states, dtype=torch.float32)
        else:
            self._prenorm_bin_states = None

    def _construct_dones(self, num_transitions, next_state_idxs, current_state_idxs):
        dones = ~np.isin(next_state_idxs, current_state_idxs)
        dones[-1] = True
        return dones

    def _construct_states(self, df, bin_states, include_bin_features, state_idxs):
        global_states = self._construct_global_features(df=df, state_idxs=state_idxs)
        if not include_bin_features:
            bin_states = None
        return global_states, bin_states

    def _construct_global_features(self, df, state_idxs):
        missing_cols = set(self.global_feature_names) - set(df.columns)
        assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe.'
        return df.iloc[state_idxs][self.global_feature_names].to_numpy()

    def _construct_actions(self, df, action_space, next_state_idxs):
        assert action_space in ['radec', 'azel', 'radec_filter', 'azel_filter', 'filter']
        next_state_df = df.iloc[next_state_idxs]

        if self.hpGrid.is_azel:
            lonlat = next_state_df[['az', 'el']].values
        else:
            lonlat = next_state_df[['ra', 'dec']].values

        bin_indices = self.hpGrid.ang2idx(lon=lonlat[:, 0], lat=lonlat[:, 1])

        if 'filter' not in action_space:
            return bin_indices
        elif ('radec' not in action_space) and ('azel' not in action_space):
            return df.iloc[next_state_idxs]['filter'].map(FILTER2IDX).values.astype(np.int32)
        else:
            assert ZENITH_FILTER not in next_state_df['filter'].values, \
                f"Invalid data: Found '{ZENITH_FILTER}' in next_state_df."
            filter_indices = next_state_df['filter'].map(FILTER2IDX).values.astype(np.int32)
            return (bin_indices * _NUM_FILTERS) + filter_indices

    def _construct_rewards(self, df, next_state_idxs, reward):
        if reward == RewardStructure.TEFF:
            R_tot = df.iloc[next_state_idxs]['teff'].fillna(0).values
        elif reward == RewardStructure.EXPERT_ACTION:
            R_tot = np.ones(len(next_state_idxs), dtype=np.float32)
        elif reward == RewardStructure.COMPOSITE:
            rw = self.reward_weights
            R_slew = self._construct_slew_reward()
            R_airmass = self._construct_airmass_reward(df, next_state_idxs, rw)
            R_tsince = self._construct_t_since_reward(df, next_state_idxs, rw)
            R_tiling = self._construct_min_tiling_reward(df, next_state_idxs, rw)
            R_tot = (rw.w_slew * R_slew
                     + rw.w_airmass * R_airmass
                     + rw.w_t_last_visit * R_tsince
                     + rw.w_min_tiling * R_tiling).astype(np.float32)
        elif reward is None:
            return np.zeros(len(next_state_idxs), dtype=np.float32)
        else:
            raise NotImplementedError

        if self.reward_norm == 'minmax':
            R_tot = (R_tot - R_tot.min()) / (R_tot.max() - R_tot.min())
        elif self.reward_norm is not None:
            logger.warning(f"Unknown reward norm: {self.reward_norm}")
        return R_tot

    # def _construct_airmass_reward(self, df, next_state_idxs, rw):
    #     airmass = df.iloc[next_state_idxs]['airmass'].values
    #     return np.clip(
    #         (rw.airmass_limit - airmass) / (rw.airmass_limit - 1.0), 0.0, 1.0
    #     )

    def _construct_slew_reward(self):
        return 1.0 - self.slew_distances.numpy() / np.pi

    def _construct_t_since_reward(self, df, next_state_idxs, rw):
        t_diff = df.groupby(['field_id', 'filter'])['timestamp'].diff()
        t_since = t_diff.iloc[next_state_idxs].fillna(rw.t_ref_seconds).values
        t_min, t_max = t_since.min(), t_since.max()
        return (t_since - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(t_since)

    def _construct_min_tiling_reward(self, df, next_state_idxs, rw):
        field_ids = df.iloc[next_state_idxs]['field_id'].values.astype(int)
        filter_idxs = df.iloc[next_state_idxs]['filter'].map(FILTER2IDX).values.astype(int)
        visits_before = df.groupby(['field_id', 'filter']).cumcount().iloc[next_state_idxs].values
        target_visits = self.lookups.target_fidfilt_counts[field_ids, filter_idxs]
        safe_target = np.where(target_visits > 0, target_visits, 1)
        assert ZENITH_FILTER not in df.iloc[next_state_idxs]['filter'].values
        return np.where(
            target_visits > 0,
            np.clip(1.0 - visits_before / safe_target, 0.0, 1.0),
            0.0,
        )

    def _construct_action_masks(self, state_df, action_space, num_states, state_idxs):
        state_df = state_df.iloc[state_idxs]
        els = np.empty((num_states, self.nbins), dtype=np.float32)

        if action_space == 'filter':
            return np.ones((num_states, self.num_filters), dtype=np.bool_)

        if self._calculate_action_mask:
            logger.info("Calculating action masks based on horizon…")
            if not self.hpGrid.is_azel:
                lon, lat = self.hpGrid.lon, self.hpGrid.lat
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
                    self.hpGrid.lat[:, np.newaxis],
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

    def _normalize_states(self, mode, cfg, norm_kwargs, z_stats, rel_stats):
        global_normalizer = StateNormalizer(
            state_feature_names=self.global_feature_names, **norm_kwargs
        )
        bin_normalizer = StateNormalizer(
            state_feature_names=self.bin_feature_names, **norm_kwargs
        )

        if mode == 'train':
            self.states, self.global_zscore_stats, self.global_rel_stats, self.global_sentinel_mask = \
                global_normalizer.fit_transform(
                    state=self.states, train_state_idxs=self.train_state_idxs
                )
        else:
            self.states, self.global_sentinel_mask = global_normalizer.transform(
                state=self.states,
                z_stats_dict=(z_stats or {}).get('global_features', {}),
                rel_stats_dict=(rel_stats or {}).get('global_features', {}),
            )
            self.global_zscore_stats, self.global_rel_stats = None, None

        if self.include_bin_features and self._prenorm_bin_states is not None:
            bin_tensor = torch.as_tensor(self._prenorm_bin_states)
            if mode == 'train':
                self._prenorm_bin_states, self.bin_zscore_stats, self.bin_rel_stats, self.bin_sentinel_mask = \
                    bin_normalizer.fit_transform(
                        state=bin_tensor, train_state_idxs=self.train_state_idxs
                    )
            else:
                self._prenorm_bin_states, self.bin_sentinel_mask = bin_normalizer.transform(
                    state=bin_tensor,
                    z_stats_dict=(z_stats or {}).get('bin_features', {}),
                    rel_stats_dict=(rel_stats or {}).get('bin_features', {}),
                )
                self.bin_zscore_stats, self.bin_rel_stats = None, None
        else:
            self.bin_sentinel_mask = None
            self.bin_zscore_stats, self.bin_rel_stats = None, None

        # (n_states, n_bins): True where bin has no sentinel values at this timestep
        if self.bin_sentinel_mask is not None:
            self.active_bin_mask = ~self.bin_sentinel_mask.any(dim=-1)
        else:
            self.active_bin_mask = None

        if mode == 'train' and (self.global_zscore_stats or self.global_rel_stats):
            self._save_norm_stats(Path(cfg.outdir))

    def _save_norm_stats(self, save_dir):
        all_stats = {
            "z_score": {
                'global_features': self.global_zscore_stats,
                'bin_features': self.bin_zscore_stats,
            },
            "rel_norm": {
                'global_features': self.global_rel_stats,
                'bin_features': self.bin_rel_stats,
            },
            "sentinel_mask": {
                "global": (
                    self.global_sentinel_mask.any(dim=0).tolist()
                    if self.global_sentinel_mask is not None else []
                ),
                "bin": (
                    self.bin_sentinel_mask.any(dim=(0, 1)).tolist()
                    if self.bin_sentinel_mask is not None else []
                ),
            },
        }
        save_path = Path(save_dir) / "checkpoints" / "normalization_stats.json"
        with open(save_path, 'w') as f:
            json.dump(all_stats, f, indent=4)
        logger.info(f"Normalization stats saved to {save_path}")

    # ------------------------------------------------------------------
    # Tensor formatting & validation
    # ------------------------------------------------------------------

    def _format_tensors_for_network(self, network_type):
        if network_type == 'mlp':
            if self.include_bin_features and self._prenorm_bin_states is not None:
                if not isinstance(self.states, torch.Tensor):
                    self.states = torch.as_tensor(self.states, dtype=torch.float32)
                if not isinstance(self._prenorm_bin_states, torch.Tensor):
                    self._prenorm_bin_states = torch.as_tensor(
                        self._prenorm_bin_states, dtype=torch.float32
                    )
                bs_flat = self._prenorm_bin_states.reshape(
                    self._prenorm_bin_states.shape[0], -1
                )
                self.states = torch.cat([self.states, bs_flat], dim=1)
                self._prenorm_bin_states = None
                self.bin_states = None
            self.bin_state_dim = 0
            self.state_dim = self.states.shape[-1]
        else:
            self.state_dim = self.states.shape[-1]
            self.bin_states = self._prenorm_bin_states
            self.bin_state_dim = (
                self.bin_states.shape[-1] if self.include_bin_features and self.bin_states is not None else 0
            )

        self.dataset_dims = {
            'state_dim': self.state_dim,
            'bin_state_dim': self.bin_state_dim,
            'num_bins': self.nbins,
            'num_filters': self.num_filters,
            'num_actions': self.num_actions,
        }
        self.dataset_feature_names = {
            'global_features': self.global_feature_names,
            'bin_features': self.bin_feature_names,
        }

    def _validate_dataset(self):
        assert self.states.shape[0] == self.action_masks.shape[0], \
            "States and masks must be 1:1"
        assert (self.actions.shape[0] == self.rewards.shape[0]
                == self.dones.shape[0] == self.num_transitions), \
            (f"Transition mismatch: actions {self.actions.shape[0]}, "
             f"rewards {self.rewards.shape[0]}, dones {self.dones.shape[0]}")
        if self.include_bin_features and self.bin_states is not None:
            assert self.states.shape[0] == self.bin_states.shape[0], \
                f"State mismatch: global {self.states.shape[0]}, bin {self.bin_states.shape[0]}"

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        c_idx = self.curr_compact_idxs[idx]
        n_idx = self.next_compact_idxs[idx]
        is_done = self.dones[idx].item()

        _zero_bin = (
            torch.zeros_like(self.bin_states[0])
            if (self.include_bin_features and self.bin_states is not None)
            else torch.as_tensor(0)
        )
        bin_c = self.bin_states[c_idx] if (self.include_bin_features and self.bin_states is not None) else torch.as_tensor(0)
        bin_n = (self.bin_states[n_idx] if not is_done else _zero_bin) \
            if (self.include_bin_features and self.bin_states is not None) else torch.as_tensor(0)

        return (
            self.states[c_idx],
            self.actions[idx],
            self.rewards[idx],
            self.states[n_idx] if not is_done else torch.zeros_like(self.states[0]),
            self.dones[idx],
            self.action_masks[c_idx],
            self.action_masks[n_idx] if not is_done else torch.zeros_like(self.action_masks[0]),
            bin_c,
            bin_n,
            self.slew_distances[idx],
        )

    def get_norm_stats(self) -> dict:
        return {
            "z_score": {
                'global_features': self.global_zscore_stats,
                'bin_features': self.bin_zscore_stats,
            },
            "rel_norm": {
                'global_features': self.global_rel_stats,
                'bin_features': self.bin_rel_stats,
            },
        }


# ---------------------------------------------------------------------------
# OfflineDataset — light DataLoader wrapper
# ---------------------------------------------------------------------------

class OfflineDataset:
    """Thin wrapper that creates train/val DataLoaders from a ``TransitionDataset``."""

    def __init__(
        self,
        dataset: TransitionDataset,
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
