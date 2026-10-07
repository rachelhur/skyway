import torch
import numpy as np
import logging

from blancops.configs.constants import _FILTER_DEP_FEATURE_NAMES
from blancops.configs.experiment_schema import NormalizationConfig
from blancops.configs.enums import has_filter
from blancops.survey.profiles import DES, SurveyProfile

logger = logging.getLogger(__name__)

def build_normalizer(state_feature_names, cfg, survey: SurveyProfile = DES):
    norm_kwargs = build_normalizer_kwargs(cfg.data.norm, has_filter(cfg.data.action_space), survey=survey)
    return StateNormalizer(state_feature_names=state_feature_names, survey=survey, **norm_kwargs)

def build_normalizer_kwargs(norm_config: NormalizationConfig, do_filt=True, survey: SurveyProfile = DES) -> dict:
    """Translates the Pydantic schema into the exact kwargs expected by StateNormalizer."""
    kwargs = {
        'cyclical_feature_names': [],
        'sin_norm_feature_names': [],
        'log_norm_feature_names': [],
        'fractional_norm_feature_names': [],
        'z_score_feature_names': [],
        'local_mean_z_score_feature_names': [],
        'local_z_score_feature_names': [],
    }

    name_map = {
        'cyclical': 'cyclical_feature_names',
        'sin': 'sin_norm_feature_names',
        'log': 'log_norm_feature_names',
        'fractional': 'fractional_norm_feature_names',
        'z_score': 'z_score_feature_names',
        'local_mean_z': 'local_mean_z_score_feature_names',
        'local_z_score': 'local_z_score_feature_names',
    }

    for feature, requested_norms in norm_config.feature_norm_mappings.items():
        if do_filt and feature in _FILTER_DEP_FEATURE_NAMES:
            feat_names = [f"{feature}_{filt}" for filt in survey.filters]
        else:
            feat_names = [feature]
        for feat_name in feat_names:
            for norm in requested_norms:
                if norm is not None:
                    kwargs[name_map[norm]].append(feat_name)
    kwargs['do_cyclical_norm'] = len(kwargs['cyclical_feature_names']) > 0
    kwargs['do_sin_norm'] = len(kwargs['sin_norm_feature_names']) > 0
    kwargs['do_log_norm'] = len(kwargs['log_norm_feature_names']) > 0
    kwargs['do_fractional_norm'] = len(kwargs['fractional_norm_feature_names']) > 0
    kwargs['do_z_score_norm'] = len(kwargs['z_score_feature_names']) > 0
    kwargs['do_local_mean_z_score'] = len(kwargs['local_mean_z_score_feature_names']) > 0
    kwargs['do_local_z_score'] = len(kwargs['local_z_score_feature_names']) > 0

    # Pass through the fix_nans flag
    kwargs['fix_nans'] = norm_config.fix_nans

    return kwargs

def expand_feature_set(feature_names, cyclical_feature_names, do_filt=True, survey: SurveyProfile = DES):
    feature_names_out = []
    for feat_name in feature_names:
        has_filt_dep = do_filt and feat_name in _FILTER_DEP_FEATURE_NAMES
        if do_filt:
            # has_filt_dep = feat_name in _FILTER_DEP_FEATURE_NAMES
            if has_filt_dep:
                [feature_names_out.append(f"{feat_name}_{filt}") for filt in survey.filters]

        is_rel_feat = feat_name.startswith('rel_')
        is_delta_feat = feat_name.startswith('delta_')
        never_cyclic_feat = is_rel_feat or is_delta_feat
        is_cyclic = any((feat_name == cyc_feat) or feat_name.endswith(f"_{cyc_feat}") for cyc_feat in cyclical_feature_names)

        is_cyclic = is_cyclic and not never_cyclic_feat
        if is_cyclic:
            logger.debug(f"Expanding {feat_name} to {feat_name}_cos and {feat_name}_sin")
            feature_names_out.extend([f"{feat_name}_cos", f"{feat_name}_sin"])
        if not has_filt_dep and not is_cyclic:
            feature_names_out.append(feat_name)
    return feature_names_out


def _base_feature_name(name: str, survey: SurveyProfile = DES) -> str:
    # Strip filter suffix
    for filt in survey.filters:
        if name.endswith(f"_{filt}"):
            name = name[: -(len(filt) + 1)]
            break
    return name


def setup_feature_names(base_global_feature_names, base_bin_feature_names, cyclical_feature_names, do_cyclical_norm, do_filt,
                        survey: SurveyProfile = DES):
    """Expands feature list to include filter dependence and cyclical normalizations where applicable."""
    if do_cyclical_norm:
        global_feature_names = expand_feature_set(base_global_feature_names.copy(), cyclical_feature_names, do_filt, survey)
        bin_feature_names = expand_feature_set(base_bin_feature_names.copy(), cyclical_feature_names, do_filt, survey)
    else:
        global_feature_names = base_global_feature_names.copy()
        bin_feature_names = base_bin_feature_names.copy()
    return global_feature_names, bin_feature_names

def apply_cyclical_features(features, base_names, cyclical_names):
    for name in base_names:
        if name.startswith(("rel_", "delta_")):
            continue

        if name not in features:
            continue

        if any(name == cyc or name.endswith(f"_{cyc}") for cyc in cyclical_names):
            features[f"{name}_cos"] = np.cos(features[name])
            features[f"{name}_sin"] = np.sin(features[name])


def _mask_cols(mask):
    """Integer column indices of a boolean feature mask."""
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    return np.where(np.asarray(mask))[0]


def _column_stats(state, train_state_idxs, mask, is_torch):
    """NaN-aware mean and population std per masked column, over train rows.

    Runs one column at a time so the masked copy of the whole state array is
    never materialized; at nside 32 that copy is tens of GB.

    Args:
        state: (..., n_features) array or tensor.
        train_state_idxs: Row indices contributing to the statistics.
        mask: Boolean mask over the feature axis.
        is_torch: Whether state is a torch tensor.

    Returns:
        Tuple of per-column mean and std, ordered as the masked columns.
    """
    idxs = torch.as_tensor(train_state_idxs) if is_torch else train_state_idxs
    means, stds = [], []
    for c in _mask_cols(mask):
        col = state[..., c][idxs]
        if is_torch:
            mn = torch.nanmean(col)
            sd = torch.clamp(torch.sqrt(torch.nanmean((col - mn) ** 2)), min=1e-6)
        else:
            mn = np.nanmean(col)
            sd = np.clip(np.nanstd(col), a_min=1e-6, a_max=None)
        means.append(mn)
        stds.append(sd)
    if is_torch:
        return torch.stack(means), torch.stack(stds)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def _scale_columns_inplace(state, mask, mean, std):
    """Apply ``(x - mean) / std`` to masked columns in place, one column at a time.

    ``state[..., c]`` is a view for both backends, so the arithmetic writes
    straight into ``state`` without allocating a copy of the masked block.

    Args:
        state: Array or tensor, modified in place.
        mask: Boolean mask over the feature axis.
        mean: Per-column means, or None to divide only.
        std: Per-column standard deviations.
    """
    for i, c in enumerate(_mask_cols(mask)):
        col = state[..., c]
        if mean is not None:
            col -= mean[i]
        col /= std[i]


class StateNormalizer:
    def __init__(
        self,
        state_feature_names,
        sin_norm_feature_names,
        log_norm_feature_names,
        fractional_norm_feature_names,
        z_score_feature_names,
        local_mean_z_score_feature_names,
        local_z_score_feature_names=None,
        do_sin_norm=True,
        do_log_norm=True,
        do_fractional_norm=True,
        do_z_score_norm=True,
        do_local_mean_z_score=True,
        do_local_z_score=False,
        fix_nans=True,
        do_cyclical_norm=None,
        cyclical_feature_names=None,
        sentinel_value=-1,
        survey: SurveyProfile = DES,
    ):
        self.feature_names = state_feature_names
        self.survey = survey

        # Config Flags
        self.do_sin = do_sin_norm
        self.do_log = do_log_norm
        self.do_frac = do_fractional_norm
        self.do_z = do_z_score_norm
        self.do_rel = do_local_mean_z_score
        self.do_local_z = do_local_z_score
        self.do_cyclical_norm = do_cyclical_norm
        self.cyclical_feature_names = cyclical_feature_names
        self.sentinel_value = sentinel_value
        self.fix_nans = fix_nans

        # Pre-compute masks ONCE during initialization
        self._build_masks(
            sin_norm_feature_names,
            log_norm_feature_names,
            fractional_norm_feature_names,
            z_score_feature_names,
            local_mean_z_score_feature_names,
            local_z_score_feature_names or [],
        )

    def _build_masks(self, sin_feats, log_feats, frac_feats, z_feats, rel_feats, local_z_feats):
        """Build boolean masks for each normalization type."""
        names = self.feature_names

        def matches(feat, allowed):
            return feat in allowed or _base_feature_name(feat, self.survey) in allowed

        self.masks = {
            'sin': np.array([matches(f, sin_feats) for f in names]),
            'log': np.array([matches(f, log_feats) for f in names]),
            'frac': np.array([matches(f, frac_feats) for f in names]),
            'z': np.array([matches(f, z_feats) for f in names]),
            'rel': np.array([matches(f, rel_feats) for f in names]),
            'local_z': np.array([matches(f, local_z_feats) for f in names]),
        }

        self.active_features = {
            name: [f for f, m in zip(names, mask) if m]
            for name, mask in self.masks.items()
        }


    def _get_backend(self, state):
        """Returns the appropriate math module and converts masks to the correct device."""
        is_torch = torch.is_tensor(state)
        math_backend = torch if is_torch else np

        # Convert pre-computed numpy masks to torch bool tensors if necessary
        active_masks = {}
        for key, mask in self.masks.items():
            if is_torch:
                active_masks[key] = torch.tensor(mask, dtype=torch.bool, device=state.device)
            else:
                active_masks[key] = mask

        return is_torch, math_backend, active_masks

    def fit_transform(self, state, train_state_idxs):
        """TRAINING MODE: Calculates stats from training indices, applies them, and returns the stats dictionaries."""
        if train_state_idxs is None:
            raise ValueError("train_state_idxs must be provided in fit_transform mode.")

        is_torch, backend, m = self._get_backend(state)
        self._apply_stateless_norms(state, backend, m)

        z_stats_out, rel_stats_out = {}, {}

        # 1. Z-Score (Global Mean/Std)
        if self.do_z and m['z'].sum() > 0:
            mean, std = _column_stats(state, train_state_idxs, m['z'], is_torch)
            _scale_columns_inplace(state, m['z'], mean, std)
            z_stats_out = self._build_stats_dict(self.active_features['z'], mean, std)

        # 2. Relative Local Mean Z-Score (Global Std only)
        if self.do_rel and m['rel'].sum() > 0:
            mean, std = _column_stats(state, train_state_idxs, m['rel'], is_torch)
            _scale_columns_inplace(state, m['rel'], None, std)
            rel_stats_out = self._build_stats_dict(self.active_features['rel'], mean, std)

        # 3. Local Z-Score (per-sample std across bins — no global stats stored)
        if self.do_local_z and m['local_z'].sum() > 0:
            self._apply_local_z_score(state, m['local_z'], is_torch)

        if self.fix_nans:
            assert not np.isnan(self.sentinel_value), "sentinel_value must not be NaN"
            nan_mask = backend.isnan(state)
            state[nan_mask] = self.sentinel_value

        return state, z_stats_out, rel_stats_out, nan_mask

    def transform(self, state, z_stats_dict, rel_stats_dict):
        """INFERENCE MODE: Applies previously calculated stats dictionaries to the state."""
        is_torch, backend, m = self._get_backend(state)
        self._apply_stateless_norms(state, backend, m)

        # 1. Apply Z-Score
        if self.do_z and m['z'].sum() > 0:
            mean, std = self._extract_stats_arrays(z_stats_dict, self.active_features['z'], backend, state)
            _scale_columns_inplace(state, m['z'], mean, std)

        # 2. Apply Relative Norm
        if self.do_rel and m['rel'].sum() > 0:
            _, std = self._extract_stats_arrays(rel_stats_dict, self.active_features['rel'], backend, state)
            _scale_columns_inplace(state, m['rel'], None, std)

        # 3. Local Z-Score (per-sample std across bins — no stored stats needed)
        if self.do_local_z and m['local_z'].sum() > 0:
            self._apply_local_z_score(state, m['local_z'], is_torch)

        nan_mask = backend.isnan(state) if self.fix_nans else None
        if self.fix_nans:
            assert not np.isnan(self.sentinel_value), "sentinel_value must not be NaN"
            state[nan_mask] = self.sentinel_value
        return state, nan_mask

    def _apply_stateless_norms(self, state, backend, m):
        if self.do_sin and m['sin'].sum() > 0:
            state[..., m['sin']] = backend.sin(state[..., m['sin']])
        if self.do_log and m['log'].sum() > 0:
            state[..., m['log']] = backend.log(state[..., m['log']] + 1e-9)
        if self.do_frac and m['frac'].sum() > 0:
            state[..., m['frac']] = 2 * (state[..., m['frac']] - 0.5)

    def _apply_local_z_score(self, state, mask, is_torch):
        # Only meaningful for 3D bin-feature tensors (N, N_bins, N_feats).
        # Silently skips for 2D global-feature tensors.
        if state.ndim != 3:
            return
        local_data = state[..., mask]  # (N, N_bins, K)
        if is_torch:
            lmean = torch.nanmean(local_data, dim=-2, keepdim=True)  # (N, 1, K)
            lvar = torch.nanmean((local_data - lmean) ** 2, dim=-2, keepdim=True)
            lstd = torch.clamp(torch.sqrt(lvar), min=1e-6)
        else:
            lmean = np.nanmean(local_data, axis=-2, keepdims=True)
            lstd = np.clip(np.nanstd(local_data, axis=-2, keepdims=True), a_min=1e-6, a_max=None)
        state[..., mask] = local_data / lstd

    def _calc_std(self, flat_data, mean, backend, is_torch):
        """Population std, NaN-aware, min-clipped to avoid zero-division. Unified across backends."""
        # var = backend.nanmean((flat_data - mean) ** 2, dim=0) if is_torch \
        #     else np.nanmean((flat_data - mean) ** 2, axis=0)
        # std = backend.sqrt(var)
        # return torch.clamp(std, min=1e-6) if is_torch else np.maximum(std, 1e-6)
        if is_torch:
            var = torch.nanmean((flat_data - mean)**2, dim=0)
            return torch.clamp(torch.sqrt(var), min=1e-6)
        else:
            return np.clip(np.nanstd(flat_data, axis=0), a_min=1e-6, a_max=None)

    def _build_stats_dict(self, active_features, mean_arr, std_arr):
        """Converts internal tensors/arrays to standard Python floats for JSON serialization."""
        return {
            feat: {'mean': float(m), 'std': float(s)}
            for feat, m, s in zip(active_features, mean_arr, std_arr)
        }

    def _extract_stats_arrays(self, stats_dict, active_features, backend, state):
        """Pulls stats from JSON-loaded dicts and formats them for math operations."""
        means, stds = [], []
        for feat in active_features:
            if feat not in stats_dict:
                raise KeyError(f"CRITICAL: Model expects normalization stats for '{feat}', but it is missing!")
            means.append(stats_dict[feat]['mean'])
            stds.append(stats_dict[feat]['std'])

        if backend == torch:
            return (
                torch.tensor(means, dtype=torch.float32, device=state.device),
                torch.tensor(stds, dtype=torch.float32, device=state.device)
            )
        return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)

    def inverse_transform(self, state, z_stats_dict=None, rel_stats_dict=None, nan_mask=None):
        """
        Reverses fit_transform / transform.

        Args:
            state: Normalized state (modified in place).
            z_stats_dict: Stats from forward z-score. Required iff any 'z' features are active.
            rel_stats_dict: Stats from forward rel norm. Required iff any 'rel' features are active.
            nan_mask: Optional boolean mask of where sentinels were written in the forward pass.
                    If provided (and self.fix_nans), those positions are restored to NaN before
                    any inverse arithmetic runs.

        Returns:
            state: Denormalized state.

        Notes:
            - sin inverse is lossy: arcsin only recovers angles in [-pi/2, pi/2].
            - Cyclical (cos, sin) expansion is NOT undone here.
            - local_z_score is NOT inverted: per-sample stds are not persisted.
            - Assumes each feature appears in at most one norm-type mask.
        """
        is_torch, backend, m = self._get_backend(state)

        # 1. Restore NaNs FIRST so sentinels aren't fed into exp/multiply/etc.
        if self.fix_nans and nan_mask is not None:
            if is_torch and not torch.is_tensor(nan_mask):
                nan_mask = torch.tensor(nan_mask, dtype=torch.bool, device=state.device)
            state[nan_mask] = float('nan')

        # 2. Undo stateful norms (reverse of forward order: forward was z then rel)
        if self.do_rel and m['rel'].sum() > 0:
            if rel_stats_dict is None:
                raise ValueError("rel_stats_dict is required to invert rel normalization.")
            _, std = self._extract_stats_arrays(rel_stats_dict, self.active_features['rel'], backend, state)
            state[..., m['rel']] = state[..., m['rel']] * std

        if self.do_z and m['z'].sum() > 0:
            if z_stats_dict is None:
                raise ValueError("z_stats_dict is required to invert z-score normalization.")
            mean, std = self._extract_stats_arrays(z_stats_dict, self.active_features['z'], backend, state)
            state[..., m['z']] = state[..., m['z']] * std + mean

        # 3. Undo stateless norms (masks are disjoint per feature, so within-step order doesn't matter)
        self._inverse_stateless_norms(state, backend, m, is_torch)

        return state

    def _inverse_stateless_norms(self, state, backend, m, is_torch):
        if self.do_frac and m['frac'].sum() > 0:
            state[..., m['frac']] = state[..., m['frac']] / 2 + 0.5
        if self.do_log and m['log'].sum() > 0:
            state[..., m['log']] = backend.exp(state[..., m['log']]) - 1e-9
        if self.do_sin and m['sin'].sum() > 0:
            # Clamp to [-1, 1] to absorb numerical drift, then arcsin.
            if is_torch:
                state[..., m['sin']] = torch.arcsin(torch.clamp(state[..., m['sin']], min=-1.0, max=1.0))
            else:
                state[..., m['sin']] = np.arcsin(np.clip(state[..., m['sin']], -1.0, 1.0))


    def inverse_transform_df(self, df, feature_names=None,
                            z_stats_dict=None, rel_stats_dict=None, drop_cyclical_components=False):
        """
        Inverse-normalize columns of a DataFrame in place.

        Args:
            df: DataFrame containing normalized columns.
            feature_names: Iterable restricting which columns are inverted.
                If None, every active feature whose name matches a column in
                df will be inverted. Pass the explicit list when df also
                holds raw-unit columns that must NOT be touched.
            z_stats_dict, rel_stats_dict: forward-pass stats. Required iff
                any z/rel-active feature is in the inversion set.

        Caveats:
            - sin inverse is lossy (arcsin only recovers [-pi/2, pi/2]).
            - NaN sentinels in the input are NOT recovered as NaN; they
            propagate as ordinary numbers through the inverse. If the
            runner ever saves a nan_mask per column, plumb it in and set
            those cells to NaN before any arithmetic.
            - Cyclical (cos, sin) expansion is not undone here.
            - Assumes each feature appears in at most one norm-type list.
        """
        if feature_names is None:
            target = None  # match anything
        else:
            target = set(feature_names)

        def _want(feat):
            return feat in df.columns and (target is None or feat in target)

        # Reverse forward order: stateful first (rel, then z), then stateless.
        if self.do_rel:
            if rel_stats_dict is None and any(_want(f) for f in self.active_features['rel']):
                raise ValueError("rel_stats_dict required to invert rel-normalized features.")
            for feat in self.active_features['rel']:
                if _want(feat):
                    if feat not in rel_stats_dict:
                        raise KeyError(f"rel_stats_dict missing '{feat}'")
                    df[feat] = df[feat].to_numpy() * rel_stats_dict[feat]['std']

        if self.do_z:
            if z_stats_dict is None and any(_want(f) for f in self.active_features['z']):
                raise ValueError("z_stats_dict required to invert z-normalized features.")
            for feat in self.active_features['z']:
                if _want(feat):
                    if feat not in z_stats_dict:
                        raise KeyError(f"z_stats_dict missing '{feat}'")
                    s = z_stats_dict[feat]['std']
                    m = z_stats_dict[feat]['mean']
                    df[feat] = df[feat].to_numpy() * s + m

        if self.do_frac:
            for feat in self.active_features['frac']:
                if _want(feat):
                    df[feat] = df[feat].to_numpy() / 2 + 0.5

        if self.do_log:
            for feat in self.active_features['log']:
                if _want(feat):
                    df[feat] = np.exp(df[feat].to_numpy()) - 1e-9

        if self.do_sin:
            for feat in self.active_features['sin']:
                if _want(feat):
                    df[feat] = np.arcsin(np.clip(df[feat].to_numpy(), -1.0, 1.0))

        if self.do_cyclical_norm and self.cyclical_feature_names:
            df = inverse_cyclical_norm(
                df=df, target=target, cyclical_feature_names=self.cyclical_feature_names,
                drop_cyclical_components=drop_cyclical_components)

        return df

def inverse_cyclical_norm(df, cyclical_feature_names, *,
                          target=None,
                          drop_cyclical_components=False,
                          wrap_to_positive=frozenset({'az', 'ra', 'lst'})):
    cyclical_wrap_to_positive = {'az', 'ra', 'lst'}

    for cyc_base in cyclical_feature_names:
        for col in list(df.columns):
            if not col.endswith('_cos'):
                continue
            prefix = col[:-len('_cos')]
            # Match: prefix == cyc_base (e.g. 'lst') OR prefix endswith '_<cyc_base>' (e.g. 'sun_ra').
            if not (prefix == cyc_base or prefix.endswith(f"_{cyc_base}")):
                continue
            sin_col = f"{prefix}_sin"
            if sin_col not in df.columns:
                continue
            if target is not None and col not in target and sin_col not in target:
                continue
            # Don't clobber a fresh-computed angle column already in df
            # (e.g. bin_ra was set by _get_bin_coords before this loop runs).
            if prefix in df.columns:
                continue

            angle = np.arctan2(df[sin_col].to_numpy(), df[col].to_numpy())
            if cyc_base in cyclical_wrap_to_positive:
                angle = np.mod(angle, 2 * np.pi)
            df[prefix] = angle

            if drop_cyclical_components:
                df.drop(columns=[col, sin_col], inplace=True)

    return df

def normalize_timestamp(timestamp, sunset_timestamp, sunrise_timestamp):
    return (timestamp - sunset_timestamp) / (sunrise_timestamp - sunset_timestamp)
