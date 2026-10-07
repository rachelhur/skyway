"""Base class for all Blanco telescope environments.

Owns shared physics, feature calculation, action/observation spaces, and the
gym contract. Subclasses customize behavior through a small set of abstract
lifecycle hooks (`_begin_episode`, `_advance_after_action`,
`_episode_terminated`) plus optional feature-context hooks
(`_get_t_survey`, `_get_fwhm`, etc.) that default to returning None.

Time and per-night state arrive through `StateSnapshot` objects; the same
primitive is used by `OfflineBlancoEnv._start_new_night` (per-night init)
and by `OnlineBlancoEnv.sync_telemetry` (mid-episode hardware sync).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Optional
from dataclasses import dataclass
from einops import rearrange
import numpy as np
import gymnasium as gym
from collections import defaultdict
from blancops import math
from blancops.configs.enums import RewardTerm
from blancops.configs.experiment_schema import ActionConstraints, ExperimentConfig, RLAlgConfig
from blancops.data.rewards import combine_rewards, normalize_rewards, uniformity_inputs
from blancops.data_quality.teff import predict_teff
from blancops.survey.profiles import DES, SurveyProfile
from blancops.data.features.bin_features import (
    # Shared per-timestep helpers — single source of truth for bin features.
    _STALENESS_BASE_KEYS,
    _SURVEY_PROGRESS_BASE_KEYS,
    compute_bin_progress_features,
    apply_relative_bin_features,
    validate_history_bin_features,
)
from blancops.data.features.candidate_features import compute_candidate_ephemeris_features
from blancops.data.features.glob_features import (
    # Shared per-timestep helpers — single source of truth for global features.
    compute_global_time_only_features,
    compute_global_pointing_features,
    calc_urgency,
    compute_global_tracker_features,
)
from blancops.data.features.normalizations import StateNormalizer, apply_cyclical_features, build_normalizer_kwargs, normalize_timestamp, setup_feature_names
from blancops.data.norm_stats import NormStats
from blancops.data.features.field_features import (
    FieldGrid, build_field_normalizer, compute_field_features, expand_field_feature_names, stack_field_features,
)
from blancops.environment.survey_tracker import SurveyProgressTracker
from blancops.environment.seeing_model import SeeingModel
from blancops.math import units
from blancops.ephemerides import astropy_ephem, ephemerides
from blancops.telescope.base import TelescopeProfile
from blancops.configs.constants import *
from blancops.configs.enums import grid_is_azel, has_filter, is_field_level

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Snapshot holds a point-in-time snapshot of the env's mutable state
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    """A point-in-time snapshot of the env's mutable state.
    Counters are optional; pass None to leave the existing value alone
    """
    timestamp: float
    field_id: int = ZENITH_FIELD_ID
    bin_num: int = ZENITH_BIN_NUM
    filter_idx: int = ZENITH_FILTER_IDX
    counts_cur : Optional[np.ndarray] = None # Can have shape (nfields,) or (nfields, NUM_FILTERS) depending on action space
    last_visit_ot_cur: Optional[np.ndarray] = None    # Can have shape (nfields,) or (nfields, NUM_FILTERS) depending on action space


# ---------------------------------------------------------------------------
# Feature → source requirements (consumed by _validate_feature_config)
# ---------------------------------------------------------------------------
#
# Each entry maps a feature name to a list of (kind, target) checks:
#
#   ('hook', name) — type(self) must override the named method
#   ('attr', name) — getattr(self, name) must be non-None at validation
#   ('flag', name) — getattr(self, name) must be truthy at validation
#                    (used to enforce do_filt for 2D-only features)
#
# Features not listed here are assumed to be unconditionally computable.

# _STALENESS_BIN_REQUIREMENTS = [("attr", "lookups.total_ot_sec")]

def _feature_requirements(filters) -> dict[str, list[tuple[str, str]]]:
    """Requirements of each conditionally computable global feature, with per-filter names for ``filters``.

    Parameters
    ----------
    filters : sequence of str
        Survey filters that suffix the per-filter feature names.

    Returns
    -------
    dict
        Feature name -> list of (kind, target) requirements.
    """
    return {
        "fwhm": [("attr", "_seeing_model")],
        "t_survey": [("hook", "_get_t_survey")],
        # Works in both 1D and 2D tracker modes - only attr check.
        "global_mean_tiling": [("attr", "_survey_progress_tracker")],
        **{
            f"survey_progress_{f}": [
                ("attr", "_survey_progress_tracker"),
                ("flag", "do_filt"),
            ]
            for f in filters
        },
        **{
            f"urgency_{f}": [
                ("attr", "_survey_progress_tracker"),
                ("flag", "do_filt"),
                ("hook", "_get_survey_night_idx"),
                ("hook", "_get_survey_nights_total"),
            ]
            for f in filters
        },
        **{
            f"global_mean_tiling_{f}": [
                ("attr", "_survey_progress_tracker"),
                ("flag", "do_filt"),
            ]
            for f in filters
        },
    }


class BaseBlancoEnv(gym.Env, ABC):
    """Abstract base for all Blanco environments.

    Owns physics (airmass / slew), feature calculation, action masks,
    observation/action spaces, normalization, and the gym `step` / `reset`
    contract. Subclasses fill in the lifecycle hooks declared below.
    """
    def __init__(
        self,
        cfg: ExperimentConfig,
        constraints_cfg: ActionConstraints,
        lookups,
        norm_stats: NormStats,
        telescope: TelescopeProfile | None = None,
        survey: SurveyProfile = DES,
    ):
        super().__init__()
        # Configuration, Normalizations, and Lookups
        self.cfg = cfg
        self.lookups = lookups
        self._norm_stats = norm_stats
        self.airmass_limit = constraints_cfg.airmass_limit
        self.airmass_failsafe = constraints_cfg.airmass_failsafe
        self.sun_el_limit = constraints_cfg.sun_el_limit

        # Telescope profile — single source for site/hardware geometry. The
        # mount's HA/Dec operational envelope (equatorial_limit) gates the
        # action mask; it is None for alt-az profiles, in which case the check
        # is skipped. ActionConstraints still owns the runtime-mutable airmass /
        # sun limits (see set_constraints).
        self._telescope = telescope if telescope is not None else survey.telescope
        self._survey = survey
        survey.check_telescope(self._telescope)
        survey.check_lookups(lookups)

        # Reward: same terms as the offline dataset, scaled with the training reward stats.
        # _step_record is set by _advance_after_action (None on WAIT).
        self._reward_cfg = cfg.model.reward if isinstance(cfg.model, RLAlgConfig) else None
        self._band_threshold = survey.acceptance_thresholds(cfg.data.acceptance)  # [n_filters] minimum accepted teff
        self._step_record: dict | None = None

        # Feature Configs
        norm_kwargs = build_normalizer_kwargs(cfg.data.norm, survey=survey)
        self.base_global_feature_names = list(cfg.data.global_features)
        self.base_bin_feature_names = list(cfg.data.bin_features)
        self.global_feature_names, self.candidate_feature_names = setup_feature_names(
            self.base_global_feature_names,
            self.base_bin_feature_names,
            norm_kwargs['cyclical_feature_names'],
            norm_kwargs['do_cyclical_norm'],
            do_filt=has_filter(cfg.data.action_space),
            survey=survey,
        )
        self.include_candidate_features = cfg.data.candidate_state_dim > 0
        self.do_filt = has_filter(cfg.data.action_space)

        # Normalizers
        self.global_normalizer = StateNormalizer(state_feature_names=self.global_feature_names, survey=survey, **norm_kwargs)
        self.candidate_normalizer = StateNormalizer(state_feature_names=self.candidate_feature_names, survey=survey,
                                                    **norm_kwargs)
        self.do_cyclical_norm = self.global_normalizer.do_cyclical_norm or self.candidate_normalizer.do_cyclical_norm

        self._has_historical_features = any(
            feat_substr in bf
            for feat_substr in (_SURVEY_PROGRESS_BASE_KEYS + _STALENESS_BASE_KEYS)
            for bf in self.candidate_feature_names
        )

        self.idx2filter = survey.idx2filter
        self.nfilters = survey.num_filters

        # Heapix Grid
        self.hpGrid = ephemerides.HealpixGrid(
            nside=cfg.data.nside,
            is_azel=grid_is_azel(cfg.data.action_space)
        )
        self.nbins = len(self.hpGrid.idx_lookup)

        # Field arrays sourced from lookups.fields, the canonical
        # per-field DataFrame indexed by field_id. The contract that
        # `field_id` doubles as a 0..N-1 array index is enforced in
        # LookupTables.__post_init__.
        self._set_field_arrays(lookups)

        # field_filter: candidates are survey fields
        self.field_level = is_field_level(cfg.data.action_space)
        self._field_feature_cache: tuple | None = None
        # Replay mode (expert replay checks): accept commands outside the current mask
        self._replay_mode = False
        if self.field_level:
            self.candidate_feature_names = expand_field_feature_names(list(cfg.data.field_features), survey=survey)
            self.candidate_normalizer = build_field_normalizer(self.candidate_feature_names)
            self._field_grid = FieldGrid(self._ra_arr, self._dec_arr)

        # Mutable runtime state — populated by reset() via _begin_episode
        self._ts: float | None = None
        self._zenith_ts: float | None = None
        self._field_id: int = ZENITH_FIELD_ID
        self._bin_num: int = ZENITH_BIN_NUM
        self._filter_idx: int = ZENITH_FILTER_IDX
        self._last_az: float = 0.0        # parked az (zenith default)
        self._last_el: float = np.pi / 2  # parked el (zenith default)
        self._sunset_ts: float | None = None
        self._sunrise_ts: float | None = None
        self._night_end_ts: float | None = None
        self._global_state: np.ndarray | None = None
        self._candidate_state: np.ndarray | None = None
        self._action_mask: np.ndarray | None = None
        self._is_new_night: bool = False
        self._valid_fields_per_bin: dict | None = None
        self._visible_bin_mask: np.ndarray | None = None

        # Coordinate-system-specific caches (RA/Dec mode only). Populated
        # lazily by _compute_bin_assignments on first use.
        self._field_bins_radec: np.ndarray | None = None
        self._field_bins_radec_v_mask: np.ndarray | None = None
        self._bins_membership_arr: list[np.ndarray] | None = None
        self._active_bins_s: np.ndarray | None = None

        self._last_glob_nan_mas: np.ndarray | None = None
        self._last_candidate_nan_mask: np.ndarray | None = None

        # Seeing strategy. Subclasses assign a concrete SeeingModel
        # (constant for the forward sim, predictive for live/historic). None
        # until set, so `_get_fwhm` is a no-op for envs that don't use it.
        self._seeing_model: SeeingModel | None = None

        # Survey progress tracker - built once at construction so that concrete subclasses can validate it
        target_counts = self.lookups.target_fidfilt_counts if self.do_filt else self.lookups.target_fid_counts
        self._survey_progress_tracker = SurveyProgressTracker(target_counts=target_counts)
        self._last_visit_ot = np.full(
            self._survey_progress_tracker.raw_counts.shape,
            np.nan, dtype=np.float64,
        )

        # Sentinel for inactive bins, written into history features at the
        # tail end of `_calculate_bin_features`. Internal computation uses
        # NaN; this is the external (on-disk) value.
        self._bin_feat_sentinel = (
            AZEL_BIN_FEAT_SENTINEL if self.hpGrid.is_azel
            else RADEC_BIN_FEAT_SENTINEL
        )

        # Fail-fast
        self._setup_action_and_obs_spaces(cfg.data.state_dim, cfg.data.candidate_state_dim)

        # Validation is NOT called here; concrete subclasses call
        # self._validate_feature_config() at the end of their __init__.

    def _set_field_arrays(self, lookups) -> None:
        """Populate field-id / RA / Dec arrays from a lookups catalog.

        Shared by construction and by LiveBlancoEnv.refresh_lookups so a
        mid-session catalog merge updates the same derived arrays.
        """
        self._fids = lookups.fields.index.to_numpy(dtype=np.int32)   # [n_fields]
        self._ra_arr = lookups.fields["ra"].to_numpy()               # [n_fields]
        self._dec_arr = lookups.fields["dec"].to_numpy()             # [n_fields]
        self.nfields = len(self._fids)
        # Exact (RA, Dec) -> field_id, keyed on the same float64 values the observation publishes
        self._coord_to_field = {(float(r), float(d)): int(f) for r, d, f in zip(self._ra_arr, self._dec_arr, self._fids)}
        if getattr(self, 'field_level', False):
            self._field_grid = FieldGrid(self._ra_arr, self._dec_arr)
            self._field_feature_cache = None

    # -----------------------------------------------------------------------
    # Gym contract — template methods. The shape of step/reset is fixed
    # here; subclasses customize via the abstract hooks below.
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        logger.debug(f"Reset environment {self.__class__.__name__} with seed {seed}")
        super().reset(seed=seed)

        # Zero out running counts by default. _begin_episode() may overwrite
        # via a StateSnapshot (Historic loads recorded visits; Online keeps
        # whatever the snapshot from telemetry contains).
        self._survey_progress_tracker.zero_counts()
        # Zero out running counts AND last-visit timestamps by default.
        # _begin_episode() may overwrite via a StateSnapshot (Historic loads
        # recorded visits; Online keeps whatever the snapshot from telemetry
        # contains; Offline optionally seeds night 0 from initial_* kwargs).
        self._survey_progress_tracker.zero_counts()
        self._last_visit_ot.fill(np.nan)

        # Subclass-defined episode start: load night 0, sync telemetry, etc.
        self._begin_episode()
        self._is_new_night = True

        self._update_action_masks()
        self._global_state = self._calculate_global_features()
        if self.include_candidate_features:
            self._candidate_state = self._calculate_candidate_features()

        return self.get_obs(), self.get_info()

    def step(self, action: dict):
        if self.field_level:
            action = self._command_to_action(action)
        else:
            assert self.action_space.contains(action), f"Invalid action {action}"

        # Subclass-defined: advance time, update visit counters, possibly roll
        # into a new night (offline) or fast-forward on WAIT (online).
        self._advance_after_action(action)

        self._update_action_masks()
        self._global_state = self._calculate_global_features()
        if self.include_candidate_features:
            self._candidate_state = self._calculate_candidate_features()

        reward = self._get_rewards()
        terminated = self._episode_terminated()
        truncated = False

        return self.get_obs(), reward, terminated, truncated, self.get_info()

    # -----------------------------------------------------------------------
    # Abstract lifecycle hooks
    # -----------------------------------------------------------------------

    @abstractmethod
    def _begin_episode(self) -> None:
        """Set up state for the start of an episode.

        Offline subclasses load night 0; the live subclass syncs telemetry.
        Implementations must populate `_ts`, `_sunset_ts`, `_sunrise_ts`,
        `_night_end_ts`, and the pointing trio (`_field_id`, `_bin_num`,
        `_filter_idx`) — typically by building a `StateSnapshot` and calling
        `_apply_state_snapshot`.
        """

    @abstractmethod
    def _advance_after_action(self, action: dict) -> None:
        """Mutate `_ts`, pointing, and visit counters from an action.

        Offline subclasses simulate by adding `exptime + slew_time`. The
        live subclass reads the wall clock or telemetry, and may also
        handle `WAIT_SIGNAL` fast-forwarding.
        """
        pass

    @abstractmethod
    def _episode_terminated(self) -> bool:
        """Whether the current episode has ended."""
        pass

    # -----------------------------------------------------------------------
    # Optional feature-context hooks. Default to None; subclasses override
    # only the ones they actually populate. `_calculate_global_features`
    # adds the corresponding feature only when the hook returns non-None.
    # -----------------------------------------------------------------------

    def _get_t_survey(self) -> Optional[float]:
        """Time-since-survey-start in units of days (discrete), or None if not provided."""
        return None

    def _get_fwhm(
        self, timestamp: float, el: Optional[float] = None,
        filter_idx: Optional[int] = None,
    ) -> Optional[float]:
        """Zenith reference-band seeing FWHM (arcsec), or None.

        Delegates to the env's SeeingModel, which returns the i-band zenith
        prediction independent of the current pointing. Returns None when no
        model is set. `el`/`filter_idx` are accepted for call-site
        compatibility but no longer used.
        """
        if self._seeing_model is None:
            return None
        return self._seeing_model.fwhm(timestamp=timestamp)

    def _get_raw_survey_progress(self) -> Optional[np.ndarray]:
        """Per-filter survey-wide visit counts (mutable), or None."""
        return None

    def _get_survey_nights_total(self) -> Optional[int]:
        """Total scheduled survey nights."""
        return None

    def _get_survey_night_idx(self) -> Optional[int]:
        """Current night index within the wider survey, if known."""
        return None

    # -----------------------------------------------------------------------
    # Shared primitive — used by both telemetry sync and per-night init
    # -----------------------------------------------------------------------

    def _apply_state_snapshot(self, snap: StateSnapshot) -> None:
        """Mutate internal state from a `StateSnapshot`.

        Counters are only overwritten when the snapshot provides them, so
        e.g. `OnlineBlancoEnv.sync_telemetry` can update pointing without
        clobbering the running visit history.
        """
        self._ts = snap.timestamp
        if snap.field_id == ZENITH_FIELD_ID:
            self._zenith_ts = snap.timestamp
        self._field_id = snap.field_id
        self._bin_num = snap.bin_num
        self._filter_idx = snap.filter_idx

        if snap.counts_cur is not None:
            tracker_shape = self._survey_progress_tracker.raw_counts.shape
            assert snap.counts_cur.shape == tracker_shape, (
                f"snapshot counts_cur shape {snap.counts_cur.shape} does not "
                f"match tracker shape {tracker_shape}"
            )
            self._survey_progress_tracker.set_counts(snap.counts_cur)
        if snap.last_visit_ot_cur is not None:
            expected_shape = self._last_visit_ot.shape
            assert snap.last_visit_ot_cur.shape == expected_shape, (
                f"snapshot last_visit_ts_cur shape {snap.last_visit_ot_cur.shape} "
                f"does not match {expected_shape}"
            )
            self._last_visit_ot[:] = snap.last_visit_ot_cur

    def _record_visit(self, field_id: int, filter_idx: int = None) -> None:
        """Bookkeeping after a successful observation.

        Keeps `_s_visits_cur`, `_s_filter_visits_cur`, and the survey-progress
        tracker in sync. Called by `_advance_after_action` in each
        concrete subclass for every non-wait action.
        """
        self._survey_progress_tracker.increment(
            field_id=field_id,
            filter_idx=filter_idx if self.do_filt else None,
        )
        # Field runs time a visit at exposure start, as the offline features do.
        visit_ts = self._step_record['t_start'] if self.field_level else self._ts
        ot_now = float(self._ot_at_sunset + (visit_ts - self._sunset_ts))
        if self.do_filt:
            self._last_visit_ot[field_id, filter_idx] = ot_now
        else:
            self._last_visit_ot[field_id] = ot_now

    # -----------------------------------------------------------------------
    # Concrete helpers for setting action mask constraints
    # -----------------------------------------------------------------------

    def set_constraints(
        self,
        *,
        airmass_limit: Optional[float] = None,
        sun_el_limit: Optional[float] = None,
    ) -> dict:
        """Update mask-gating constraints, refresh the action mask,
        and return a fresh info dict.

        Use before running a chunk of exposures so the agent's first
        choice in the chunk reads the correctly-gated mask:

            info = env.set_constraints(airmass_limit=2.5, sun_el_limit=-15.0)
            for _ in range(chunk_size):
                action = agent.choose_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)

        When sun_el currently exceeds sun_el_limit, the returned mask
        is all-False, forcing the agent to pick WAIT_SIGNAL.

        Note: in offline envs, twilight boundaries (`_sunset_ts`,
        `_sunrise_ts`) are computed once at night start. Calling this
        mid-night updates the mask but not the rollover schedule;
        the next night will pick up the new limit via
        `_get_night_config`.
        """
        if airmass_limit is not None:
            self.airmass_limit = airmass_limit
        if sun_el_limit is not None:
            self.sun_el_limit = sun_el_limit
        self._update_action_masks()
        return self.get_info()

    def compute_action_mask(self) -> np.ndarray:
        """Recompute action mask under current constraints. Use after set_constraints."""
        return self._update_action_masks()
    # -----------------------------------------------------------------------
    # Concrete helpers for all chilcdren
    # -----------------------------------------------------------------------

    def get_obs(self) -> dict:
        """Normalizes and returns the current state"""
        global_state = np.array(self._global_state, dtype=np.float32)
        global_state_normed, glob_nan_mask = self.global_normalizer.transform(
            global_state, **self._norm_stats.normalizer_kwargs('global_features')
        )
        if self.include_candidate_features:
            candidate_state_arr = np.array(self._candidate_state, dtype=np.float32)
            group = 'field_features' if self.field_level else 'bin_features'
            candidate_state_normed, candidate_nan_mask = self.candidate_normalizer.transform(
                candidate_state_arr, **self._norm_stats.normalizer_kwargs(group)
            )
        else:
            candidate_state_normed = np.array([], dtype=np.float32)
            candidate_nan_mask = None

        self._last_glob_nan_mask = glob_nan_mask
        self._last_candidate_nan_mask = candidate_nan_mask

        return {"global_state": global_state_normed, "candidate_state": candidate_state_normed}

    def get_info(self) -> dict:
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        info_dict = {
            'action_mask': self._action_mask.copy(),
            # 's_visited': self._s_visits_cur.copy(),
            # 'n_visited': self._n_visits_cur.copy(),
            'survey_progress_tracker': self._survey_progress_tracker.copy(),
            'valid_fields_per_bin': dict(self._valid_fields_per_bin) if self._valid_fields_per_bin is not None else {},
            'visible_bin_mask': self._visible_bin_mask.copy() if self._visible_bin_mask is not None else None,
            'timestamp': self._ts,
            'is_new_night': bool(self._is_new_night),
            'night_idx': int(self._night_idx),
            'bin': int(self._bin_num),
            'field_id': int(self._field_id),
            'glob_nan_mask': self._last_glob_nan_mask,
            'candidate_nan_mask': self._last_candidate_nan_mask,
        }
        return info_dict




    def _slew_distance(self, last_fid: int, current_fid: int) -> float:
        """On-sky distance between two fields; the zenith start uses the zenith at self._ts.

        Parameters
        ----------
        last_fid : int
            Previous field id, or ZENITH_FIELD_ID at the start of a night.
        current_fid : int
            Next field id.

        Returns
        -------
        float
            Angular separation in radians.
        """
        if last_fid == ZENITH_FIELD_ID:
            # Field runs slew from the zenith where the telescope parked (night start), as the dataset does.
            zenith_ts = self._zenith_ts if self.field_level and self._zenith_ts is not None else self._ts
            blanco = ephemerides.blanco_observer(time=float(zenith_ts))
            last_pos = np.array(blanco.radec_of('0', '90'))
        else:
            last_pos = self._ra_arr[last_fid], self._dec_arr[last_fid]

        current_pos = self._ra_arr[current_fid], self._dec_arr[current_fid]
        return math.geometry.angular_separation(last_pos, current_pos)

    def _get_dead_time(self, distance: float, filter_change: bool) -> float:
        """Time between exposures from the telescope profile: max(visit_overhead, slew_time(d)).

        Parameters
        ----------
        distance : float
            On-sky slew distance in radians.
        filter_change : bool
            Whether the filter changes between the two exposures.

        Returns
        -------
        float
            Dead time in seconds.
        """
        return float(self._telescope.parameters.dead_time(distance / units.deg, filter_change))

    def _get_exposure_time(self, field_id=None, filter_idx=None):
        """Per-(field, filter) exposure time from the lookups matrix.

        Returns 0.0 for negative `field_id` (sentinel — no real
        observation, so no time consumed) and the 90.0 s default when
        `filter_idx` is None (e.g. when an offline env advances the
        clock by an exposure-tick on a WAIT action without specifying
        a filter).
        """
        if field_id is None or int(field_id) < 0:
            return 0.0
        if filter_idx is None:
            return 90.0
        return float(
            self.lookups.fidfilt_exptime[int(field_id), int(filter_idx)]
        )

    def _uses_reward_term(self, term: RewardTerm) -> bool:
        """Whether the configured reward includes a term."""
        return self._reward_cfg is not None and term in self._reward_cfg.terms

    def _uses_teff_reward(self) -> bool:
        """Whether a reward term reads the step's predicted teff."""
        return self._uses_reward_term(RewardTerm.TEFF) or self._uses_reward_term(RewardTerm.TEFF_ACCEPTED)

    def _needs_seeing_model(self) -> bool:
        """Whether the fwhm feature or a teff-based reward term (via predicted teff) needs a seeing model."""
        return "fwhm" in self.global_feature_names or self._uses_teff_reward()

    def _cache_step_reward_inputs(self, s: dict) -> None:
        """Store a step's exposure-time reward inputs on its record, before the step can roll into the next night.

        Parameters
        ----------
        s : dict
            Step record with field_id, filter_idx, and t_start; gains 'uniformity' and, when a
            teff-based term is used, 'teff_pred'.
        """
        if self._uses_teff_reward():
            self._teff_pred(s)
        s['uniformity'] = self._uniformity_step_inputs(s)

    def _teff_pred(self, s: dict) -> float:
        """Predicted teff of a step."""
        if 'teff_pred' not in s:
            s['teff_pred'] = self._predict_teff(s)
        return s['teff_pred']

    def _min_teff(self, s: dict) -> float:
        """Minimum accepted teff for a step's filter under the run's acceptance rule."""
        return float(self._band_threshold[int(s['filter_idx'])])

    def _uniformity_step_inputs(self, s: dict) -> dict | None:
        """Uniformity reward inputs from survey counts before this exposure; None when the term is unused.

        Parameters
        ----------
        s : dict
            Step record with field_id, filter_idx, and t_start.

        Returns
        -------
        dict or None
            Keyword arguments for uniformity_reward, or None.
        """
        if not self._uses_reward_term(RewardTerm.UNIFORMITY):
            return None
        tracker = self._survey_progress_tracker
        return uniformity_inputs(tracker.raw_counts, tracker.target_counts, s['field_id'], s['filter_idx'])

    def _get_rewards(self) -> float:
        """Reward for the last step, from the dataset's reward terms and the training reward stats.

        Returns
        -------
        float
            Normalized reward; 0.0 without a reward config or when the step took no exposure (WAIT).
        """
        if self._reward_cfg is None or self._step_record is None:
            return 0.0
        R_tot = combine_rewards(self._reward_cfg, self._reward_term_inputs())
        return float(normalize_rewards(self._reward_cfg, R_tot, self._norm_stats.reward)[0])

    def _reward_term_inputs(self) -> dict:
        """Per reward term, a callable returning its keyword arguments for the last step."""
        s = self._step_record
        params = self._telescope.parameters
        return {
            RewardTerm.EXPERT: lambda: dict(n_transitions=1),
            RewardTerm.TEFF: lambda: dict(teff=np.array([self._teff_pred(s)])),
            RewardTerm.TEFF_ACCEPTED: lambda: dict(
                teff=np.array([self._teff_pred(s)]), min_teff=np.array([self._min_teff(s)]),
                exptime=np.array([self._get_exposure_time(s['field_id'], s['filter_idx'])])),
            RewardTerm.SLEW: lambda: dict(excess_times=np.array(
                [s['dead_time'] - params.visit_overhead(s['filter_change'])])),
            RewardTerm.UNIFORMITY: lambda: s['uniformity'],
        }

    def _predict_teff(self, s: dict) -> float:
        """Model teff at exposure start, with the env's seeing model for the delivered FWHM.

        Parameters
        ----------
        s : dict
            Step record with field_id, filter_idx, and t_start (Unix seconds).

        Returns
        -------
        float
            Predicted effective exposure time factor tau.
        """
        if self._seeing_model is None:
            raise ValueError("teff reward requires a seeing model in the environment.")
        band = self.idx2filter[s['filter_idx']]
        ra, dec = self._ra_arr[s['field_id']], self._dec_arr[s['field_id']]
        _, el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=s['t_start'])
        fwhm = self._seeing_model.fwhm(s['t_start'], band=band, el=el) * units.arcsec
        return float(predict_teff(time=s['t_start'], ra=ra, dec=dec, band=band, el=el, fwhm=fwhm))

    def _calculate_global_features(self) -> list:
        """Compute the global feature vector for the current state.

        Thin orchestration over the shared helpers in
        ``blancops.data.features.glob_features``:

          1. Time-only ephemeris via ``compute_global_time_only_features``
             (LST, Sun/Moon positions and phase). LST is needed before
             resolving the pointing in the zenith branch, so this runs first.
          2. Resolve pointing RA/Dec (with the zenith / WAIT / real-field
             fork from the original env).
          3. Pointing-derived ephemeris via
             ``compute_global_pointing_features`` (az/el/ha/airmass and
             per-filter sky brightness).
          4. Filter features (filter_wave / filter_idx / one-hot) — kept
             inline because the live env's branchy zenith-vs-WAIT-vs-real
             logic doesn't match the offline pipeline's df.map pattern.
          5. Hook-derived features (fwhm, t_survey).
          6. Tracker-derived per-filter features (survey_progress, urgency).
          7. ``t_night`` from the env's cached sunrise/sunset boundaries.
          8. Cyclical norms via ``apply_cyclical_global_features``.
          9. Stack into a list in ``self.global_feature_names`` order;
             raise on any NaN.

        With ``_validate_feature_config`` having run at construction,
        hooks/tracker/flags are guaranteed available for every feature in
        ``self.global_feature_names``.
        """
        timestamp = self._state_timestamp()

        # 1. Time-only ephemeris (gives us LST so we can resolve zenith pointing).
        new_features = compute_global_time_only_features(timestamp=timestamp, telescope=self._telescope)

        # 2. Resolve pointing RA/Dec. Preserves the original env's zenith branch
        #    (`lst, blanco.lon`) — see the migration notes; this disagrees with
        #    `blanco.radec_of('0','90')` used elsewhere and is worth a separate
        #    look, but isn't changed in this refactor.
        if self._field_id == ZENITH_FIELD_ID:
            blanco = ephemerides.blanco_observer(time=timestamp)
            ra, dec = new_features['lst'], blanco.lat
        elif self._bin_num == WAIT_SIGNAL:
            ra, dec = ephemerides.topographic_to_equatorial(
                self._last_az, self._last_el, time=timestamp
            )
        else:
            ra = self._ra_arr[self._field_id]
            dec = self._dec_arr[self._field_id]
        new_features['ra'] = ra
        new_features['dec'] = dec

        # 3. Pointing-derived ephemeris (az/el/ha/airmass + sky brightness).
        new_features.update(
            compute_global_pointing_features(timestamp=timestamp, ra=ra, dec=dec, moon_radec=(new_features['moon_ra'], new_features['moon_dec']),
                                             survey=self._survey)
        )

        # 4. Filter features. The live env has three cases (zenith / WAIT or
        #    no-filter-action / real-field-with-filter), all subtly different
        #    from each other and from the offline df.map pipeline — kept
        #    inline rather than abstracted.
        if self._field_id == ZENITH_FIELD_ID:
            new_features['filter_wave'] = 0.
            new_features['filter_idx'] = ZENITH_FILTER_IDX
            for filt in self._survey.filters:
                new_features[f'is_filter_{filt}'] = 0
        else:
            if self._bin_num == WAIT_SIGNAL or (not self.do_filt):
                new_features['filter_wave'] = 0
                new_features['filter_idx'] = self._filter_idx
            else:
                wave_norm = self._telescope.parameters.filter_wave_norm
                new_features['filter_wave'] = self._survey.idx2wave[self._filter_idx] / wave_norm
                new_features['filter_idx'] = self._filter_idx
            filt_str = self.idx2filter[self._filter_idx]
            for filt in self._survey.filters:
                new_features[f'is_filter_{filt}'] = filt_str == filt

        # 5. Hook-derived features — only populated if their source is available.
        #    el/filter_idx are passed so forward-sim envs can rescale a
        #    reference seeing to the pointing being evaluated.
        fwhm = self._get_fwhm(timestamp, el=new_features['el'], filter_idx=self._filter_idx)
        if fwhm is not None:
            new_features['fwhm'] = fwhm
        t_survey = self._get_t_survey()
        if t_survey is not None:
            new_features["t_survey"] = t_survey

        # 6. Tracker-derived features. The dispatcher in glob_features.py decides
        # which families to compute based on what's in self.global_feature_names;
        # we just merge the result. Adding a new per-filter family does not
        # require any change here.
        tracker = self._survey_progress_tracker
        ctx = {
            "tracker": tracker,
            "idx2filter": self.idx2filter,
            "survey": self._survey,
            # Eagerly resolve the hooks the urgency family needs. Cheap, and
            # keeps `compute_global_tracker_features` free of env coupling.
            # These will be None for envs that don't override the hooks, but
            # the urgency family won't be invoked unless an urgency_* feature
            # is requested, and _validate_feature_config has already enforced
            # that those envs DO override the hooks.
            "survey_night_idx": (
                self._get_survey_night_idx() if tracker._is_field_filter else None
            ),
            "survey_nights_total": (
                self._get_survey_nights_total() if tracker._is_field_filter else None
            ),
        }
        new_features.update(
            compute_global_tracker_features(
                requested_names=self.global_feature_names,
                tracker=tracker,
                ctx=ctx,
            )
        )


        # tracker = self._survey_progress_tracker
        # if tracker._is_field_filter:
        #     survey_night_idx = self._get_survey_night_idx()
        #     survey_nights_total = self._get_survey_nights_total()
        #     for filt, idx in self._survey.filter2idx.items():
        #         p_name = f"survey_progress_{filt}"
        #         u_name = f"urgency_{filt}"
        #         if p_name in self.global_feature_names:
        #             new_features[p_name] = tracker.get_filter_progress(idx)
        #         if u_name in self.global_feature_names:
        #             visits = int(tracker.raw_counts[:, idx].sum())
        #             target = int(tracker.target_counts[:, idx].sum())
        #             if target == 0:
        #                 new_features[u_name] = 0
        #             else:
        #                 new_features[u_name] = calc_urgency(
        #                     filter_counts_arr=visits,
        #                     filter_counts_max=target,
        #                     survey_night_indices=survey_night_idx,
        #                     survey_nights_max=survey_nights_total,
        #                 )
                    # print(u_name, new_features[u_name])

        # 7. t_night from cached night boundaries.
        if self._sunrise_ts <= self._sunset_ts:
            raise AssertionError(
                "Sunrise time is not after sunset time. Check night_str argument - "
                "it should be a time between sunset and sunrise"
            )
        new_features['t_night'] = normalize_timestamp(
            timestamp,
            sunset_timestamp=self._sunset_ts,
            sunrise_timestamp=self._sunrise_ts,
        )
        # 8. Cyclical norms via the shared helper.
        if self.global_normalizer.do_cyclical_norm:
            apply_cyclical_features(
                new_features,
                self.base_global_feature_names,
                self.global_normalizer.cyclical_feature_names,
            )

        # 9. Stack in requested order; surface any missing/NaN features loudly.
        global_state_features = [
            new_features.get(feat, np.nan) for feat in self.global_feature_names
        ]

        nan_feats = np.isnan(global_state_features)
        if any(nan_feats):
            nan_idx = int(np.where(nan_feats)[0][0])
            raise ValueError(
                f"Calculated nan value for global feature "
                f"{self.global_feature_names[nan_idx]}"
            )

        return global_state_features

    # -----------------------------------------------------------------------
    # Candidate features — field features (field_features.py) or bin
    # features (bin_features.py), depending on the action space.
    # -----------------------------------------------------------------------

    def _state_timestamp(self) -> float:
        """Time the state describes: the start of the latest exposure on the field path (as the dataset rows
        are timestamped, and as DES queues the next exposure during the current one), else the clock.

        Returns
        -------
        float
            Unix timestamp.
        """
        if self.field_level and self._step_record is not None and not self._is_new_night:
            return float(self._step_record['t_start'])
        return self._ts

    def _current_pointing_radec(self) -> np.ndarray:
        """Current pointing (RA, Dec) in radians; the zenith for WAIT and sentinel pointings.

        Returns
        -------
        np.ndarray
            (2,) RA and Dec.
        """
        if self._bin_num == WAIT_SIGNAL or self._field_id == ZENITH_FIELD_ID:
            blanco = ephemerides.blanco_observer(time=self._ts)
            return np.array(blanco.radec_of('0', '90'))
        return np.array([self._ra_arr[self._field_id], self._dec_arr[self._field_id]])

    def _field_features_now(self) -> dict:
        """Raw field features at the current state, computed once per state and reused by the mask.

        Returns
        -------
        dict
            Output of `compute_field_features` for every field.
        """
        tracker = self._survey_progress_tracker
        ts = self._state_timestamp()
        key = (ts, self._field_id, int(tracker.raw_counts.sum()))
        if self._field_feature_cache is None or self._field_feature_cache[0] != key:
            feats = compute_field_features(
                timestamp=ts, pointing_radec=self._current_pointing_radec(), field_grid=self._field_grid,
                night_duration_sec=self._sunrise_ts - self._sunset_ts, counts=tracker.raw_counts,
                targets=tracker.target_counts, last_visit_ot=self._last_visit_ot,
                ot_now=float(self._ot_at_sunset + (ts - self._sunset_ts)), survey=self._survey,
            )
            self._field_feature_cache = (key, feats)
        return self._field_feature_cache[1]

    def _update_field_action_mask(self) -> np.ndarray:
        """Field-level mask: visible (airmass, mount envelope, from the field features) and in plan and incomplete.

        Returns
        -------
        np.ndarray
            Flat boolean mask over n_fields * n_filters actions; all False while the Sun is up.
        """
        self._valid_fields_per_bin = defaultdict(list)
        self._visible_bin_mask = None
        ts = self._state_timestamp()
        sun_radec = ephemerides.get_source_ra_dec('sun', time=ts)
        _, sun_el = ephemerides.equatorial_to_topographic(sun_radec[0], sun_radec[1], time=ts)
        if sun_el / units.deg > self.sun_el_limit:
            self._action_mask = np.zeros(self.nfields * self.nfilters, dtype=bool)
            return self._action_mask
        feats = self._field_features_now()
        visible = self._telescope.visible(
            feats['el'], feats['ha'], self._dec_arr, min(self.airmass_limit, self.airmass_failsafe)
        )
        sel_valid = self._survey_progress_tracker.get_incomplete_mask() & visible[:, np.newaxis]
        self._action_mask = self._apply_field_mask(sel_valid).flatten()
        return self._action_mask

    def _calculate_candidate_features(self) -> np.ndarray:
        """Compute the candidate feature tensor for the current state.

        Returns
        -------
        np.ndarray
            (n_candidates, n_feats) in the order of `self.candidate_feature_names`; candidates are survey
            fields for field_filter, else HEALPix bins.
        """
        if self.field_level:
            return stack_field_features(self._field_features_now(), self.candidate_feature_names)
        return self._calculate_bin_features()

    def _calculate_bin_features(self) -> np.ndarray:
        """Compute the HEALPix bin feature tensor for the current state.

        Thin orchestration over the shared helpers in
        `blancops.data.features.bin_features`:

          1. Resolve the current pointing in RA/Dec (handling the WAIT/zenith
             fallback).
          2. Compute ephemeris features via `compute_candidate_ephemeris_features`.
          3. If any history features are configured, compute the field->bin
             assignment + visibility mask, then call
             `compute_bin_history_features` (NaN sentinels for inactive bins).
          4. Apply rel / cyclical norms over the per-timestep dict.
          5. Validate history features (NaN-aware).
          6. Convert internal NaN sentinels to the external value (-1.0).
          7. Stack into `(nbins, nfeats)` in the order of
             `self.candidate_feature_names`.
        """
        timestamp = self._ts
        tracker = self._survey_progress_tracker

        # 1. Pointing in RA/Dec, with zenith fallback for WAIT / sentinel pointings.
        pointing_radec = self._current_pointing_radec()

        # 2. Ephemeris features (always produced for every key).
        features = compute_candidate_ephemeris_features(
            timestamp=timestamp,
            pointing_radec=pointing_radec,
            grid=self.hpGrid,
            night_duration_in_sec=self._sunrise_ts - self._sunset_ts
        )

        # 3. History features (only when configured).
        if self._has_historical_features:
            bins_per_field, v_mask = self._compute_bin_assignments(timestamp)
            current_counts, target_counts = self._tracker_counts_for_history(tracker)
            ot_now = self._ot_at_sunset + (self._ts - self._sunset_ts)
            # print("dtype:", self._last_visit_ot.dtype,
            #         "nan count:", np.isnan(self._last_visit_ot).sum(),
            #         "sentinel count:", (self._last_visit_ot < -1e17).sum())
            # print(self.lookups.total_ot_sec, self._last_visit_ot, self._sunrise_ts - self._sunset_ts)
            features.update(
                compute_bin_progress_features(
                    current_counts=current_counts,
                    target_counts=target_counts,
                    bins_per_field=bins_per_field,
                    v_mask=v_mask,
                    nbins=self.nbins,
                    do_filt=self.do_filt,
                    idx2filter=self.idx2filter,
                    timestamp=ot_now,
                    last_visit_timestamps=self._last_visit_ot,
                    t_since_last_visit_divisor=None, #self.lookups.total_ot_sec
                )
            )
        # for key in self.candidate_feature_names:
        #     print(key)
        # for key in features.keys():
        #     print(key)

        # 4. Relative + cyclical features (in place on the dict).
        el_mask = features['el'] > 0
        apply_relative_bin_features(
            features, el_mask, self._has_historical_features, self.do_filt, self._survey
        )
        if self.candidate_normalizer.do_cyclical_norm:
            apply_cyclical_features(
                features,
                self.base_bin_feature_names,
                self.candidate_normalizer.cyclical_feature_names,
            )
        # 5. Validate (NaN-aware). Only meaningful when history features exist.
        if self._has_historical_features:
            validate_history_bin_features(
                features, self.do_filt, self.idx2filter
            )

        # 6. Convert internal NaN sentinels (inactive bins) to the external value.
        # Is now done in state normalizer
        # replace_invalid_with_sentinel(features, self._bin_feat_sentinel)

        # 7. Stack in requested order. After step 6 there should be no NaNs;
        #    a remaining NaN means a missing feature implementation or a leak
        #    from a non-history feature, which we surface loudly.
        final_arrays = []
        for key in self.candidate_feature_names:
            if key not in features:
                raise ValueError(
                    f"Requested feature '{key}' was not calculated by the pipeline."
                )
            arr = features.pop(key)
            # assert not np.isnan(arr).any(), (
            #     f"NaN values found in feature '{key}' after sentinel "
            #     f"replacement: {arr}"
            # )
            final_arrays.append(arr)
        assert len(final_arrays) == len(self.candidate_feature_names), (
            "Number of final arrays should match number of requested bin features"
        )

        bin_states = np.array(final_arrays)
        return rearrange(bin_states, 'nfeats nbins -> nbins nfeats')

    def _compute_bin_assignments(self, timestamp):
        """Return `(bins_per_field, v_mask)` appropriate for the coord system.

        RA/Dec: bin assignment is static across time; cached on `self` after
        the first call. The visibility mask is "has a valid (non-zenith)
        bin" — RA/Dec features aren't horizon-gated at the bin level since
        the field-to-bin mapping doesn't change with the sky's rotation.

        AzEl: recomputed on every call because the field-to-bin mapping
        changes as the sky rotates. The visibility mask additionally drops
        fields below the horizon.
        """
        if not self.hpGrid.is_azel:
            if self._field_bins_radec is None:
                bins_raw = self.hpGrid.ang2idx(
                    lon=self._ra_arr, lat=self._dec_arr
                )
                self._field_bins_radec = np.array(
                    [b if b is not None else ZENITH_BIN_NUM for b in bins_raw],
                    dtype=np.int32,
                )
                self._field_bins_radec_v_mask = (
                    self._field_bins_radec != ZENITH_BIN_NUM
                )
            return self._field_bins_radec, self._field_bins_radec_v_mask

        # AzEl — recompute every step.
        az, el = ephemerides.equatorial_to_topographic(
            ra=self._ra_arr, dec=self._dec_arr, time=timestamp
        )
        bins_raw = self.hpGrid.ang2idx(lon=az, lat=el)
        bins = np.array(
            [b if b is not None else ZENITH_BIN_NUM for b in bins_raw],
            dtype=np.int32,
        )
        v_mask = (el > 0) & (bins != ZENITH_BIN_NUM)
        return bins, v_mask

    @staticmethod
    def _tracker_counts_for_history(tracker):
        """Pull `(current_counts, target_counts)` out of a survey tracker
        in the shape expected by `compute_bin_progress_features`.

        When the tracker is per-(field, filter), returns the 2D arrays
        directly. When it's per-field only, returns 1D arrays. The shared
        helper key off shape rather than a flag.
        """
        if tracker._is_field_filter:
            return tracker.raw_counts, tracker.target_counts
        return tracker.raw_counts, tracker.target_field_counts

    # -----------------------------------------------------------------------
    # Action / observation spaces and masks
    # -----------------------------------------------------------------------

    def _command_to_action(self, command: dict) -> dict:
        """Validate a physical command and translate it to the internal (bin, field_id, filter_idx) action.

        The pointing must equal a candidate field's (RA, Dec) exactly and, outside replay mode, be allowed by
        the current mask; otherwise the command is rejected before any state changes.

        Parameters
        ----------
        command : dict
            {'pointing': (RA, Dec) float64 radians, 'filter': int, 'wait': 0 or 1}.

        Returns
        -------
        dict
            Internal action; 'bin' is WAIT_SIGNAL for a wait, else the field's grid bin (bookkeeping only).
        """
        assert self.action_space.contains(command), f"Invalid command {command}"
        if int(command['wait']):
            return {'bin': np.int32(WAIT_SIGNAL), 'field_id': np.int32(self._field_id),
                    'filter_idx': np.int32(self._filter_idx)}
        ra, dec = (float(x) for x in command['pointing'])
        field_id = self._coord_to_field.get((ra, dec))
        if field_id is None:
            raise ValueError(f"Commanded pointing ({ra}, {dec}) matches no candidate field.")
        filter_idx = int(command['filter'])
        if not self._replay_mode and not self._action_mask[field_id * self.nfilters + filter_idx]:
            raise ValueError(f"Field {field_id} in filter {filter_idx} is not observable now.")
        if self.hpGrid.is_azel:
            lon, lat = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=self._ts)
        else:
            lon, lat = ra, dec
        grid_bin = self.hpGrid.ang2idx(lon=lon, lat=lat)
        return {'bin': np.int32(grid_bin if grid_bin is not None else 0), 'field_id': np.int32(field_id),
                'filter_idx': np.int32(filter_idx)}

    def _setup_action_and_obs_spaces(self, state_dim, candidate_state_dim):
        if self.field_level:
            # Fixed spaces: candidates are a variable-length list in the observation, the action is a pointing.
            self.observation_space = gym.spaces.Dict({
                "global_state": gym.spaces.Box(-1e5, 1e5, shape=(state_dim,), dtype=np.float32),
                "candidate_state": gym.spaces.Sequence(
                    gym.spaces.Box(-1e5, 1e5, shape=(candidate_state_dim,), dtype=np.float32), stack=True),
            })
            self.action_space = gym.spaces.Dict({
                "pointing": gym.spaces.Box(np.array([0.0, -np.pi / 2]), np.array([2 * np.pi, np.pi / 2]),
                                           dtype=np.float64),
                "filter": gym.spaces.Discrete(self.nfilters),
                "wait": gym.spaces.Discrete(2),
            })
            return
        if self.include_candidate_features:
            bin_state_shape = (self.nbins, candidate_state_dim)
        else:
            bin_state_shape = (0,)

        self.observation_space = gym.spaces.Dict({
            "global_state": gym.spaces.Box(-1e5, 1e5, shape=(state_dim,), dtype=np.float32),
            "candidate_state": gym.spaces.Box(-1e5, 1e5, shape=bin_state_shape, dtype=np.float32),
        })

        smallest_sentinel = min([WAIT_SIGNAL, ZENITH_BIN_NUM])
        self.action_space = gym.spaces.Dict({
            "bin": gym.spaces.Discrete(
                self.nbins - smallest_sentinel,
                start=min([WAIT_SIGNAL, ZENITH_BIN_NUM]),
            ),
            "field_id": gym.spaces.Discrete(
                len(self._fids) - smallest_sentinel,
                start=min([WAIT_SIGNAL, ZENITH_FIELD_ID]),
            ),
            "filter_idx": gym.spaces.Discrete(
                self.nfilters - smallest_sentinel,
                start=min([WAIT_SIGNAL, ZENITH_FILTER_IDX]),
            ),
        })

    def _apply_field_mask(self, sel_valid: np.ndarray) -> np.ndarray:
        """Optional hook: drop operator-masked fields from the validity mask.

        Default is identity (no masking). LiveBlancoEnv overrides this to zero
        the rows of `sel_valid` for fields the operator has masked. Called by
        `_update_action_masks` after visibility gating so that both the action
        mask and `_valid_fields_per_bin` reflect the masking consistently.
        """
        return sel_valid

    def _update_action_masks(self):
        """Construct the action mask based on airmass / horizon / completion."""
        if self.field_level:
            return self._update_field_action_mask()
        sun_radec = ephemerides.get_source_ra_dec('sun', time=self._ts)
        _, sun_el = ephemerides.equatorial_to_topographic(sun_radec[0], sun_radec[1], time=self._ts)

        if sun_el / units.deg > self.sun_el_limit:
            logger.warning(f"Sun ({sun_el / units.deg}) is above the horizon ({self.sun_el_limit}), no actions will be available.")

            if self.do_filt:
                self._action_mask = np.zeros(
                    shape=(self.nbins * self.nfilters,), dtype=bool
                )
            else:
                self._action_mask = np.zeros(shape=self.nbins, dtype=bool)
            self._valid_fields_per_bin = defaultdict(list)
            self._visible_bin_mask = np.zeros(shape=self.nbins, dtype=bool)
            return self._action_mask

        fields_az, fields_el = ephemerides.equatorial_to_topographic(
            ra=self._ra_arr, dec=self._dec_arr, time=self._ts
        )
        # Airmass limit and the equatorial mount's HA/Dec envelope.
        effective_airmass_limit = min(self.airmass_limit, self.airmass_failsafe)
        mask_visibility = self._telescope.visible(
            fields_el, astropy_ephem.hour_angle(self._ra_arr, self._ts, self._telescope.site.lon), self._dec_arr, effective_airmass_limit
        )

        sel_valid = self._survey_progress_tracker.get_incomplete_mask()
        if self.do_filt:
            sel_valid = sel_valid & mask_visibility[:, np.newaxis]
        else:
            sel_valid = sel_valid & mask_visibility
        sel_valid = self._apply_field_mask(sel_valid)
        if self.do_filt:
            sel_valid_fields = sel_valid.any(axis=1)
        else:
            sel_valid_fields = sel_valid

        if self.hpGrid.is_azel:
            valid_field_bins = self.hpGrid.ang2idx(
                lon=fields_az[sel_valid_fields], lat=fields_el[sel_valid_fields]
            )
        else:
            valid_field_bins = self.hpGrid.ang2idx(
                lon=self._ra_arr[sel_valid_fields], lat=self._dec_arr[sel_valid_fields]
            )
        valid_bin_mask = np.array(valid_field_bins) != None
        clean_bins = np.array(valid_field_bins)[valid_bin_mask].astype(int)

        if self.do_filt:
            action_mask = np.zeros(shape=(self.nbins, self.nfilters), dtype=bool)
            clean_ff = sel_valid[sel_valid_fields][valid_bin_mask]
            np.logical_or.at(action_mask, clean_bins, clean_ff)
            action_mask = action_mask.flatten()
        else:
            action_mask = np.zeros(shape=self.nbins, dtype=bool)
            action_mask[clean_bins] = True

        valid_fids = self._fids[sel_valid_fields]
        clean_fids = valid_fids[valid_bin_mask]
        self._valid_fields_per_bin = defaultdict(list)
        for b, fid in zip(clean_bins, clean_fids):
            self._valid_fields_per_bin[b].append(fid)

        self._action_mask = action_mask
        self._visible_bin_mask = self._compute_visible_bin_mask()
        return action_mask

    def _compute_visible_bin_mask(self, timestamp: float | None = None) -> np.ndarray:
        """Boolean mask over HEALPix bins observable at a given timestamp.

        Evaluates the horizon, airmass, and equatorial-envelope constraints at
        each bin center rather than at catalog fields, so a bin counts as
        visible even when no field currently lives in it. This is the "sky as
        if fully populated" view the filter-first decoder uses to score filter
        preference in the dense regime the policy was trained on, decoupled
        from sparse ("island") field layouts. It is not a legality mask and
        does not gate `step`.

        Args
        ----
        timestamp: Unix time to evaluate the mask at. Defaults to the env's
            current timestamp. Pass an explicit value to compute the mask
            off-line (e.g. for one-step-ahead evaluation). Ignored for alt-az
            grids (time-invariant bins).

        Returns
        -------
        Boolean array of shape ``(nbins,)``; True where the bin center clears
        the horizon, airmass, and (for equatorial mounts) HA/Dec envelope.
        """
        ts = self._ts if timestamp is None else timestamp
        lon = self.hpGrid.lon  # [nbins] radians (az if azel grid, else ra)
        lat = self.hpGrid.lat  # [nbins] radians (el if azel grid, else dec)

        if self.hpGrid.is_azel:
            bin_el = lat
        else:
            _, bin_el = ephemerides.equatorial_to_topographic(
                ra=lon, dec=lat, time=ts
            )

        # Equatorial mount HA/Dec envelope is skipped for alt-az grids, where bin centers carry no RA.
        ha = None if self.hpGrid.is_azel else astropy_ephem.hour_angle(lon, ts, self._telescope.site.lon)
        effective_airmass_limit = min(self.airmass_limit, self.airmass_failsafe)
        return self._telescope.visible(bin_el, ha, lat, effective_airmass_limit)

    # -----------------------------------------------------------------------
    # Construction-time hook and feature validation
    # -----------------------------------------------------------------------

    def _validate_feature_config(self) -> None:
        """Fail fast when a requested feature has no source on this env.

        Concrete subclasses MUST call this as the last line of
        __init__. Three kinds of check, dispatched off
        `_feature_requirements(self._survey.filters)`:

          * ('hook', name) — type(self) overrides the named method
          * ('attr', name) — getattr(self, name) is not None
          * ('flag', name) — getattr(self, name) is truthy
        """
        cls = type(self)
        issues: list[str] = []

        requirements = _feature_requirements(self._survey.filters)
        for feat in self.global_feature_names:
            for kind, target in requirements.get(feat, []):
                if kind == "hook":
                    if not self._is_hook_overridden(target):
                        issues.append(
                            f"  - feature '{feat}' requires {cls.__name__} "
                            f"to override hook '{target}', but it is the "
                            f"default no-op."
                        )
                elif kind == "attr":
                    if getattr(self, target, None) is None:
                        issues.append(
                            f"  - feature '{feat}' requires self.{target} "
                            f"to be set, but it is None."
                        )
                elif kind == "flag":
                    if not getattr(self, target, False):
                        issues.append(
                            f"  - feature '{feat}' requires self.{target}=True "
                            f"(typically: filter in action space), but it is False."
                        )
                else:  # pragma: no cover
                    raise AssertionError(f"unknown requirement kind: {kind}")

        if issues:
            raise ValueError(
                f"Feature configuration mismatch in {cls.__name__}:\n"
                + "\n".join(issues)
                + f"\nConfigured global_features: {self.global_feature_names}"
            )

        # if self._has_historical_features and self.lookups.total_ot_sec is None:
        #     raise ValueError(
        #         f"{cls.__name__}: bin features include staleness/history terms "
        #         f"(found in candidate_feature_names: "
        #         f"{[b for b in self.candidate_feature_names if any(k in b for k in _STALENESS_BASE_KEYS)]}) "
        #         f"but lookups.total_ot_sec is None. This must be the same "
        #         f"normalization constant the policy was trained with — typically "
        #         f"loaded from the training data directory's total_ot_seconds file."
            # )
    def _is_hook_overridden(self, hook_name: str) -> bool:
        base_fn = BaseBlancoEnv.__dict__.get(hook_name)
        if base_fn is None:
            return True
        actual_fn = getattr(type(self), hook_name, None)
        return actual_fn is not base_fn
