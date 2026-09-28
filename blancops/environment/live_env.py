
"""Live operational environment driven by hardware telemetry.

Branches off `BaseBlancoEnv` directly rather than going through
`OfflineBlancoEnvBase` because the multi-night scaffolding doesn't
apply: an online episode is a single open-ended night driven by reality,
with time and pointing coming from the telescope rather than simulation.

`sync_telemetry` is the public, operator-facing way to align the env
with hardware state. It can be called any time (mid-step, between
actions, on recovery from a weather pause) and is idempotent. Internally
it uses the same `_apply_state_snapshot` primitive that offline envs use
to seed each new night, but with `counts_cur=None` so running visit
history is preserved across syncs — only `_advance_after_action` (via
`_record_visit`) accumulates counts in the live setting.
"""
from __future__ import annotations

from typing import Optional
import logging

import numpy as np

from blancops.environment.base import BaseBlancoEnv, StateSnapshot
from blancops.environment.field_mask_schedule import MaskRule, resolve_positional_mask
from blancops.environment.seeing_model import PredictiveSeeingModel
from blancops.environment.survey_tracker import SurveyProgressTracker
from blancops.data.features.glob_features import get_night_boundaries
from blancops.ephemerides import ephemerides
from blancops.configs.constants import (
    WAIT_SIGNAL, ZENITH_FILTER_IDX, FILTER2IDX, IDX2FILTER, FWHM_REF_FILTER,
)

logger = logging.getLogger(__name__)


class LiveBlancoEnv(BaseBlancoEnv):
    """Real-time environment for live telescope operation.

    Constructor is keyword-only. `observing_night` should be a
    timezone-aware datetime; only the date is consulted — twilights are
    computed from it.
    """

    def __init__(
        self,
        cfg,
        constraints_cfg,
        lookups,
        norm_stats,
        telemetry_init,
        survey_night_idx=0,
        telescope=None,
        seeing_window=None
    ):
        self._survey_night_idx = survey_night_idx

        self._priority_trigger = False
        self._priority1_positional = None   # [n_fields] bool
        self._priority_rule = None          # MaskRule(keep_only priority-1 ids)
        self._masked_field_ids = []         # list of operator-masked field ids

        super().__init__(
            cfg=cfg,
            constraints_cfg=constraints_cfg,
            lookups=lookups,
            norm_stats=norm_stats,
            telescope=telescope,
        )
        self._build_priority_mask()
        # airmass_limit and sun_el_limit are stored on self by base.
        # Live sessions always start fresh; offline envs load this from
        # historical snapshots via offline_base.py._apply_state_snapshot.
        self._ot_at_sunset = 0.0
        # Rolling seeing predictor, fed by real telemetry readings on each
        # sync. Built before the first sync below so telemetry_init can seed
        # it. Cold start falls back to the nominal median in Seeing.predict.
        if self._needs_seeing_model():
            if seeing_window:
                cfg.data.seeing.window = seeing_window
            self._seeing_model = PredictiveSeeingModel(cfg.data.seeing)
        self.sync_telemetry(telemetry=telemetry_init)
        self._validate_feature_config()
    # -----------------------------------------------------------------------
    # Public API: operator-facing telemetry sync
    # -----------------------------------------------------------------------

    def sync_telemetry(self, telemetry: Optional[dict] = None) -> None:
        """Align internal state with current hardware state.

        Idempotent. Safe to call mid-episode whenever the simulator and
        the telescope have drifted (manual override, weather pause,
        operator intervention). Visit counters are NOT reset — the
        running per-night history is preserved by omitting `counts_cur`
        from the snapshot.

        Args
        ----
        telemetry : Optional[dict]
            Expected keys: 'timestamp' (unix ts), 'ra' / 'dec' (matching the
            unit convention of `self._ra_arr` / `self._dec_arr`), and
            optionally 'filter_idx'. If None, calls
            `self._telemetry.read()`.
        """
        if telemetry is not None:
            filter_idx = telemetry.get(
                "filter_idx", FILTER2IDX.get(telemetry.get("filter"), ZENITH_FILTER_IDX)
            )
            snap = StateSnapshot(
                timestamp=telemetry["timestamp"],
                field_id=self._match_pointing_to_fid(ra=telemetry["ra"], dec=telemetry["dec"]),
                filter_idx=filter_idx,
                # counts_cur intentionally omitted — preserve running history.
            )
            self._apply_state_snapshot(snap)

            if telemetry.get("seeing") is not None and self._seeing_model is not None:
                self._seeing_model.replace(telemetry.get("seeing"))

        self._refresh_night_boundaries()
        self._recompute_derived_state()

    def record_visit(self, obs_row) -> None:
        """Record that an observation was submitted to the telescope.

        Called by the orchestrator immediately after each hardware submission so
        that subsequent rollouts reflect the in-progress observation.  Unlike
        ``compute_post_action_snapshot``, the visit OT is anchored to the
        current real timestamp (``self._ts``), not a simulated future time —
        this avoids the clock-skew that produced negative staleness values.

        Does not advance ``_ts``; timestamp is only updated via
        ``sync_telemetry`` from real hardware telemetry.
        """
        field_id = int(obs_row["field_id"])
        filter_idx = int(FILTER2IDX[obs_row["filter"]])
        # Field runs count the submission only if its predicted teff is valid (judged at submission time).
        self._step_record = dict(field_id=field_id, filter_idx=filter_idx, t_start=float(self._ts))
        self._record_visit(field_id=field_id, filter_idx=filter_idx)
        self._field_id = field_id
        self._filter_idx = filter_idx
        self._recompute_derived_state()

    def save_snapshot(self) -> StateSnapshot:
        """Capture the full mutable state — pointing, time, and visit counts."""
        return StateSnapshot(
            timestamp=self._ts,
            field_id=self._field_id,
            bin_num=self._bin_num,
            filter_idx=self._filter_idx,
            counts_cur=self._survey_progress_tracker.raw_counts.copy(),
            last_visit_ot_cur=self._last_visit_ot.copy(),
        )

    def restore_snapshot(self, snap: StateSnapshot) -> None:
        """Restore mutable state from a snapshot and recompute derived arrays."""
        self._apply_state_snapshot(snap)
        self._refresh_night_boundaries()
        self._recompute_derived_state()

    def resume_interrupted_session(self, completed_obs) -> None:
        """Rebuild visit counts and last-visit times from a completed-observation log.

        Replays this night's persisted observing history (one row per completed
        exposure) into the survey-progress tracker and last-visit-OT array so a
        session that was terminated mid-night resumes with correct cumulative
        state. Counts increment once per
        row; ``last_visit_ot`` is anchored to each row's own ``timestamp`` via
        ``ot_at_sunset + (timestamp - sunset_ts)``, keeping the latest value per
        field (and filter when ``do_filt``). Rows whose ``field_id`` falls
        outside the current catalog are skipped.

        Args
        ----
        completed_obs : pandas.DataFrame
            Completed observations with at least ``field_id``, ``filter``, and
            ``timestamp`` columns. An empty or None frame is a no-op.
        """
        if completed_obs is None or len(completed_obs) == 0:
            return

        self._refresh_night_boundaries()

        counts = np.zeros_like(self._survey_progress_tracker.raw_counts)
        last_visit_ot = np.full(
            self._last_visit_ot.shape, np.nan, dtype=self._last_visit_ot.dtype
        )

        for row in completed_obs.itertuples(index=False):
            field_id = int(row.field_id)
            if not 0 <= field_id < self.nfields:
                logger.warning(
                    "[Live] Skipping resume row with out-of-catalog field_id %d.",
                    field_id,
                )
                continue
            ot = float(self._ot_at_sunset + (int(row.timestamp) - self._sunset_ts))
            if self.field_level and not self._step_is_valid(dict(
                    field_id=field_id, filter_idx=int(FILTER2IDX[row.filter]), t_start=float(row.timestamp))):
                continue
            if self.do_filt:
                filter_idx = int(FILTER2IDX[row.filter])
                counts[field_id, filter_idx] += 1
                prev = last_visit_ot[field_id, filter_idx]
                if np.isnan(prev) or ot > prev:
                    last_visit_ot[field_id, filter_idx] = ot
            else:
                counts[field_id] += 1
                prev = last_visit_ot[field_id]
                if np.isnan(prev) or ot > prev:
                    last_visit_ot[field_id] = ot

        snap = StateSnapshot(
            timestamp=self._ts,
            field_id=self._field_id,
            bin_num=self._bin_num,
            filter_idx=self._filter_idx,
            counts_cur=counts,
            last_visit_ot_cur=last_visit_ot,
        )
        self.restore_snapshot(snap)
        logger.info(
            "[Live] Resumed survey state from %d completed observations.",
            len(completed_obs),
        )

    def _build_priority_mask(self) -> None:
        """Compute the priority-1 positional mask and keep_only rule.

        No-op-safe: when the lookups carry no `priority` column or no field
        has priority 1, `_priority1_positional` is all-False so the gate never
        masks anything. Rebuilt on `refresh_lookups`.
        """
        if "priority" in self.lookups.fields.columns:
            priorities = self.lookups.fields["priority"].to_numpy()   # [n_fields]
            self._priority1_positional = (priorities == 1)            # [n_fields] bool
        else:
            self._priority1_positional = np.zeros(self.nfields, dtype=bool)
        priority1_ids = self._fids[self._priority1_positional]
        self._priority_rule = MaskRule(
            field_ids=frozenset(int(f) for f in priority1_ids),
            mode="keep_only",
        )

    def refresh_lookups(self, new_lookups) -> None:
        """Adopt a merged catalog mid-session, growing field-shaped state.

        Rebinds lookups; rebuilds field arrays and the priority mask; grows the
        survey tracker and last-visit-OT array to the new field count, zero/NaN
        padding appended fields; then refreshes derived state. The trained
        policy is unaffected (action head is bins x filters, not fields).
        """
        old_n = self.nfields
        self.lookups = new_lookups
        self._set_field_arrays(new_lookups)

        target_counts = (new_lookups.target_fidfilt_counts if self.do_filt
                         else new_lookups.target_fid_counts)
        old_counts = self._survey_progress_tracker.raw_counts
        grown = np.zeros(target_counts.shape, dtype=old_counts.dtype)   # [n_fields(, n_filters)]
        grown[:old_n] = old_counts
        self._survey_progress_tracker = SurveyProgressTracker(target_counts=target_counts)
        self._survey_progress_tracker.set_counts(grown)

        new_ot = np.full(target_counts.shape, np.nan, dtype=np.float64)  # [n_fields(, n_filters)]
        new_ot[:old_n] = self._last_visit_ot
        self._last_visit_ot = new_ot

        self._build_priority_mask()
        self._recompute_derived_state()

    def set_priority_trigger(self, active: bool) -> None:
        """Set the operator priority-scheduling flag and refresh derived state.

        Args
        ----
        active : bool
            When True, the priority gate engages (see `_apply_field_mask`).
        """
        self._priority_trigger = bool(active)
        self._recompute_derived_state()

    def set_field_mask(self, masked_field_ids) -> None:
        """Set the operator field mask and refresh derived state.

        Args
        ----
        masked_field_ids : Iterable[int] or None
            Field ids to drop from the action space. None or empty clears the
            mask. Applied on top of the priority gate (see `_apply_field_mask`).
        """
        self._masked_field_ids = (
            [int(f) for f in masked_field_ids] if masked_field_ids else []
        )
        self._recompute_derived_state()

    def _apply_field_mask(self, sel_valid: np.ndarray) -> np.ndarray:
        """Apply the priority gate and operator field mask to the action mask.

        First applies the priority gate, then drops any operator-masked fields.
        Both act independently, so a field is observable only if it passes the
        gate AND is not in `_masked_field_ids`. `sel_valid` is a fresh `&`
        result, so in-place mutation is safe. Works for `(nfields, nfilters)`
        and `(nfields,)`.
        """
        sel_valid = self._apply_priority_gate(sel_valid)
        if self._masked_field_ids:
            sel_valid[np.isin(self._fids, self._masked_field_ids)] = False
        return sel_valid

    def _apply_priority_gate(self, sel_valid: np.ndarray) -> np.ndarray:
        """Zero non-priority-1 fields while the gate is engaged.

        While the trigger is on and any priority-1 field is still incomplete,
        zero the rows of every non-priority-1 field (keep_only priority-1).
        Auto-releases once no incomplete priority-1 field remains, so the rest
        of the catalog reopens.
        """
        if not self._priority_trigger or not self._priority1_positional.any():
            return sel_valid
        incomplete = self._survey_progress_tracker.get_incomplete_mask()
        incomplete_fields = incomplete.any(axis=1) if incomplete.ndim == 2 else incomplete
        if not (self._priority1_positional & incomplete_fields).any():
            return sel_valid   # priority-1 complete -> release the gate
        sel_valid[resolve_positional_mask(self._priority_rule, self._fids)] = False
        return sel_valid

    def _recompute_derived_state(self) -> None:
        """Refresh action masks and feature vectors after a state change."""
        self._update_action_masks()
        self._global_state = self._calculate_global_features()
        if self.include_bin_features:
            self._bin_state = self._calculate_bin_features()

    # -----------------------------------------------------------------------
    # BaseBlancoEnv lifecycle hooks
    # -----------------------------------------------------------------------

    def reset(self, **kwargs):
        raise RuntimeError(
            "LiveBlancoEnv.reset() must not be called mid-session. "
            "Use sync_telemetry() to realign with hardware state, or "
            "construct a new LiveBlancoEnv for a fresh session."
        )

    def _begin_episode(self, ot_at_sunset=0) -> None:
        self._ot_at_sunset = ot_at_sunset
        self.sync_telemetry()

    def _advance_after_action(self, action: dict) -> None:
        bin_num = int(action["bin"])
        field_id = int(action["field_id"])
        filter_idx = int(action["filter_idx"])

        if bin_num == WAIT_SIGNAL:
            old_ts = self._ts
            self._ts = self._fast_forward()
            logger.info(f"Waited {(self._ts - old_ts) / 60:.1f} minutes")
            # Field/filter unchanged on wait; only bin_num updates below.
            # No visit accumulation — a wait is not an observation.
            self._step_record = None
        else:
            last_field_id, last_filter_idx = self._field_id, self._filter_idx
            distance = self._slew_distance(last_field_id, field_id)
            filter_change = last_filter_idx != ZENITH_FILTER_IDX and last_filter_idx != filter_idx
            dead_time = self._get_dead_time(distance, filter_change)
            exptime = self._get_exposure_time(field_id=field_id, filter_idx=filter_idx)

            self._step_record = dict(
                field_id=field_id, filter_idx=filter_idx, t_start=self._ts + dead_time,
                dead_time=dead_time, filter_change=filter_change,
            )
            self._step_record['uniformity'] = self._uniformity_step_inputs(self._step_record)
            self._ts += dead_time + exptime

            # _record_visit lives on BaseBlancoEnv and translates the
            # action's filter_idx to None automatically when the tracker
            # is field-only (1D).
            self._record_visit(field_id=field_id, filter_idx=filter_idx)

            self._field_id = field_id
            self._filter_idx = filter_idx

        self._bin_num = bin_num
        # Online runs a single open-ended night; never set _is_new_night
        # to True after the initial reset.
        self._is_new_night = False

    def _episode_terminated(self) -> bool:
        # Open-ended; ends only at sunrise. Operator-driven termination
        # (weather, hardware fault) is handled outside the env.
        return self._ts >= self._sunrise_ts

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _refresh_night_boundaries(self) -> None:
        """Recompute sunrise/sunset for the configured observing night."""
        self._sunset_ts, self._sunrise_ts = get_night_boundaries(
            self._ts, sun_el_limit=self.sun_el_limit
        )
        self._night_end_ts = self._sunrise_ts

    def _match_pointing_to_fid(self, ra: float, dec: float) -> int:
        """Return the field ID closest to the given (ra, dec).
        """
        cos_sep = (
            np.sin(self._dec_arr) * np.sin(dec)
            + np.cos(self._dec_arr) * np.cos(dec)
            * np.cos(self._ra_arr - ra)
        )
        return int(self._fids[int(np.argmax(cos_sep))])

    def _fast_forward(self, step_seconds: float = 60.0) -> float:
        """Advance the clock until at least one valid action exists.

        Steps in `step_seconds` increments and re-runs the action mask
        until any field becomes observable, capped at sunrise so we
        never wait past the night. `self._ts` is mutated as a side
        effect during iteration so that `_update_action_masks` (which
        reads `self._ts`) sees the trial time.
        """
        while self._ts < self._sunrise_ts:
            self._ts += step_seconds
            mask = self._update_action_masks()
            if np.any(mask):
                return self._ts
        return self._sunrise_ts


    # -----------------------------------------------------------------------
    # Feature-context hook overrides
    # -----------------------------------------------------------------------

    def _get_t_survey(self) -> Optional[float]:
        s_night_idx = self._get_survey_night_idx()
        s_night_tot = self._get_survey_nights_total()
        return s_night_idx / s_night_tot

    def _get_survey_nights_total(self) -> Optional[int]:
        return 2

    def _get_survey_night_idx(self) -> Optional[int]:
        return self._survey_night_idx

    @property
    def _night_idx(self):
        return self._survey_night_idx
