"""Forward-simulation environment driven by explicit date strings."""
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

import numpy as np

from blancops.environment.base import StateSnapshot
from blancops.environment.offline_base import BaseBlancoOfflineEnv
from blancops.environment.field_mask_schedule import resolve_positional_mask
from blancops.data.features.glob_features import calc_twilight, get_night_boundaries
from blancops.environment.seeing_model import ConstantSeeingModel, PredictiveSeeingModel
from blancops.configs.constants import FWHM_REF_FILTER
from blancops.ephemerides.time_utils import unix_to_datetime

import logging
logger = logging.getLogger(__name__)


class OfflineBlancoEnv(BaseBlancoOfflineEnv):
    """Multi-night forward simulation from explicit date strings.

    Accepts optional seeds (counts, last-visit OT timestamps, OT clock at
    sunset of night 0) for continuing a survey mid-stream. With nothing
    seeded, night 0 starts at OT=0 with no prior visits, and the OT
    clock cascades night-to-night at wall-clock rate from sunset to
    sunrise (regardless of half/full portion — the only formula that
    keeps ot_now = ot_at_sunset + (ts - sunset_ts) monotonic across
    half-night transitions).
    """

    def __init__(
        self,
        *,
        cfg,
        constraints_cfg,
        lookups,
        norm_stats,
        observing_night_strs: list[str],
        initial_counts: Optional[np.ndarray] = None,
        initial_last_visit_ot: Optional[np.ndarray] = None,
        initial_ot_at_sunset: float = 0.0,
        initial_fwhm: Optional[float] = None,
        start_time: Optional[float] = None,
        stop_time: Optional[float] = None,
        downtime_windows=None,
        seeing_trajectory=None,
        field_mask_schedule=None,
        telescope=None,
    ):
        # Parse before super so max_nights is known in time.
        self._night_info = self._parse_night_strs(observing_night_strs)
        # Initialize mask state before super().__init__ so any action-mask
        # refresh during base init is safe (schedule disabled => identity); the
        # positional masks are resolved once _fids exists, then enabled below.
        self._field_mask_schedule = None
        self._rule_positional_masks: dict = {}
        super().__init__(
            cfg=cfg,
            constraints_cfg=constraints_cfg,
            lookups=lookups,
            norm_stats=norm_stats,
            telescope=telescope,
            max_nights=len(self._night_info),
        )
        self._initial_counts = initial_counts
        self._initial_last_visit_ot = initial_last_visit_ot
        self._initial_ot_at_sunset = float(initial_ot_at_sunset)

        # Optional start, stop, and downtime intervals
        self._start_time = None if start_time is None else float(start_time)
        self._stop_time = None if stop_time is None else float(stop_time)
        # Wall-clock intervals in which the telescope was not observing, as
        # (start, end) unix timestamps. Replays idle through them so that a
        # simulated night covers the same observing time a real one did.
        self._downtime_windows = sorted(
            [] if downtime_windows is None
            else [(float(a), float(b)) for a, b in downtime_windows]
        )
        self._downtime_idx = 0
        if self._downtime_windows:
            total = sum(b - a for a, b in self._downtime_windows)
            logger.info(
                f"Downtime: {len(self._downtime_windows)} intervals, "
                f"{total / 60.0:.1f} min total"
            )

        # Seed seeing for the forward simulation. Two mutually exclusive modes:
        #   1. `seeing_trajectory`: replay a real night's measured seeing via a
        #      PredictiveSeeingModel, rebuilt and re-aligned to each night's
        #      sunset in `_start_new_night` (mirrors HistoricBlancoEnv).
        #   2. `initial_fwhm`: assumed delivered zenith seeing in the reference
        #      band, projected per pointing by a ConstantSeeingModel and held
        #      constant across the run.
        # The trajectory wins when both are given.
        self._seeing_trajectory = seeing_trajectory
        if seeing_trajectory is not None:
            if initial_fwhm is not None:
                logger.warning(
                    "OfflineBlancoEnv: both seeing_trajectory and initial_fwhm "
                    "given; replaying seeing_trajectory and ignoring initial_fwhm."
                )
            # Empty predictor so feature validation passes; _start_new_night
            # repopulates it per night, re-aligned to that night's sunset.
            if self._needs_seeing_model():
                self._seeing_model = PredictiveSeeingModel(self.cfg.data.seeing)
        elif initial_fwhm is not None:
            self._seeing_model = ConstantSeeingModel(
                zenith_seeing=float(initial_fwhm), ref_band=FWHM_REF_FILTER,
            )

        # Cache prevents double-advancement if _get_night_config
        # is re-entered for a night that's already been started.
        self._night_cfg_cache: dict[int, dict] = {}

        if self._needs_seeing_model() and self._seeing_model is None:
            raise ValueError(
                "OfflineBlancoEnv: the 'fwhm' feature or the teff reward needs seeing, but "
                "no seeing source was given. Pass initial_fwhm (assumed zenith "
                "seeing, arcsec) for a constant model, or seeing_trajectory to "
                "replay a measured night, so the sim can project seeing per "
                "pointing."
            )

        self._validate_feature_config()

        # Resolve the time-windowed mask schedule to one positional boolean mask
        # per rule (over self._fids), then enable it. Done after super().__init__
        # so self._fids exists. field_ids resolved directly over self._fids.
        if field_mask_schedule is not None:
            for rule in field_mask_schedule.rules():
                self._rule_positional_masks[rule] = resolve_positional_mask(
                    rule, self._fids
                )
            self._field_mask_schedule = field_mask_schedule

    @staticmethod
    def _parse_night_strs(night_strs: list[str]) -> list[tuple[date, str]]:
        """Parse strings like '2026-06-23-half1' or '2026-06-23-full'.

        Returns a `date` (the evening date), not a `datetime`, so
        get_night_boundaries uses it as the canonical evening-date instead of
        treating it as a midnight-UTC instant in the prior local evening.
        """
        parsed = []
        for s in night_strs:
            parts = s.split("-")
            night_date = datetime.strptime(
                "-".join(parts[:3]), "%Y-%m-%d"
            ).date()
            parsed.append((night_date, parts[-1]))
        return parsed

    # -----------------------------------------------------------------------
    # OfflineBlancoEnv hooks
    # -----------------------------------------------------------------------

    def _begin_episode(self) -> None:
        # Restart OT cascade and night cache on every reset; otherwise
        # state leaks from previous episodes.
        self._night_cfg_cache = {}
        super()._begin_episode()

    def _start_new_night(self) -> None:
        super()._start_new_night()
        if self._seeing_trajectory is not None and self._needs_seeing_model():
            self._rebuild_seeing_model_from_trajectory()

    def _rebuild_seeing_model_from_trajectory(self) -> None:
        """Rebuild the seeing predictor from the replay trajectory.

        The trajectory's `sec_since_sunset` offsets are added to this night's
        sunset timestamp, re-aligning the same measured night onto the current
        sim night's clock. Mirrors HistoricBlancoEnv._rebuild_seeing_model.
        """
        traj = self._seeing_trajectory
        model = PredictiveSeeingModel(self.cfg.data.seeing)
        model.add(
            date=self._sunset_ts + traj["sec_since_sunset"].to_numpy(dtype=float),
            seeing=traj["fwhm"].to_numpy(dtype=float),
            band=list(traj["band"]),
            el=traj["el"].to_numpy(dtype=float),
        )
        self._seeing_model = model

    def _get_night_config(self, night_idx: int) -> dict:
        if night_idx in self._night_cfg_cache:
            return self._night_cfg_cache[night_idx]

        night_dt, portion = self._night_info[night_idx]
        sunset_ts, sunrise_ts = get_night_boundaries(
            night_dt, self.sun_el_limit - 0.1
        )

        start_ts, end_ts = sunset_ts, sunrise_ts
        if portion == "half1":
            end_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2
        elif portion == "half2":
            start_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2

        if self._start_time is not None and start_ts < self._start_time < end_ts:
            logger.info(
                "Night start clamped from %s to start_time %s",
                unix_to_datetime(start_ts), unix_to_datetime(self._start_time),
            )
            start_ts = self._start_time

        if self._stop_time is not None and start_ts < self._stop_time < end_ts:
            logger.info(
                "Night end clamped from %s to stop_time %s",
                unix_to_datetime(end_ts), unix_to_datetime(self._stop_time),
            )
            end_ts = self._stop_time


        # Anchor ot_at_sunset so that
        #     ot_now @ start_ts  ==  OT clock at the moment we rolled
        #                            over from the previous night.
        # Derivation: ot_now = ot_at_sunset + (ts - sunset_ts), so for the
        # equality to hold at ts=start_ts:
        #     ot_at_sunset = prev_OT_at_rollover - (start_ts - sunset_ts)
        #
        # For half1/full this offset is 0 (start_ts == sunset_ts).
        # For half2, ot_at_sunset gets a NEGATIVE anchor of -half_dur so that
        # ot_now at the mid-night start point picks up exactly where the
        # previous night left off, instead of jumping ahead by half a night.
        if night_idx == 0:
            prev_OT_at_rollover = self._initial_ot_at_sunset
        else:
            prev_OT_at_rollover = (
                self._ot_at_sunset + (self._ts - self._sunset_ts)
            )
        ot_at_sunset = prev_OT_at_rollover - (start_ts - sunset_ts)

        cfg = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "sunset_ts": sunset_ts,
            "sunrise_ts": sunrise_ts,
            "ot_at_sunset": ot_at_sunset,
        }
        self._night_cfg_cache[night_idx] = cfg
        return cfg
    # def _get_night_config(self, night_idx: int) -> dict:
    #     # Idempotent re-read for the same night (diagnostics, tests).
    #     if night_idx in self._night_cfg_cache:
    #         return self._night_cfg_cache[night_idx]

    #     night_dt, portion = self._night_info[night_idx]
    #     # ts = night_dt.timestamp()
    #     sunset_ts, sunrise_ts = get_night_boundaries(night_dt, self.sun_el_limit - .1)

    #     # sunset = calc_twilight(ts, "set", self.sun_el_limit)
    #     # sunrise = calc_twilight(ts, "rise", self.sun_el_limit)

    #     start_ts, end_ts = sunset_ts, sunrise_ts
    #     if portion == "half1":
    #         end_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2
    #     elif portion == "half2":
    #         start_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2

    #     cfg = {
    #         "start_ts": start_ts,
    #         "end_ts": end_ts,
    #         "sunset_ts": sunset_ts,
    #         "sunrise_ts": sunrise_ts,
    #         "ot_at_sunset": self._running_ot_at_sunset,
    #     }
    #     self._night_cfg_cache[night_idx] = cfg
    #     # Advance OT cascade for the next night: full sunset→sunrise
    #     # span, irrespective of half-night portion. See class docstring.
    #     self._running_ot_at_sunset += (end_ts - start_ts)
    #     return cfg

    def _build_night_start_snapshot(self, night_idx: int) -> StateSnapshot:
        cfg = self._get_night_config(night_idx)

        if night_idx == 0:
            # Seed counters and last-visit OT timestamps from constructor
            # kwargs if provided. None means "leave the (already-zeroed-
            # by-reset) state alone" — see base.reset().
            counts_cur = (
                self._initial_counts.copy()
                if self._initial_counts is not None
                else None
            )
            last_visit_ot_cur = (
                self._initial_last_visit_ot.copy()
                if self._initial_last_visit_ot is not None
                else None
            )
            return StateSnapshot(
                timestamp=cfg["start_ts"],
                counts_cur=counts_cur,
                last_visit_ot_cur=last_visit_ot_cur,
            )

        # Subsequent nights: just advance the clock. Tracker and
        # _last_visit_ot carry forward via _apply_state_snapshot's
        # None-skip behaviour.
        return StateSnapshot(timestamp=cfg["start_ts"])

    def _apply_field_mask(self, sel_valid: np.ndarray) -> np.ndarray:
        """Zero validity rows for fields masked by the active schedule rule.

        The active rule is selected by the current sim time (self._ts), so the
        masking tracks the schedule's time windows. Identity when no schedule is
        set. Works for both the (nfields, nfilters) and (nfields,) mask shapes
        since field index is axis 0 in both.

        A `keep_only` window releases as soon as every field it keeps is
        complete (mimics gw followup in live scheduling)
        """
        if self._field_mask_schedule is None:
            return sel_valid
        rule = self._field_mask_schedule.active_rule(self._ts)
        if rule.mode == "keep_only" and self._keep_only_satisfied(rule):
            rule = self._field_mask_schedule.baseline
            if rule.mode == "keep_only" and self._keep_only_satisfied(rule):
                return sel_valid
        positional = self._rule_positional_masks[rule]
        sel_valid[positional] = False
        return sel_valid

    def _keep_only_satisfied(self, rule) -> bool:
        """Whether every field a `keep_only` rule retains is already complete.

        Args:
            rule: The active MaskRule.

        Returns:
            True when no field kept by the rule still owes a visit, so the rule
            has nothing left to enforce.
        """
        if not rule.field_ids:
            return False
        kept = np.isin(self._fids, np.fromiter(rule.field_ids, dtype=int))
        incomplete = self._survey_progress_tracker.get_incomplete_mask()
        if incomplete.ndim == 2:
            incomplete = incomplete.any(axis=1)
        return not bool((kept & incomplete).any())
