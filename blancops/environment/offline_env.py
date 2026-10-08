"""Forward-simulation environment over resolved observing windows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from blancops.environment.base import StateSnapshot
from blancops.environment.offline_base import BaseBlancoOfflineEnv
from blancops.environment.field_mask_schedule import resolve_positional_mask
from blancops.data.features.glob_features import get_night_boundaries
from blancops.environment.seeing_model import ConstantSeeingModel, PredictiveSeeingModel
from blancops.survey.profiles import DES
from blancops.ephemerides.time_utils import unix_to_datetime

import logging
logger = logging.getLogger(__name__)

NIGHT_PORTIONS = ("full", "half1", "half2")


@dataclass(frozen=True)
class ObservingWindow:
    """One night's observing window; timestamps are unix s (UTC).

    Parameters
    ----------
    label : str
        Night name used in output file names, starting with the evening date.
    start_ts, end_ts : float
        Observing start and end.
    sunset_ts, sunrise_ts : float
        That night's sunset and sunrise at the sun elevation limit.
    """
    label: str
    start_ts: float
    end_ts: float
    sunset_ts: float
    sunrise_ts: float


def resolve_observing_windows(
    sun_el_limit: float,
    *,
    observing_nights: Optional[list[str]] = None,
    start_time: Optional[float] = None,
    stop_time: Optional[float] = None,
) -> list[ObservingWindow]:
    """Observing windows from night strings or from one exact start/stop window.

    Parameters
    ----------
    sun_el_limit : float
        Highest sun elevation for observing, in degrees.
    observing_nights : list of str, optional
        Night strings `YYYY-MM-DD-<portion>`, portion in `NIGHT_PORTIONS`.
    start_time, stop_time : float, optional
        Window start and stop (unix s). A missing end is that night's sunset or sunrise.

    Returns
    -------
    list of ObservingWindow
        One window per night string, or a single window labeled `<evening date>-window`.
    """
    limit = sun_el_limit - 0.1
    has_window = start_time is not None or stop_time is not None
    if bool(observing_nights) == has_window:
        raise ValueError("Give either --observing_nights or --start_time / --stop_time, not both or neither.")

    if has_window:
        anchor = start_time if start_time is not None else stop_time
        sunset_ts, sunrise_ts = get_night_boundaries(float(anchor), limit)
        start_ts = sunset_ts if start_time is None else float(start_time)
        end_ts = sunrise_ts if stop_time is None else float(stop_time)
        if not (sunset_ts <= start_ts < end_ts <= sunrise_ts):
            raise ValueError(
                f"The window must start before it stops and lie within one night at "
                f"sun_el_limit={sun_el_limit}: sunset {unix_to_datetime(sunset_ts):%Y-%m-%d %H:%M:%S}, "
                f"sunrise {unix_to_datetime(sunrise_ts):%Y-%m-%d %H:%M:%S} UTC."
            )
        evening_date = (unix_to_datetime(sunset_ts) - timedelta(hours=12)).date()
        return [ObservingWindow(label=f"{evening_date}-window", start_ts=start_ts, end_ts=end_ts,
                     sunset_ts=sunset_ts, sunrise_ts=sunrise_ts)]

    windows = []
    for night_str in observing_nights:
        date_str, _, portion = night_str.rpartition("-")
        if portion not in NIGHT_PORTIONS:
            raise ValueError(f"Observing night {night_str!r} must end in one of {', '.join(NIGHT_PORTIONS)}.")
        sunset_ts, sunrise_ts = get_night_boundaries(datetime.strptime(date_str, "%Y-%m-%d").date(), limit)
        mid_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2
        windows.append(ObservingWindow(
            label=night_str,
            start_ts=mid_ts if portion == "half2" else sunset_ts,
            end_ts=mid_ts if portion == "half1" else sunrise_ts,
            sunset_ts=sunset_ts, sunrise_ts=sunrise_ts,
        ))
    return windows


class OfflineBlancoEnv(BaseBlancoOfflineEnv):
    """Multi-night forward simulation over resolved observing windows.

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
        observing_windows: list[ObservingWindow],
        initial_counts: Optional[np.ndarray] = None,
        initial_last_visit_ot: Optional[np.ndarray] = None,
        initial_ot_at_sunset: float = 0.0,
        initial_fwhm: Optional[float] = None,
        downtime_windows=None,
        seeing_trajectory=None,
        field_mask_schedule=None,
        telescope=None,
        survey=DES,
        reset_counts_on_exhaustion: bool = False,
    ):
        self._observing_windows = list(observing_windows)
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
            survey=survey,
            max_nights=len(self._observing_windows),
            reset_counts_on_exhaustion=reset_counts_on_exhaustion,
        )
        self._check_windows_within_nights()
        self._initial_counts = initial_counts
        self._initial_last_visit_ot = initial_last_visit_ot
        self._initial_ot_at_sunset = float(initial_ot_at_sunset)

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
                zenith_seeing=float(initial_fwhm), ref_band=self._survey.seeing_ref_filter,
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

    # -----------------------------------------------------------------------
    # OfflineBlancoEnv hooks
    # -----------------------------------------------------------------------

    def _check_windows_within_nights(self) -> None:
        """Raise if a window is not inside its night at this env's `sun_el_limit`.

        Raises
        ------
        ValueError
            A window starts before sunset or ends after sunrise, as computed by
            `get_night_boundaries(..., sun_el_limit - 0.1)`.
        """
        for window in self._observing_windows:
            sunset_ts, sunrise_ts = get_night_boundaries(window.sunset_ts, self.sun_el_limit - 0.1)
            if not (sunset_ts <= window.start_ts < window.end_ts <= sunrise_ts):
                raise ValueError(
                    f"Observing window {window.label!r} is not inside its night at the env's "
                    f"sun_el_limit={self.sun_el_limit}; resolve it with the same sun_el_limit."
                )

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

    def night_label(self, night_idx: int) -> str:
        """The window's label, e.g. '2026-06-23-half2' or '2026-06-23-window'."""
        return self._observing_windows[night_idx].label

    def _get_night_config(self, night_idx: int) -> dict:
        if night_idx in self._night_cfg_cache:
            return self._night_cfg_cache[night_idx]

        window = self._observing_windows[night_idx]
        start_ts, sunset_ts = window.start_ts, window.sunset_ts

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

        cfg = dict(asdict(window), ot_at_sunset=ot_at_sunset)
        self._night_cfg_cache[night_idx] = cfg
        return cfg

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
