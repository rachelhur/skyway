"""Validation environment driven by recorded survey nights."""
from __future__ import annotations

from typing import Optional

import numpy as np

from blancops.environment.base import StateSnapshot
from blancops.environment.seeing_model import PredictiveSeeingModel
from blancops.data.features.glob_features import get_night_boundaries
from blancops.configs.constants import IDX2FILTER, FWHM_REF_FILTER

import logging

from blancops.environment.offline_base import BaseBlancoOfflineEnv
logger = logging.getLogger(__name__)

class HistoricBlancoEnv(BaseBlancoOfflineEnv):
    """Validation against historically observed nights.

    Driven by a pandas groupby keyed on night. Each night's initial visit
    state comes from `lookups.night2fidfilt_visit_hist[night_id]` (full-survey
    seeded); per-night seeing splines are passed at construction for the
    fwhm feature hook.

    Survey-position context (``_survey_night_idx`` and ``_get_survey_nights_total``)
    now derived from ``lookups.night2idx`` and ``lookups.total_nights``,
    """

    def __init__(
        self,
        *,
        cfg,
        constraints_cfg,
        lookups,
        z_score_stats,
        rel_norm_stats,
        global_pd_nightgroup,
        night_start_bin_states: Optional[np.ndarray] = None,
        telescope=None,
    ):
        super().__init__(
            cfg=cfg,
            constraints_cfg=constraints_cfg,
            lookups=lookups,
            z_score_stats=z_score_stats,
            rel_norm_stats=rel_norm_stats,
            telescope=telescope,
            max_nights=global_pd_nightgroup.ngroups,
        )
        self._groupbynight = global_pd_nightgroup
        self._night_keys = list(global_pd_nightgroup.groups.keys())
        self._night_start_bin_states = night_start_bin_states

        # Per-night feature context.
        self._survey_night_idx = 0  # set per-night in _get_night_config

        # Seeing predictor. A model is required at construction so feature
        # validation passes; _start_new_night rebuilds it from each night's
        # measurements. Empty until then (predict falls back to nominal).
        if "fwhm" in self.global_feature_names:
            self._seeing_model = PredictiveSeeingModel(self.cfg.data.seeing)

        # Guard rails: features that need full-survey context cannot be
        # served if the lookups don't carry it. Fail loudly at construction
        # rather than producing wrong-but-plausible values at training time.
        if "t_survey" in self.global_feature_names:
            if not hasattr(lookups, "night2idx") or lookups.night2idx is None:
                raise ValueError(
                    "HistoricBlancoEnv: 't_survey' feature requested but "
                    "`lookups.night2idx` is missing. Rebuild lookups with "
                    "`build_train_lookups.py` to populate the night index."
                )
        self._validate_feature_config()

    # -----------------------------------------------------------------------
    # OfflineBlancoEnv hooks
    # -----------------------------------------------------------------------

    def _get_night_config(self, night_idx: int) -> dict:
        """Build the per-night timing/seed config.

        `night_idx` is the EPISODE-LOCAL counter (0..max_nights-1 within
        this run). `self._survey_night_idx` is the FULL-SURVEY index of
        the same night, derived from `lookups.night2idx`.
        """
        night_key = self._night_keys[night_idx]
        night_df = self._groupbynight.get_group(night_key)
        first_row = night_df.iloc[0]
        last_row = night_df.iloc[-1]

        sunset_ts, sunrise_ts = get_night_boundaries(first_row["night"], self.sun_el_limit)
        # self._ot_at_sunset: float | None = None,

        # Full-survey night index from lookups, not from a preprocessing
        # column. The preprocessing `night_idx` (if present) is subset-local
        # and would give wrong t_survey / urgency values for any training
        # subset that doesn't start at the beginning of the survey.
        if self.lookups.night2idx is not None:
            self._survey_night_idx = int(self.lookups.night2idx[first_row["night"]])
        else:
            # Fallback: episode-local. Only correct if the episode IS the
            # full survey. Logged at debug since most callers don't request
            # survey-position features.
            self._survey_night_idx = night_idx

        field_id = int(first_row["field_id"])
        filter_idx = int(first_row["filter_idx"])
        bin_num = int(first_row["bin"])

        night2ot = self.lookups.night2ot_clock_seconds
        if night2ot is None:
            raise ValueError(
                "HistoricBlancoEnv requires lookups.night2ot_clock_seconds for "
                "OT-clock staleness; rebuild lookups via build_train_lookups.py."
            )

        return {
            "start_ts":   first_row["timestamp"],
            "end_ts":     last_row["timestamp"],
            "sunset_ts":  sunset_ts,
            "sunrise_ts": sunrise_ts,
            "ot_at_sunset": int(night2ot[first_row["night"]]),
            "field_id":   field_id,
            "filter_idx": filter_idx,
            "bin_num":    bin_num,
        }


    def _build_night_start_snapshot(self, night_idx: int) -> StateSnapshot:
        night_id = self._night_keys[night_idx]
        night_cfg = self._get_night_config(night_idx)

        if self._night_start_bin_states is not None and self.include_bin_features:
            self._bin_state = self._night_start_bin_states[night_idx]

        counts_lookup = (
            self.lookups.night2fidfilt_visit_hist
            if self.do_filt
            else self.lookups.night2fid_visit_hist
        )
        last_visit_ot_lookup = (
            self.lookups.night2fidfilt_last_visit_ot
            if self.do_filt
            else self.lookups.night2fid_last_visit_ot
        )
        if last_visit_ot_lookup is None:
            raise ValueError(
                "HistoricBlancoEnv requires night2{fid,fidfilt}_last_visit_ot in "
                "lookups for OT-clock staleness; rebuild lookups."
            )

        return StateSnapshot(
            timestamp=night_cfg["start_ts"],
            field_id=night_cfg["field_id"],
            bin_num=night_cfg["bin_num"],
            filter_idx=night_cfg["filter_idx"],
            counts_cur=counts_lookup[night_id].copy(),
            last_visit_ot_cur=last_visit_ot_lookup[night_id].copy().astype(np.float64),
        )

    # -----------------------------------------------------------------------
    # Feature-context hook overrides
    # -----------------------------------------------------------------------

    def _get_t_survey(self) -> Optional[float]:
        """t_survey = survey_night_idx / total_nights. Both derived from
        lookups, so the value is correct regardless of training subset."""
        total = self._get_survey_nights_total()
        if total is None or total == 0:
            return None
        return self._survey_night_idx / total

    def _start_new_night(self) -> None:
        super()._start_new_night()
        self._rebuild_seeing_model()

    def _rebuild_seeing_model(self) -> None:
        """Build a fresh seeing predictor from this night's measurements.

        The whole night's measured (timestamp, fwhm, band, el) is ingested
        up front; predict()'s strict-past window keeps each step causal,
        and the agent's pointing (band/el) is honored per query via the base
        _get_fwhm delegate.

        No-op when the config does not request the fwhm feature: leaving
        `_seeing_model` as None keeps `_get_fwhm` a no-op and avoids the
        predictor's empty-history fallback warning.
        """
        if "fwhm" not in self.global_feature_names:
            return
        night_key = self._night_keys[self._night_idx]
        night_df = self._groupbynight.get_group(night_key)
        model = PredictiveSeeingModel(self.cfg.data.seeing)
        fwhm_vals = night_df['fwhm'].to_numpy(dtype=float)
        valid = ~np.isnan(fwhm_vals)
        if valid.any():
            filt = night_df['filter_idx'].to_numpy()[valid]
            bands = [IDX2FILTER.get(int(f), FWHM_REF_FILTER) for f in filt]
            model.add(
                date=night_df['timestamp'].to_numpy(dtype=float)[valid],
                seeing=fwhm_vals[valid],
                band=bands,
                el=night_df['el'].to_numpy(dtype=float)[valid],
            )
        self._seeing_model = model

    def _get_survey_nights_total(self) -> Optional[int]:
        return self.lookups.total_nights

    def _get_survey_night_idx(self) -> Optional[int]:
        return self._survey_night_idx

