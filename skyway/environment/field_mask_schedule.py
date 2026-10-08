"""Field-masking schedules for online and offline rollouts.
Holds masking rules for:
    - Field priority-mask schedule for online and offline rollouts.
    - Time-windowed field-mask schedule for offline rollouts.

Holds the masking spec (a baseline rule plus time windows) and selects the
active rule for a given timestamp. Pure spec and selection: no ephemeris or
environment coupling, so it is independently testable. Resolution of a rule to a
positional boolean mask over a field-id array lives in `resolve_positional_mask`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_VALID_MODES = ("mask", "keep_only")


@dataclass(frozen=True)
class MaskRule:
    """A single masking rule.

    mode "mask" masks the listed field_ids; mode "keep_only" masks the
    complement (every field whose id is not listed).
    """
    field_ids: frozenset
    mode: str

    def __post_init__(self):
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"MaskRule.mode must be one of {_VALID_MODES}, got {self.mode!r}."
            )


@dataclass(frozen=True)
class MaskWindow:
    """A masking rule active over the half-open interval [start, end)."""
    start: float
    end: float
    rule: MaskRule

    def __post_init__(self):
        if not self.start < self.end:
            raise ValueError(
                f"MaskWindow requires start < end, got start={self.start}, "
                f"end={self.end}."
            )

    def contains(self, ts: float) -> bool:
        return self.start <= ts < self.end


@dataclass(frozen=True)
class FieldMaskSchedule:
    """Baseline mask plus time-windowed overrides.

    `active_rule(ts)` returns the first window whose [start, end) contains ts,
    else the baseline rule.
    """
    baseline: MaskRule
    windows: tuple = ()

    def rules(self) -> list:
        """All rules in the schedule (baseline first, then each window)."""
        return [self.baseline] + [w.rule for w in self.windows]

    def active_rule(self, ts: float) -> MaskRule:
        for window in self.windows:
            if window.contains(ts):
                return window.rule
        return self.baseline

    @classmethod
    def build(cls, baseline_field_ids, baseline_mode="mask",
              window_start=None, window_end=None,
              window_field_ids=None, window_mode="keep_only"):
        """Build a schedule from flat CLI-style args.

        Returns None when no baseline field_ids are given (no masking). A
        single window is added when start, end, and field_ids are all provided.
        """
        if not baseline_field_ids:
            return None
        baseline = MaskRule(field_ids=frozenset(int(f) for f in baseline_field_ids),
                            mode=baseline_mode)

        windows = ()
        if window_start is not None and window_end is not None and window_field_ids:
            windows = (
                MaskWindow(
                    start=float(window_start),
                    end=float(window_end),
                    rule=MaskRule(
                        field_ids=frozenset(int(f) for f in window_field_ids),
                        mode=window_mode,
                    ),
                ),
            )
        return cls(baseline=baseline, windows=windows)


def resolve_positional_mask(rule, fids) -> np.ndarray:
    """Resolve a MaskRule to a positional boolean mask over `fids`.

    Args
    ----
    rule : MaskRule
        Rule carrying the field_ids to act on and the mode.
    fids : np.ndarray
        Field-id array (axis 0 of the action mask).

    Returns
    -------
    np.ndarray
        Boolean mask, True at positions of fields to drop: the rule's
        field_ids for "mask", the complement for "keep_only".
    """
    positional = np.isin(fids, sorted(rule.field_ids))   # [n_fields]
    return ~positional if rule.mode == "keep_only" else positional
