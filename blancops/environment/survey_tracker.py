"""Survey-wide progress tracking, separated from per-episode visit counters.
 
`SurveyProgressTracker` tracks survey-wide cumulative visit counts. The
backing array is either 1D (per-field, when `do_filt=False`) or 2D
(per-field-and-filter, when `do_filt=True`); the env owns one tracker
whose shape matches the configured action space, and feature/feature-mask
code asks the tracker for derived quantities rather than indexing arrays
directly.
 
Lifecycles by env:
 
  * HistoricBlancoEnv reloads counts at every night boundary from a
    per-night lookup (`set_counts` from a snapshot).
  * SimBlancoEnv carries forward across nights and accumulates per visit
    via `increment` (called from `_record_visit`).
  * OnlineBlancoEnv syncs from telemetry on `sync_telemetry` when the
    snapshot includes counts; otherwise running counts are preserved
    and only `increment` updates them between syncs.
"""

from __future__ import annotations
 
from typing import Optional, Union
import numpy as np
 
 
class SurveyProgressTracker:
    """Cumulative per-field-filter visit counts at the survey level.
 
    Three explicit update modes:
        * `set_counts(arr)` — overwrite running counts (per-night reload,
        telemetry sync, initial seeding).
        * `increment()` — accumulate one visit (called from
        * `zero_counts()` — reset all counts to zero.    
        `_record_visit` after each successful observation).
 
    Querying:
      * `get_filter_progress(filter_idx)` — fractional progress across all fields for a filter
      * `get_field_progress(field_id)` — fractional progress across all filters for a field
      * `get_ff_progress(field_id, filter_idx)` — fractional progress for a specific field-filter pair
      * `get_survey_progress()` — fractional progress across all field-filter pairs.
      * `raw_counts` — read-only view of the raw counts
      * `target_counts` — read-only view of the target counts"""
 
    def __init__(
        self,
        target_counts: np.ndarray,
        initial_counts: Optional[np.ndarray] = None,
    ):
        target_counts = np.asarray(target_counts, dtype=np.int32)
        if target_counts.ndim not in (1, 2):
            raise ValueError(
                "target_counts must be 1D (field-only) or 2D (field-filter); "
                f"got ndim={target_counts.ndim}"
            )

        self._target_counts = target_counts
        self._is_field_filter = self._target_counts.ndim == 2
        
        if initial_counts is None:
            self._counts = np.zeros_like(self._target_counts)
        else:
            initial_counts = np.asarray(initial_counts, dtype=np.int32)
            if initial_counts.shape != self._target_counts.shape:
                raise ValueError(
                    f"initial_counts shape {initial_counts.shape} does not match "
                    f"target_counts shape {self._target_counts.shape}"
                )
            self._counts = initial_counts.copy()
    
    # properties ------------------------------------------------

    @property
    def shape(self) -> tuple:
        return self._counts.shape

    # idx helper ---------------------------------------------------


    def _get_counts_idx(self, field_id: int, filter_idx: Optional[int] = None) -> tuple[int, int] | int:
        if self._is_field_filter:
            if filter_idx is None:
                raise ValueError("filter_idx must be provided for 2D counts")
            return field_id, filter_idx
        if filter_idx is not None:
            raise ValueError("filter_idx must be None for 1D (field-only) counts")
        return field_id
    
    # updates ---------------------------------------------------
        
    def zero_counts(self) -> None:
        """Reset all counts to zero (targets preserved)"""
        self._counts.fill(0)
        
    def set_counts(self, counts: np.ndarray) -> None:
        """Overwrite running counts (per-night reload, telemetry sync)."""
        counts = np.asarray(counts, dtype=np.int32)
        if counts.shape != self._counts.shape:
            raise ValueError(
                f"counts shape {counts.shape} does not match tracker shape "
                f"{self._counts.shape}"
            )
        self._counts[:] = counts
 
    def add_new_field(self, count: np.ndarray | int, targets: np.ndarray | int) -> None:
        """Extends tracker """
        targets = np.asarray(targets, dtype=np.int32)
        if targets.shape != self._target_counts.shape:
            raise ValueError(
                f"targets shape {targets.shape} does not match tracker shape "
                f"{self._target_counts.shape}"
            )
        self._target_counts[:] = targets
        raise NotImplementedError("Adding new fields is not yet implemented")
        
    def increment(self, field_id: int, filter_idx: Optional[int] = None, n: int = 1) -> None:
        """Accumulate `n` visits at the given index."""
        idx = self._get_counts_idx(field_id, filter_idx)
        self._counts[idx] += n

    def copy(self) -> 'SurveyProgressTracker':
        """Return an independent copy (safe to pass to callers that must not mutate live state)."""
        return SurveyProgressTracker(
            target_counts=self._target_counts.copy(),
            initial_counts=self._counts.copy(),
        )

    # raw queries ------------------------------------------------------

    def get_counts(self) -> np.ndarray:
        """Return a writable copy of the raw counts.
 
        Used by callers that need to mutate (e.g. `get_info` building an
        info dict). For read-only access prefer `raw_counts` (cheap view).
        """
        return self._counts.copy()
 
    @property
    def raw_counts(self) -> np.ndarray:
        """Read-only view of raw counts (cheap, no copy)."""
        view = self._counts.view()
        view.flags.writeable = False
        return view
 
    @property
    def target_counts(self) -> np.ndarray:
        """Read-only view of target counts."""
        view = self._target_counts.view()
        view.flags.writeable = False
        return view
 
    @property
    def target_field_counts(self) -> np.ndarray:
        """Per-field target counts (summed over filters in 2D mode)."""
        if self._is_field_filter:
            return self._target_counts.sum(axis=1)
        return self._target_counts.copy()   
 
    @property
    def target_filter_counts(self) -> np.ndarray:
        """Per-filter target counts. Field-filter (2D) mode only."""
        if not self._is_field_filter:
            raise ValueError(
                "target_filter_counts requires a field-filter (2D) tracker"
            )
        return self._target_counts.sum(axis=0)
       
    @property
    def field_counts(self) -> np.ndarray:
        """Per-field current counts (summed over filters in 2D mode).
 
        Always returns a 1D array of length `nfields`.
        """
        if self._is_field_filter:
            return self._counts.sum(axis=1)
        return self._counts.copy()
 
    @property
    def filter_counts(self) -> np.ndarray:
        """Per-filter current counts. Field-filter (2D) mode only."""
        if not self._is_field_filter:
            raise ValueError("filter_counts requires a field-filter (2D) tracker")
        return self._counts.sum(axis=0)
 
 
    # progress queries ----------------------------------------------------

    def get_filter_progress(self, filter_idx: int) -> float:
        """Fractional progress for one filter, summed across fields. 2D only."""
        if not self._is_field_filter:
            raise ValueError(
                "get_filter_progress requires a field-filter (2D) tracker"
            )
        target = int(self._target_counts[:, filter_idx].sum())
        if target == 0:
            return 0.0
        counts = int(self._counts[:, filter_idx].sum())
        return counts / target
    
    def get_field_progress(self, field_id: int) -> float:
        """Fractional progress for one field (summed over filters in 2D mode)."""
        if self._is_field_filter:
            target = int(self._target_counts[field_id].sum())
            counts = int(self._counts[field_id].sum())
        else:
            target = int(self._target_counts[field_id])
            counts = int(self._counts[field_id])
        return counts / target if target > 0 else 0.0

    def get_ff_progress(self, field_id: int, filter_idx: int) -> float:
        """Fractional progress for one (field, filter) cell. 2D only."""
        idx = self._get_counts_idx(field_id, filter_idx)
        target = int(self._target_counts[idx])
        if target == 0:
            return 0.0
        return float(self._counts[idx]) / target

    def get_survey_field_progress(self) -> Union[float, np.ndarray]:
        """Overall progress.
 
        In 1D mode, returns a scalar (total counts / total target).
        In 2D mode, returns a 1D per-field progress array.
        """
        if self._is_field_filter:
            target = self._target_counts.sum(axis=1)
            counts = self._counts.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(target > 0, counts / target, 0.0)
        target = int(self._target_counts.sum())
        if target == 0:
            return 0.0
        return int(self._counts.sum()) / target
    
    def get_survey_progress(self) -> np.ndarray:
        """Element-wise fractional progress; same shape as raw_counts."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(
                self._target_counts > 0,
                self._counts.astype(np.float64) / self._target_counts,
                0.0,
            )
        return out
    
    # mask / completion ---------------------------------------------

    def get_incomplete_mask(self) -> np.ndarray:
        """Boolean mask where counts < target. Same shape as raw_counts.

        Returns a writable array — callers (e.g. `_update_action_masks`)
        combine it in place with visibility/airmass masks via `&=`.
        """
        return self._counts < self._target_counts

        
    def check_completion(self) -> bool:
        """True iff every cell has met or exceeded its target."""
        return bool(np.all(self._counts >= self._target_counts))

    def dispersion_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-filter dispersion of completion, D_b = Var_b(x) / m_b, and its random-visit baseline.

        x = count / target over in-plan fields (target > 0), uncapped. For Poisson visits,
        Var(x_f) = m_b / T_f, so the baseline is mean_f(1 / T_f): D_b below it means more even than random.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (D_b, baseline_b), each shape (n_filters,) (shape (1,) for a field-only tracker); NaN where
            a filter has no in-plan fields or m_b = 0.
        """
        counts = self._counts.reshape(len(self._counts), -1).astype(np.float64)       # [n_fields, n_filters]
        targets = self._target_counts.reshape(len(self._counts), -1).astype(np.float64)
        in_plan = targets > 0
        n = in_plan.sum(axis=0)
        x = np.divide(counts, targets, out=np.zeros_like(counts), where=in_plan)
        inv_t = np.divide(1.0, targets, out=np.zeros_like(targets), where=in_plan)
        mean = np.divide(x.sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)
        sq_dev = np.where(in_plan, (x - np.nan_to_num(mean)) ** 2, 0.0).sum(axis=0)
        var = np.divide(sq_dev, n, out=np.full(n.shape, np.nan), where=n > 0)
        dispersion = np.divide(var, mean, out=np.full(n.shape, np.nan), where=(n > 0) & (mean > 0))
        baseline = np.divide(inv_t.sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)
        return dispersion, baseline
 
    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:
        target_total = int(self._target_counts.sum())
        if target_total == 0:
            progress = 0.0
        else:
            progress = int(self._counts.sum()) / target_total
        return (
            f"SurveyProgressTracker("
            f"shape={self._counts.shape}, "
            f"counts={int(self._counts.sum())}/{target_total}, "
            f"progress={progress:.3f})"
        )
