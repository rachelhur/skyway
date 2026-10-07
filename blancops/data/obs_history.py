"""Seed offline-scheduler state from a prior observing history.

Reads either a schedule CSV (this package's ``run_offline_scheduler`` /
``OfflineRunner`` output) or a live JSONL observing log (``live_scheduler``
``ProgressManager`` output) and reconstructs the survey-progress seed values
consumed by ``OfflineBlancoEnv``:

- ``initial_counts``        per-(field, filter) visit counts
- ``initial_last_visit_ot`` per-(field, filter) last-visit observing-time (OT)
- ``initial_ot_at_sunset``  cumulative OT clock at the first new night's sunset

The per-night OT/counts accumulation mirrors ``build_DES_lookups`` in
``blancops/data/preprocessing.py`` (the FITS training pipeline) so seeded runs
share the same OT frame as the trained model.
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from blancops.survey.profiles import DES, SurveyProfile
from blancops.data.features.glob_features import get_night_boundaries

import logging
logger = logging.getLogger(__name__)


def build_synthetic_obs_history(
    lookups,
    span_nights,
    end_date,
    sun_el_limit,
    *,
    visits_by_propid,
    seed=42,
) -> pd.DataFrame:
    """Synthesize an obs-history frame seeding completed visits per (field, filter).

    Completed-visit counts are defined entirely by ``visits_by_propid``, a
    required ``{propid: {filter: completed_visits}}`` mapping giving an absolute
    completed count for every (propid, filter) -- e.g. ``0`` to start a
    time-critical program fresh. Every targeted (field, filter) must have a
    defined count: a missing propid or filter, or a count exceeding that cell's
    target, raises ``ValueError``. Requires a ``propid`` column on
    ``lookups.fields``. This reads the merged lookup directly, so a merged
    multi-program lookup needs no separate history merge: field ids and
    per-program targets already carry through.

    Each field's most-recent visit night is drawn uniformly across a
    ``span_nights`` window of consecutive observing nights ending at
    ``end_date``, and its visits are placed on that night and the preceding ones
    (per filter). Spreading the most-recent night across fields gives the
    cross-field staleness spread that ``rel_t_since_last_visit`` is normalized
    against; the window's total observing time sets that spread's magnitude.

    Visit timestamps are night midpoints (via ``get_night_boundaries``), so every
    row's night key and last-visit OT resolve consistently in
    ``load_seed_state_from_obs_history``.

    Args:
        lookups: LookupTables; supplies field ids, per-(field, filter) targets,
            and (for overrides) the propid column.
        span_nights: number of consecutive nights the history spans.
        end_date: evening-date of the latest history night (str or date-like).
        sun_el_limit: sun-elevation limit (deg); must match the consuming run.
        visits_by_propid: required {propid: {filter: completed_visits}} mapping;
            every targeted (propid, filter) must be present.
        seed: RNG seed for per-field recency assignment.

    Returns:
        DataFrame with columns field_id, filter, timestamp.
    """
    if span_nights <= 0:
        raise ValueError(f"span_nights must be positive; got {span_nights}.")

    targets = lookups.target_fidfilt_counts            # (nfields, nfilters)
    field_ids = lookups.fields.index.to_numpy()
    idx2filter = lookups.survey.idx2filter

    if not visits_by_propid:
        raise ValueError("visits_by_propid is required and must be non-empty.")
    if "propid" not in lookups.fields.columns:
        raise ValueError(
            "visits_by_propid requires a 'propid' column on lookups.fields."
        )
    propids = lookups.fields["propid"].to_numpy()

    end = pd.Timestamp(end_date).date()
    # Evening-dates, earliest first; index span_nights-1 is the latest night.
    dates = [end - timedelta(days=span_nights - 1 - k) for k in range(span_nights)]
    midpoints = np.array(
        [0.5 * sum(get_night_boundaries(d, sun_el_limit=sun_el_limit)) for d in dates],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    recent = rng.integers(0, span_nights, size=len(field_ids))

    records = []
    for row, (fid, r) in enumerate(zip(field_ids, recent)):
        propid = propids[row]
        filt_counts = visits_by_propid.get(propid)
        if filt_counts is None:
            raise ValueError(
                f"propid {propid!r} has no entry in visits_by_propid "
                f"(defined: {sorted(visits_by_propid)})."
            )
        for f in np.nonzero(targets[row] > 0)[0]:
            target_f = int(targets[row, f])
            filt = idx2filter[int(f)]
            if filt not in filt_counts:
                raise ValueError(
                    f"propid {propid!r} is missing a count for targeted filter "
                    f"{filt!r} in visits_by_propid (has: {sorted(filt_counts)})."
                )
            completed = int(filt_counts[filt])
            if completed < 0 or completed > target_f:
                raise ValueError(
                    f"visits_by_propid[{propid!r}][{filt!r}]={completed} is out "
                    f"of range; must be in [0, {target_f}] (the (field {int(fid)}, "
                    f"{filt}) target)."
                )
            for k in range(completed):
                j = max(0, int(r) - k)
                records.append((int(fid), filt, float(midpoints[j])))

    if not records:
        raise ValueError(
            "build_synthetic_obs_history produced no visits: every (field, "
            "filter) resolved to 0 completed visits. Check that the "
            f"visits_by_propid counts {visits_by_propid} are intended."
        )
    return pd.DataFrame(records, columns=["field_id", "filter", "timestamp"])


# Candidate column names across the two supported formats. Schedule CSVs use the
# agent_* names from io.schedule_io.SCHEDULE_KEYS; live JSONL logs use the bare
# proposal-row names emitted by live_scheduler.model_runner.
_FIELD_ID_COLS = ("agent_field_id", "field_id")
_TIMESTAMP_COLS = ("agent_timestamp", "timestamp")
_FILTER_IDX_COLS = ("agent_filter_idx", "filter_idx")
_FILTER_NAME_COLS = ("agent_filter", "filter")


def _first_present(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Observation history is missing a {what} column "
        f"(looked for {list(candidates)}); found {list(df.columns)}."
    )


def _read_obs_history(path: Path) -> pd.DataFrame:
    """Load a schedule CSV or live JSONL log into a raw DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".jsonl", ".json"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(
        f"Unsupported obs-history format '{suffix}' for {path}; "
        f"expected .csv (schedule) or .jsonl/.json (live log)."
    )


def _normalize(df: pd.DataFrame, survey: SurveyProfile = DES) -> pd.DataFrame:
    """Reduce a raw history frame to columns: field_id, filt_idx, timestamp, night."""
    if df.empty:
        raise ValueError("Observation history is empty.")

    field_col = _first_present(df, _FIELD_ID_COLS, "field id")
    ts_col = _first_present(df, _TIMESTAMP_COLS, "timestamp")

    field_id = df[field_col].to_numpy(dtype=np.int64)
    timestamp = df[ts_col].to_numpy(dtype=np.float64)

    # Filter index may be stored directly (CSV) or as a name needing the survey's filter2idx.
    idx_col = next((c for c in _FILTER_IDX_COLS if c in df.columns), None)
    if idx_col is not None:
        filt_idx = df[idx_col].to_numpy(dtype=np.int64)
    else:
        name_col = _first_present(df, _FILTER_NAME_COLS, "filter")
        mapped = df[name_col].map(survey.filter2idx)
        if mapped.isna().any():
            bad = sorted(set(df[name_col][mapped.isna()]))
            raise ValueError(f"Unrecognized filter name(s) in history: {bad}.")
        filt_idx = mapped.to_numpy(dtype=np.int64)

    out = pd.DataFrame(
        {"field_id": field_id, "filt_idx": filt_idx, "timestamp": timestamp}
    )
    # Night key: matches fits_io._add_night (datetime - 12h -> date) so night
    # grouping is consistent with the FITS training lookups.
    dt = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    out["night"] = (dt - pd.Timedelta(hours=12)).dt.date
    return out


def load_seed_state_from_obs_history(path, lookups, sun_el_limit):
    """Reconstruct survey-progress seeds from a prior observing history.

    Args:
        path: schedule CSV (.csv) or live JSONL log (.jsonl/.json).
        lookups: LookupTables; supplies the (n_fields, n_filters) shape.
        sun_el_limit: sun-elevation limit (deg) bounding each observing night.

    Returns:
        (initial_counts, initial_last_visit_ot, initial_ot_at_sunset)
        with shapes matching lookups.target_fidfilt_counts. Mirrors the
        per-night accumulation in blancops/data/preprocessing.py.
    """
    path = Path(path)
    df = _normalize(_read_obs_history(path), lookups.survey)

    n_fields, n_filters = lookups.target_fidfilt_counts.shape
    counts = np.zeros((n_fields, n_filters), dtype=np.int64)
    last_visit_ot = np.full((n_fields, n_filters), np.nan, dtype=np.float64)

    cum_ot = 0.0
    for night, night_df in df.groupby("night"):
        sunset_ts, sunrise_ts = get_night_boundaries(night, sun_el_limit=sun_el_limit)
        night_dur = sunrise_ts - sunset_ts

        # Visit counts: every logged row is a real exposure (no teff filter,
        # unlike the FITS pipeline whose raw catalog includes invalid frames).
        np.add.at(
            counts,
            (night_df["field_id"].to_numpy(), night_df["filt_idx"].to_numpy()),
            1,
        )

        # Per-(field, filter) last-visit OT, in cumulative seconds since survey
        # start: ot = cum_ot + (timestamp - sunset_ts). np.fmax keeps the latest
        # visit and is NaN-aware (incoming value wins over an existing NaN).
        ot = cum_ot + (night_df["timestamp"].to_numpy() - sunset_ts)
        ff_max = night_df.assign(ot=ot).groupby(["field_id", "filt_idx"])["ot"].max()
        keys = np.array(ff_max.index.tolist(), dtype=np.int64)
        rows, cols = keys[:, 0], keys[:, 1]
        last_visit_ot[rows, cols] = np.fmax(last_visit_ot[rows, cols], ff_max.to_numpy())

        cum_ot += night_dur

    initial_ot_at_sunset = float(cum_ot)
    logger.info(
        "Seeded survey state from %s: %d visits across %d (field, filter) cells; "
        "initial_ot_at_sunset=%.1f s.",
        path.name, int(counts.sum()), int((counts > 0).sum()), initial_ot_at_sunset,
    )
    return counts, last_visit_ot, initial_ot_at_sunset
