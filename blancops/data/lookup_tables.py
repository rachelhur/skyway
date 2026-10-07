import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from blancops.configs.enums import AcceptanceRule, LookupKeys
from blancops.survey.profiles import DES, SurveyProfile
from blancops.data.features.glob_features import get_night_boundaries
from blancops.math import units

logger = logging.getLogger(__name__)

def _calc_total_survey_ot(observing_nights, sun_el_limit=-10, per_night_overshoot_s=120.0,):
    ot_total = 0
    for i, night_str in enumerate(observing_nights):
        parts = night_str.split('-')
        night_portion = parts[-1]
        date_str = "-".join(parts[:3])
        sunset, sunrise = get_night_boundaries(date_str, sun_el_limit=sun_el_limit)
        night_dur = sunrise - sunset
        is_half = (night_portion == 'half1') or (night_portion == 'half2') or (night_portion == 'half')
        if night_portion == 'full':
            ot_total += night_dur
        elif is_half:
            ot_total += night_dur / 2
        else:
            raise ValueError(f"The observing night portion must be one of `full`, `half1`, `half2` or `half`. \
                                The observing night given at index {i} was {night_str}")
    return ot_total + per_night_overshoot_s #  buffer

@dataclass(frozen=True)
class LookupTables:
    """Universal container for telescope/survey metadata.

    **Shape contract: `*_fidfilt_*` are of shape `(len(fields), survey.num_filters)`,
        indexed by `field_id` along axis 0 and `filter_idx` along axis 1.
        The `fields` index must be `0..N-1` contiguous so array index and `field_id` coincide;
        `__post_init__` enforces this.
        #TODO do I want to change from contiguous to saving an additional fid->idx mapping?

    Identity: `field_id` (the `fields` index) is the sole field identifier. The
    `field` column is a human-readable label only and may repeat across propids
    (distinct programs can reuse a name); never use it to identify a field.
    """
    # Required lookup tables
    fields: pd.DataFrame                # index=field_id (sole identity); cols: field (label), ra, dec
    target_fidfilt_counts: np.ndarray   # (nfields, nfilters) int
    fidfilt_exptime: np.ndarray         # (nfields, nfilters) float
    dir: Path
    # total_ot_sec: float

    # Derived marginals — populated in __post_init__, never set by callers.
    target_fid_counts: np.ndarray = field(init=False, default=None, repr=False)
    target_filt_counts: np.ndarray = field(init=False, default=None, repr=False)

    # Optional historical counts
    historic_df: Optional[pd.DataFrame] = None

    # Survey whose filter order indexes axis 1 of the `*_fidfilt_*` arrays
    survey: SurveyProfile = DES

    # Total survey time (past and future)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def _load_base_kwargs(cls, data_dir: Path, overrides: Dict[LookupKeys, str]) -> dict:
        """Helper to parse and load the base attributes shared by all lookup classes."""
        def get_path(key):
            return data_dir / overrides.get(key, key.value)

        # Fields DataFrame (canonical per-field data)
        fields = pd.read_json(get_path(LookupKeys.FIELDS))
        if "field_id" in fields.columns:
            fields = fields.set_index("field_id")
        fields = fields.sort_index()
        fields.index.name = "field_id"

        # Per-(field, filter) matrices
        with open(get_path(LookupKeys.TARGET_FIDFILT_COUNTS), "rb") as f:
            target_fidfilt_counts = pickle.load(f)
        with open(get_path(LookupKeys.FIDFILT_EXPTIME), "rb") as f:
            fidfilt_exptime = pickle.load(f)
        # with open(get_path(LookupKeys.TOTAL_OT_SECONDS), "rb") as f:
        #     total_ot_sec = np.float64(f.read())

        return {
            "fields": fields,
            "target_fidfilt_counts": target_fidfilt_counts,
            "fidfilt_exptime": fidfilt_exptime,
            "dir": data_dir,
            # "total_ot_sec": total_ot_sec
        }

    def _load_historic_kwargs(cls, data_dir: Path, overrides: Dict[LookupKeys, str]) -> dict:
        raise NotImplementedError("On the todo list...")
        # def get_path(key):
        #     return data_dir / overrides.get(key, key.value)

        # historic_df = pd.read_json(get_path(LookupKeys.HISTORIC_OBSERVATIONS))
        # required = {"ra", "dec", "filter", "count", "exptime"}
        # missing = required - set(historic_df.columns)
        # if missing:
        #     raise ValueError(f"Missing columns: {missing}")

        # np.add.at(
        #     fidfilt_running,
        #     (valid_night["field_id"].values, valid_night["filt_idx"].values),
        #     1,
        # )


    @classmethod
    def _get_required_columns(cls):
        """Allows subclasses to override expected columns based on input data shape."""
        return {"ra", "dec", "filter", "count", "exptime"}

    @classmethod
    def load_from_dir(
        cls,
        data_dir: Path,
        overrides: Optional[Dict[LookupKeys, str]] = None,
        include_historic: bool = False
    ) -> "LookupTables":
        """Load base lookups from a directory."""
        overrides = overrides or {}
        data_dir = Path(data_dir).resolve()

        kwargs = cls._load_base_kwargs(data_dir, overrides)
        if include_historic:
            raise NotImplementedError("On the todo list...")
        return cls(**kwargs)

    def write_to_disk(self, outdir: Optional[Path] = None) -> None:
        """Persist non-derived state. Marginals are recomputed on load."""
        outdir = Path(outdir if outdir is not None else self.dir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Round-trip the index by resetting it as a column before save.
        self.fields.reset_index().to_json(outdir / LookupKeys.FIELDS.value, orient='records')

        # TARGET COUNTS
        with open(outdir / LookupKeys.TARGET_FIDFILT_COUNTS.value, "wb") as f:
            pickle.dump(self.target_fidfilt_counts, f)
        # EXPOSURE TIME PER (FIELD, FILTER) PAIR
        with open(outdir / LookupKeys.FIDFILT_EXPTIME.value, "wb") as f:
            pickle.dump(self.fidfilt_exptime, f)
        # TOTAL SURVEY OBSERVING TIME IN SECONDS
        # with open(outdir / LookupKeys.TOTAL_OT_SECONDS.value, "w") as f:
            # f.write(f"{self.total_ot_sec}")

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def _get_merge_base_kwargs(self, new_lookups: "LookupTables", new_dir: Optional[Path] = None) -> dict:
        """Helper to compute merged base attributes."""
        offset = (self.fields.index.max() + 1) if len(self.fields) else 0

        new_fields = new_lookups.fields.copy()
        new_fields.index = new_fields.index + offset
        new_fields.index.name = self.fields.index.name

        merged_fields = pd.concat([self.fields, new_fields])
        self._validate_field_names(merged_fields)
        merged_fidfilt_counts = np.vstack([
            self.target_fidfilt_counts,
            new_lookups.target_fidfilt_counts,
        ])
        merged_fidfilt_exptime = np.vstack([
            self.fidfilt_exptime,
            new_lookups.fidfilt_exptime,
        ])

        return {
            "fields": merged_fields,
            "target_fidfilt_counts": merged_fidfilt_counts,
            "fidfilt_exptime": merged_fidfilt_exptime,
            "dir": new_dir if new_dir is not None else self.dir,
        }

    def merge(
        self,
        new_lookups: "LookupTables",
        new_dir: Optional[Path] = None,
    ) -> "LookupTables":
        """Append new field targets to existing lookup table, returning a new LookupTables."""
        kwargs = self._get_merge_base_kwargs(new_lookups, new_dir)
        return LookupTables(**kwargs)

    # ------------------------------------------------------------------
    # Field Lookup Construction Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_contiguous_field_ids(df):
        unique_fields = df[['ra', 'dec']].drop_duplicates().copy()
        unique_fields['field_id'] = range(len(unique_fields))
        df = df.merge(unique_fields, on=['ra', 'dec'], how='left')
        return df

    @staticmethod
    def _build_target_count_lookup(df, survey: SurveyProfile = DES):
        filter_order = list(survey.filters)
        pivot_df = df.pivot(index='field_id', columns='filter', values='count')
        pivot_df = pivot_df.fillna(0).astype(int)
        counts_matrix = pivot_df.reindex(columns=filter_order, fill_value=0).to_numpy()
        return counts_matrix

    @staticmethod
    def _build_exptime_lookup(df, survey: SurveyProfile = DES):
        filter_order = list(survey.filters)
        pivot_df = df.pivot(index='field_id', columns='filter', values='exptime')
        pivot_df = pivot_df.fillna(0).astype(int)
        exptime_matrix = pivot_df.reindex(columns=filter_order, fill_value=0).to_numpy()
        return exptime_matrix

    @staticmethod
    def _validate_field_ids(df):
        """
        Validates that field_id cleanly maps 1:1 with unique (ra, dec) pairs.
        Raises ValueError if duplicate mappings or inconsistencies are found.
        """
        if 'field_id' not in df.columns:
            return False  # No field_id column; caller should assign new contiguous ids.

        # Check 1: Does a single (ra, dec) pair map to more than one field_id?
        coord_groups = df.groupby(['ra', 'dec'])['field_id'].nunique()
        invalid_coords = coord_groups[coord_groups > 1]

        if not invalid_coords.empty:
            example_coord = invalid_coords.index[0]
            raise ValueError(
                f"Data Check Failed: The coordinate pair {example_coord} "
                f"is assigned to multiple different field_ids."
            )

        # Check 2: Does a single field_id map to more than one (ra, dec) pair?
        id_groups = df.groupby('field_id')[['ra', 'dec']].nunique()
        invalid_ids = id_groups[(id_groups['ra'] > 1) | (id_groups['dec'] > 1)]

        if not invalid_ids.empty:
            example_id = invalid_ids.index[0]
            raise ValueError(
                f"Data Check Failed: field_id '{example_id}' is assigned to "
                f"multiple distinct (ra, dec) coordinate pairs."
            )

        print("Data Check Passed: field_id uniquely maps 1:1 to all (ra, dec) pairs.")
        return True

    @staticmethod
    def _validate_field_names(fields_df: pd.DataFrame, tolerance_deg: float = 1e-2) -> None:
        """Raise if any field name identifies more than one field within a propid.

        `field` is a display label, not an identifier; the same name may repeat
        across distinct propids and is kept as-is. A name reused for two
        field_ids in the same propid (or table-wide when no `propid` column is
        present) is a data error. tolerance_deg only shapes the message:
        redundant duplicate vs distinct positions.
        """
        dup_mask = fields_df["field"].duplicated(keep=False)
        if not dup_mask.any():
            return

        # Without propid we cannot tell programs apart, so treat the whole
        # table as a single program: any same-name field_ids clash.
        has_propid = "propid" in fields_df.columns
        tol_rad = tolerance_deg * units.deg
        for name, grp in fields_df[dup_mask].groupby("field", sort=False):
            if has_propid:
                propid_counts = grp["propid"].value_counts()
                clashing = propid_counts[propid_counts > 1]
                if clashing.empty:
                    continue  # distinct propids only: allowed, names kept
                propid = clashing.index[0]
                same = grp[grp["propid"] == propid]
            else:
                propid = None
                same = grp

            ra0, dec0 = same.iloc[0][["ra", "dec"]]
            within = bool(
                ((same["ra"] - ra0).abs() <= tol_rad).all()
                and ((same["dec"] - dec0).abs() <= tol_rad).all()
            )
            coords = list(zip(same["ra"].tolist(), same["dec"].tolist()))
            detail = (
                f"their coordinates agree within {tolerance_deg} deg, which "
                f"looks like a redundant duplicate; remove the extra entry"
                if within else
                f"their coordinates differ by more than {tolerance_deg} deg, "
                f"so they are distinct positions; give them distinct names"
            )
            scope = (
                f"the same propid {propid!r}" if has_propid
                else "one program (no propid column to disambiguate)"
            )
            raise ValueError(
                f"Field name {name!r} is assigned to {len(same)} fields under "
                f"{scope} (field_ids {list(same.index)}, (ra, dec) {coords}); "
                f"{detail}."
            )

    # ------------------------------------------------------------------
    # Construction from raw fields file
    # ------------------------------------------------------------------

    @classmethod
    def build_lookups_from_fields(
        cls,
        fields_df: Optional[pd.DataFrame] = None,
        fields_path: Optional[str | Path] = None,
        outdir: Optional[Path] = None,
        write_to_disk: bool = False,
        survey: SurveyProfile = DES,
    ) -> "LookupTables":
        """Build a LookupTables from a JSON fields file."""
        # Data and arg checks -------------------------------------------------
        if write_to_disk and outdir is None:
            raise ValueError("Must specify `outdir` if `write_to_disk` is True")

        if fields_df is None and fields_path is None:
            raise ValueError("Must specify either `fields_df` or `fields_path`")

        if fields_df is not None:
            df = fields_df.copy()
        else:
            fields_path = Path(fields_path)
            df = pd.read_json(fields_path)

        # Ensure all columns are lowercase ---------------------------------
        df.columns = df.columns.str.lower()

        # Ensure required columns are present ------------------------------
        required = cls._get_required_columns()
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Check RA/Dec values are in radians --------------------------------
        if (df["ra"] > 2 * np.pi).any() or (df["dec"].abs() > np.pi / 2).any():
            raise ValueError(
                "Data Check Failed: At least one RA/Dec values degrees exceed 2pi); "
                "please convert to radians before building lookups."
            )

        # Resolve name column from common aliases ---------------------------
        if "field_name" in df.columns:
            df = df.rename(columns={"field_name": "field"})
        elif "fieldname" in df.columns:
            df = df.rename(columns={"fieldname": "field"})
        else:
            df["field"] = (
                "field_"
                + df.groupby(["ra", "dec"], sort=False).ngroup().astype(str)
            )

        # Assign field_id
        has_consistent_fid = cls._validate_field_ids(df)
        if not has_consistent_fid:
            df = cls._get_contiguous_field_ids(df)

        # Filter idx
        df["filter_idx"] = df["filter"].map(survey.filter2idx).fillna(-1).astype(int)
        if (df["filter_idx"] == -1).any():
            bad = df.loc[df["filter_idx"] == -1, "filter"].unique()
            raise ValueError(f"Unknown filter(s): {list(bad)}")

        # Validate per-field columns
        per_field_cols = ["field", "ra", "dec"]
        if "propid" in df.columns:
            per_field_cols = per_field_cols + ["propid"]
        if "priority" in df.columns:
            per_field_cols = per_field_cols + ["priority"]
        for col in per_field_cols:
            counts = df.groupby("field_id")[col].nunique()
            if (counts > 1).any():
                bad = list(counts[counts > 1].index)
                raise ValueError(
                    f"Column {col!r} varies within a field_id; cannot "
                    f"deduplicate. Affected field_ids: {bad}"
                )

        # Construct lookups ---------------------------------------
        field_cols = ['field_id', 'field', 'ra', 'dec']
        if "propid" in df.columns:
            field_cols = field_cols + ['propid']
        if "priority" in df.columns:
            field_cols = field_cols + ['priority']
        fields_lookup = (
            df[field_cols]
            .drop_duplicates()
            .sort_values(by='field_id')
            .set_index('field_id')
        )
        cls._validate_field_names(fields_lookup)

        target_fidfilt_counts = cls._build_target_count_lookup(df, survey)
        fidfilt_exptime = cls._build_exptime_lookup(df, survey)

        if outdir is not None:
            resolved_dir = Path(outdir).resolve()
        elif fields_path is not None:
            resolved_dir = fields_path.parent.resolve()
        else:
            resolved_dir = Path.cwd().resolve()

        lookups = cls(
            fields=fields_lookup,
            target_fidfilt_counts=target_fidfilt_counts,
            fidfilt_exptime=fidfilt_exptime,
            dir=resolved_dir,
            survey=survey,
        )

        if write_to_disk:
            lookups.write_to_disk(Path(outdir))

        return lookups

    # ------------------------------------------------------------------
    # Validation + derived attributes
    # ------------------------------------------------------------------

    def __post_init__(self):
        # Coerce arrays to canonical dtype/layout
        object.__setattr__(
            self, "target_fidfilt_counts",
            np.ascontiguousarray(self.target_fidfilt_counts, dtype=np.int64),
        )
        object.__setattr__(
            self, "fidfilt_exptime",
            np.ascontiguousarray(self.fidfilt_exptime, dtype=np.float64),
        )

        # Validate fields index is 0..N-1 contiguous
        expected_idx = pd.RangeIndex(len(self.fields))
        if not self.fields.index.equals(expected_idx):
            raise ValueError(
                "`fields` index must be 0..N-1 contiguous; got "
                f"min={self.fields.index.min()}, "
                f"max={self.fields.index.max()}, "
                f"len={len(self.fields)}, "
                f"unique={self.fields.index.nunique()}"
            )

        # Validate matrix shapes
        nfields, nfilters = self.target_fidfilt_counts.shape
        if nfields != len(self.fields):
            raise ValueError(
                f"target_fidfilt_counts has {nfields} rows but `fields` has "
                f"{len(self.fields)}"
            )
        if nfilters != self.survey.num_filters:
            raise ValueError(
                f"target_fidfilt_counts has {nfilters} filter columns but "
                f"survey '{self.survey.key}' defines {self.survey.num_filters}"
            )
        if self.fidfilt_exptime.shape != self.target_fidfilt_counts.shape:
            raise ValueError(
                f"fidfilt_exptime shape {self.fidfilt_exptime.shape} does not "
                f"match target_fidfilt_counts shape "
                f"{self.target_fidfilt_counts.shape}"
            )

        # Validate per-field columns
        required_cols = {"field", "ra", "dec"}
        missing = required_cols - set(self.fields.columns)
        if missing:
            raise ValueError(f"`fields` is missing required columns: {missing}")

        # Compute base marginals
        object.__setattr__(
            self, "target_fid_counts", self.target_fidfilt_counts.sum(axis=1)
        )
        object.__setattr__(
            self, "target_filt_counts", self.target_fidfilt_counts.sum(axis=0)
        )


@dataclass(frozen=True)
class TrainLookupTables(LookupTables):
    """Container for telescope/survey metadata utilized during training.

    Includes historical visit dicts which are snapshots taken at the START
    of each observing night, derived from the FULL survey history.
    """
    night2fid_visit_hist: Optional[dict] = None
    night2fidfilt_visit_hist: Optional[dict] = None
    night2fid_last_visit_ts: Optional[dict] = None
    night2fidfilt_last_visit_ts: Optional[dict] = None
    night2fid_last_visit_ot: Optional[dict] = None
    night2fidfilt_last_visit_ot: Optional[dict] = None
    night2ot_clock_seconds: Optional[dict] = None
    # Which exposures counted toward targets and visit history
    acceptance: AcceptanceRule = AcceptanceRule.UNIFORM

    # Derived marginals
    night2idx: Optional[dict] = None
    total_nights: Optional[int] = None

    @classmethod
    def _get_required_columns(cls):
        # We calculate targets from teff, so 'count' is no longer required in the raw data
        return {"ra", "dec", "filter", "exptime", "teff"}

    @classmethod
    def load_from_dir(
        cls,
        data_dir: Path,
        overrides: Optional[Dict[LookupKeys, str]] = None,
        acceptance: Optional[AcceptanceRule | str] = None,
    ) -> "TrainLookupTables":
        """Load lookups from a directory, including historic context.

        Parameters
        ----------
        data_dir : Path
            Lookups folder.
        overrides : dict or None
            File name overrides per lookup key.
        acceptance : AcceptanceRule, str, or None
            Rule the caller expects the lookups to be built with; None skips the check.

        Returns
        -------
        TrainLookupTables
            The loaded lookups.
        """
        overrides = overrides or {}
        data_dir = Path(data_dir).resolve()

        def get_path(key):
            return data_dir / overrides.get(key, key.value)

        accept_rule = cls.load_acceptance(data_dir)
        if acceptance is not None:
            accept_rule.require(acceptance, data_dir)

        # 1. Start with base kwargs
        kwargs = cls._load_base_kwargs(data_dir, overrides)
        kwargs["acceptance"] = accept_rule

        # 2. Add historical tables
        with open(get_path(LookupKeys.NIGHT2FID_VISIT_HIST), "rb") as f:
            kwargs["night2fid_visit_hist"] = pickle.load(f)
        with open(get_path(LookupKeys.NIGHT2FIDFILT_VISIT_HIST), "rb") as f:
            kwargs["night2fidfilt_visit_hist"] = pickle.load(f)
        with open(get_path(LookupKeys.NIGHT2OT_CLOCK_SECONDS), "rb") as f:
            kwargs["night2ot_clock_seconds"] = pickle.load(f)

        # 3. Add last-visit timestamps
        fid_lv_path = get_path(LookupKeys.NIGHT2FID_LAST_VISIT_TS)
        ff_lv_path = get_path(LookupKeys.NIGHT2FIDFILT_LAST_VISIT_TS)

        if fid_lv_path.exists():
            with open(fid_lv_path, "rb") as f:
                kwargs["night2fid_last_visit_ts"] = pickle.load(f)
        else:
            logger.warning(
                f"{fid_lv_path.name} not found in {data_dir}; "
                f"t_since_last_visit will start from sentinel for every "
                f"field. Rebuild lookups to enable staleness seeding."
            )

        if ff_lv_path.exists():
            with open(ff_lv_path, "rb") as f:
                kwargs["night2fidfilt_last_visit_ts"] = pickle.load(f)
        else:
            logger.warning(
                f"{ff_lv_path.name} not found in {data_dir}; "
                f"per-filter t_since_last_visit will start from sentinel."
            )

        # 4. Add last-visit dicts in ot
        fid_lv_ot_path = get_path(LookupKeys.NIGHT2FID_LAST_VISIT_OT)
        ff_lv_ot_path = get_path(LookupKeys.NIGHT2FIDFILT_LAST_VISIT_OT)

        if fid_lv_ot_path.exists():
            with open(fid_lv_ot_path, "rb") as f:
                kwargs["night2fid_last_visit_ot"] = pickle.load(f)
        else:
            logger.warning(
                f"{fid_lv_ot_path.name} not found in {data_dir}; "
                f"t_since_last_visit will start from sentinel for every "
                f"field. Rebuild lookups to enable staleness seeding."
            )

        if ff_lv_ot_path.exists():
            with open(ff_lv_ot_path, "rb") as f:
                kwargs["night2fidfilt_last_visit_ot"] = pickle.load(f)
        else:
            logger.warning(
                f"{ff_lv_ot_path.name} not found in {data_dir}; "
                f"per-filter t_since_last_visit_ot will start from sentinel."
            )

        return cls(**kwargs)

    @staticmethod
    def load_acceptance(data_dir: Path) -> AcceptanceRule:
        """Acceptance rule a lookups folder was built with."""
        record = json.loads((Path(data_dir) / LookupKeys.ACCEPTANCE.value).read_text())
        return AcceptanceRule(record["acceptance"])

    def write_to_disk(self, outdir: Optional[Path] = None) -> None:
        """Persist training state alongside base lookups."""
        super().write_to_disk(outdir)

        outdir = Path(outdir if outdir is not None else self.dir)

        # ACCEPTANCE RULE
        (outdir / LookupKeys.ACCEPTANCE.value).write_text(json.dumps({"acceptance": self.acceptance.value}))

        # VISIT HISTORY
        if self.night2fid_visit_hist is not None:
            with open(outdir / LookupKeys.NIGHT2FID_VISIT_HIST.value, "wb") as f:
                pickle.dump(self.night2fid_visit_hist, f)
        if self.night2fidfilt_visit_hist is not None:
            with open(outdir / LookupKeys.NIGHT2FIDFILT_VISIT_HIST.value, "wb") as f:
                pickle.dump(self.night2fidfilt_visit_hist, f)

        # LAST VISIT TIMESTAMP
        if self.night2fid_last_visit_ts is not None:
            with open(outdir / LookupKeys.NIGHT2FID_LAST_VISIT_TS.value, "wb") as f:
                pickle.dump(self.night2fid_last_visit_ts, f)
        if self.night2fidfilt_last_visit_ts is not None:
            with open(outdir / LookupKeys.NIGHT2FIDFILT_LAST_VISIT_TS.value, "wb") as f:
                pickle.dump(self.night2fidfilt_last_visit_ts, f)

        # LAST VISIT TIME IN UNITS OBSERVING TIME
        if self.night2fid_last_visit_ot is not None:
            with open(outdir / LookupKeys.NIGHT2FID_LAST_VISIT_OT.value, "wb") as f:
                pickle.dump(self.night2fid_last_visit_ot, f)
        if self.night2fidfilt_last_visit_ot is not None:
            with open(outdir / LookupKeys.NIGHT2FIDFILT_LAST_VISIT_OT.value, "wb") as f:
                pickle.dump(self.night2fidfilt_last_visit_ot, f)

        # TOTAL OBSERVABLE SECONDS IN SURVEY
        if self.night2ot_clock_seconds is not None:
            with open(outdir / LookupKeys.NIGHT2OT_CLOCK_SECONDS.value, "wb") as f:
                pickle.dump(self.night2ot_clock_seconds, f)

    def merge(
        self,
        new_lookups: "TrainLookupTables",
        new_dir: Optional[Path] = None,
    ) -> "TrainLookupTables":
        raise NotImplementedError("Merging of TrainLookupTables is not yet implemented.")
        kwargs = self._get_merge_base_kwargs(new_lookups, new_dir)

        num_new_fields = len(new_lookups.fields)
        nfilters = self.survey.num_filters

        def _pad_1d(hist_dict, pad_val=0):
            if hist_dict is None: return None
            return {k: np.pad(v, (0, num_new_fields), constant_values=pad_val) for k, v in hist_dict.items()}

        def _pad_2d(hist_dict, pad_val=0):
            if hist_dict is None: return None
            return {k: np.pad(v, ((0, num_new_fields), (0, 0)), constant_values=pad_val) for k, v in hist_dict.items()}

        kwargs.update({
            "night2fid_visit_hist": _pad_1d(self.night2fid_visit_hist, pad_val=0),
            "night2fidfilt_visit_hist": _pad_2d(self.night2fidfilt_visit_hist, pad_val=0),
            "night2fid_last_visit_ts": _pad_1d(self.night2fid_last_visit_ts, pad_val=np.nan),
            "night2fidfilt_last_visit_ts": _pad_2d(self.night2fidfilt_last_visit_ts, pad_val=np.nan),
            "night2fid_last_visit_ot": _pad_1d(self.night2fid_last_visit_ot, pad_val=np.nan),
            "night2fidfilt_last_visit_ot": _pad_2d(self.night2fidfilt_last_visit_ot, pad_val=np.nan),
            "night2ot_clock_seconds": self.night2ot_clock_seconds,
            "acceptance": self.acceptance,
        })
        return TrainLookupTables(**kwargs)


    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "acceptance", AcceptanceRule(self.acceptance))
        nfields, nfilters = self.target_fidfilt_counts.shape
        self._validate_history_shapes(nfields, nfilters)

        # Compute child marginals
        if self.night2fidfilt_visit_hist is not None:
            object.__setattr__(
                self, "night2idx",
                {night: i for i, night in enumerate(self.night2fidfilt_visit_hist.keys())}
            )
            object.__setattr__(
                self, "total_nights", len(self.night2idx)
            )

    def _validate_history_shapes(self, nfields, nfilters):
        """Validates that all historical snapshots match the per-night layout."""
        if self.night2fid_visit_hist is not None:
            for night, arr in self.night2fid_visit_hist.items():
                if arr.shape != (nfields,):
                    raise ValueError(
                        f"night2fid_visit_hist[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields},)"
                    )
        if self.night2fidfilt_visit_hist is not None:
            for night, arr in self.night2fidfilt_visit_hist.items():
                if arr.shape != (nfields, nfilters):
                    raise ValueError(
                        f"night2fidfilt_visit_hist[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields}, {nfilters})"
                    )
        if self.night2fid_last_visit_ts is not None:
            for night, arr in self.night2fid_last_visit_ts.items():
                if arr.shape != (nfields,):
                    raise ValueError(
                        f"night2fid_last_visit[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields},)"
                    )
        if self.night2fidfilt_last_visit_ts is not None:
            for night, arr in self.night2fidfilt_last_visit_ts.items():
                if arr.shape != (nfields, nfilters):
                    raise ValueError(
                        f"night2fidfilt_last_visit_ts[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields}, {nfilters})"
                    )

        def _check_keys(a, a_name, b, b_name):
            if a is not None and b is not None:
                if set(a.keys()) != set(b.keys()):
                    raise ValueError(
                        f"{a_name} and {b_name} have mismatched night keys: "
                        f"symmetric_diff="
                        f"{set(a.keys()).symmetric_difference(set(b.keys()))}"
                    )
        _check_keys(
            self.night2fid_visit_hist, "night2fid_visit_hist",
            self.night2fid_last_visit_ts, "night2fid_last_visit_ts",
        )
        _check_keys(
            self.night2fidfilt_visit_hist, "night2fidfilt_visit_hist",
            self.night2fidfilt_last_visit_ts, "night2fidfilt_last_visit_ts",
        )
