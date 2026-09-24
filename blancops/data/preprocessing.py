import operator
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import timedelta
import re


from blancops.data.features.glob_features import get_night_boundaries
from blancops.io.file_io import (
    _add_night, _add_timestamp, _drop_nan_dts,
    _replace_with_pd_dt, fits_to_df
)
from blancops.math import units

from blancops.configs.paths import workspace
from blancops.configs.constants import FILTER2IDX
from blancops.data.lookup_tables import TrainLookupTables
from blancops.math import units

import logging

from blancops.survey.profiles import DES
logger = logging.getLogger(__name__)


_OP_MAP = {
    '=': operator.eq,
    '==': operator.eq,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '!=': operator.ne
}

# Applies AND based selections given the following criteria:
_DES_SELECTION_CRITERIA = [
    ('propid', '2012B-0001', '=='),
    ('exptime', 100, '<'),
    ('exptime', 40, '>'),
    ('datetime', pd.Timestamp("2013-08-30", tz="UTC"), '>='),
    ('teff', np.nan, '!='),
    ('program', 'supernova', '!='),
    ('program', 'des gw', '!='),
]
# Specific objects to remove from dataset (non-astronomical targets or non DES wide-field survey targets).
# Most (but not all) of these should be taken care of by the 'program' and 'propid' selection criteria.
_DES_UNWANTED_OBJECTS = [
    "guide",
    "pointing",
    "DES vvds",
    "J0",
    "SN",
    "gwh",
    "DESGW",
    "Alhambra",
    "cosmos",
    "COSMOS hex",
    "TMO",
    "LDS",
    "WD0",
    "DES supernova hex",
    "NGC",
    "ec",
    ]



def preprocess_fits(fits_path): # XXX move to preprocessing.py
    df = fits_to_df(fits_path)
    df = df.pipe(_replace_with_pd_dt)\
            .pipe(_drop_nan_dts)\
            .pipe(_add_timestamp)\
            .pipe(_add_night)
    return df


def load_and_process_historic_data(
    fits_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    valid_years=None,
    valid_months=None,
    valid_days=None,
    valid_filters=None,
    selections=_DES_SELECTION_CRITERIA,
    objects_to_remove=_DES_UNWANTED_OBJECTS,
    outlier_cutoff_dist=3.5
    ) -> pd.DataFrame:
    """Loads data from fits file, applies selection criteria, and processes into a clean DataFrame ready for lookup construction and feature engineering.

    Args:
    =====
    fits_path (str | Path): str | Path
        Path to fits file
    df (pd.DataFrame | None): pd.DataFrame | None
        DataFrame to process
    selections (dict): dict
        Selection criteria for the dataframe. Currently only supports equal, greater than, less than.
    """
    assert fits_path or df, "Provide either fits_path or df, not both."
    if df is None:
        df = preprocess_fits(fits_path)


    df = df \
        .pipe(_apply_selection_criteria, selections=selections) \
        .pipe(_remove_outliers, col='object', cutoff_dist=outlier_cutoff_dist) \
        .pipe(_convert_df_to_radians) \
        .pipe(_filter_observing_timeframe, start_date=start_date, end_date=end_date, valid_years=valid_years, valid_months=valid_months, valid_days=valid_days, valid_filters=valid_filters) \
        .pipe(_remove_unwanted_objects, objects_to_remove=objects_to_remove) \
        .pipe(_add_essential_cols) \
        .pipe(_add_field_col) \

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    return df

def _apply_selection_criteria(df, selections: list):
    sel_mask = np.ones(len(df), dtype=bool)

    for crit_key, val, op_str in selections:
        try:
            op_func = _OP_MAP[op_str]
        except KeyError:
            raise ValueError(f"Operator '{op_str}' is not supported.")

        # Special handling for NaN comparisons
        if pd.isna(val):
            if op_str == '!=':
                sel_mask &= df[crit_key].notna()
            elif op_str == '==':
                sel_mask &= df[crit_key].isna()
        else:
            sel_mask &= op_func(df[crit_key], val)

    return df[sel_mask].copy()

def _convert_df_to_radians(df: pd.DataFrame, key_list: list | None = None) -> pd.DataFrame:
    """Converts specified columns in the dataframe from degrees to radians.

    If key_list is None, it defaults to converting 'ra', 'dec', 'az', 'zd', and 'ha'.
    """
    if key_list is None:
        key_list = []

    key_list.extend(['ra', 'dec', 'az', 'zd', 'ha'])
    key_list = list(set(key_list))
    if not all(key in df.columns for key in key_list):
        raise ValueError(f"Not all keys in key_list are present in the dataframe. Missing keys: {[key for key in key_list if key not in df.columns]}")
    df.loc[:, key_list] *= units.deg
    return df

def _add_essential_cols(df):
    df['el'] = np.pi/2 - df['zd'].values
    return df

def _remove_unwanted_objects(df, objects_to_remove) -> pd.DataFrame:
    """Removes objects that have specific substrings in their object name"""
    pattern = '|'.join(re.escape(obj) for obj in objects_to_remove)
    mask = ~df['object'].str.contains(pattern, case=False, na=False, regex=True) & (df['object'] != '')

    # Filter the DataFrame
    df = df[mask]
    return df

def _remove_outliers(df, col='object', cutoff_dist=3.5) -> pd.DataFrame:
    """Remove rows if they are outside of a certain cutoff from the object's median RA/Dec.

    Args
    ----
    df (pd.DataFrame): The dataframe with object names and RA/Dec positions.
    cutoff_dist (float): The cutoff distance in degrees.

    Returns
    -------
    cleaned_df (pd.DataFrame): The cleaned dataframe with relabelled objects removed.
    """
    indices_to_drop = []
    for obj_name, g in df.groupby(col):
        median_ra = _get_median_ra(g.ra)
        median_dec = g.dec.median()

        # Calculate true on-sky distance from the robust center
        distances = get_haversine_dist(median_ra, median_dec, g.ra.values, g.dec.values)

        # Mask and re-label
        mask_outlier = distances > cutoff_dist

        if np.count_nonzero(mask_outlier) > 0:
            outlier_indices = g.index[mask_outlier].values
            indices_to_drop.extend(outlier_indices)
    cleaned_df = df.drop(index=indices_to_drop)
    return cleaned_df

def _filter_observing_timeframe(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    valid_years: list | int | None = None,
    valid_months: list | int | None = None,
    valid_days: list | int | None = None,
    valid_filters: list | str | None = None
) -> pd.DataFrame:
    """
    Filters dataframe by a continuous date range and/or discrete allowed intervals.
    """
    # Helper to safely handle scalars or lists
    def _to_list(val):
        if val is None:
            return None
        return [val] if isinstance(val, (int, str)) else list(val)

    mask = pd.Series(True, index=df.index)
    night_dt = pd.to_datetime(df['night'])

    # 1. Continuous Range Bounds
    if start_date is not None:
        mask &= night_dt >= pd.to_datetime(start_date)
        if not mask.any():
            raise ValueError(f"No data found after start_date: {start_date}")

    if end_date is not None:
        mask &= night_dt <= pd.to_datetime(end_date)
        if not mask.any():
            raise ValueError(f"No data found before end_date: {end_date} (within prior cuts)")

    # 2. Discrete Interval Inclusions
    valid_years = _to_list(valid_years)
    if valid_years:
        mask &= night_dt.dt.year.isin(valid_years)
        if not mask.any():
            raise ValueError(f"No data matches years: {valid_years} (within prior cuts)")

    valid_months = _to_list(valid_months)
    if valid_months:
        mask &= night_dt.dt.month.isin(valid_months)
        if not mask.any():
            raise ValueError(f"No data matches months: {valid_months} (within prior cuts)")

    valid_days = _to_list(valid_days)
    if valid_days:
        mask &= night_dt.dt.day.isin(valid_days)
        if not mask.any():
            raise ValueError(f"No data matches days: {valid_days} (within prior cuts)")

    valid_filters = _to_list(valid_filters)
    if valid_filters:
        mask &= df['filter'].isin(valid_filters)
        if not mask.any():
            raise ValueError(f"No data matches filters: {valid_filters} (within prior cuts)")

    return df[mask].copy()


def _get_median_ra(ra_arr):
    """Returns the median RA for a given array of RA values.
    Useful for edge aware centering.

    Args
    ----
    ra_arr (np.ndarray): Array of RA values in degrees.
    """
    ra_rad = np.radians(ra_arr)

    mean_x = np.mean(np.cos(ra_rad))
    mean_y = np.mean(np.sin(ra_rad))

    anchor_ra = np.degrees(np.arctan2(mean_y, mean_x)) % 360
    shifted_ra = (ra_arr - anchor_ra + 180) % 360 - 180

    return (np.median(shifted_ra) + anchor_ra) % 360

def _get_mean_ra(ra_arr):
    """Returns the mean RA for a given array of RA values.
    Useful for edge aware centering.

    Args
    ----
    ra_arr (np.ndarray): Array of RA values in degrees.
    """
    ra_rad = np.radians(ra_arr)

    mean_x = np.mean(np.cos(ra_rad))
    mean_y = np.mean(np.sin(ra_rad))

    anchor_ra = np.degrees(np.arctan2(mean_y, mean_x)) % 360
    shifted_ra = (ra_arr - anchor_ra + 180) % 360 - 180

    return (np.mean(shifted_ra) + anchor_ra) % 360

def get_haversine_dist(ra_center, dec_center, ra_array, dec_array):
    """Calculates great-circle distance between a center point and an array of points."""
    ra1, dec1 = np.radians(ra_center), np.radians(dec_center)
    ra2, dec2 = np.radians(ra_array), np.radians(dec_array)

    d_ra = ra2 - ra1
    d_dec = dec2 - dec1

    a = np.sin(d_dec / 2.0)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(np.sqrt(a)))

def _add_field_col(df):
    """Adds a 'field' column to the dataframe by stripping tiling information from the 'object' column."""
    assert all('tiling' in obj_name for obj_name in df['object'].unique()), "All object names must contain 'tiling' for this function to work as intended."
    df['field'] = df['object'].str.replace(r'\s*tiling\s+\d+', '', regex=True)
    return df


def build_DES_lookups(fits_path=None, outdir=None):
    fits_path = Path(fits_path or workspace().des_fits).resolve()
    outdir = Path(outdir or workspace().des_data).resolve()

    df = load_and_process_historic_data(fits_path=fits_path)
    if len(df) == 0: # Fixed the logical bug here: len(df) == 0 means no obs found
        logger.warning("No observations found for the specified year/month/day/filter selections.")
        raise ValueError

    # Require field_id is 0..N-1 contiguous
    field2idx = {obj_name: idx for idx, obj_name in enumerate(sorted(df['field'].unique()))}
    df['field_id'] = df['field'].map(field2idx)
    df["filt_idx"] = df["filter"].map(FILTER2IDX)

    num_fields = df["field_id"].nunique()
    nfilters = len(FILTER2IDX)

    # Resolve survey-profile constants once for use below and in the night loop.
    sun_el_limit = DES.sun_el_limit
    valid_teff_threshold = DES.valid_teff_threshold

    # Quality threshold — only targets and per-night history derive
    # from this set, so completion checks and seeded state agree.
    valid_df = df[df["teff"] > valid_teff_threshold].copy()
    if len(valid_df) == 0:
        raise ValueError(
            f"No observations with teff > {valid_teff_threshold} in "
            f"{fits_path}; check input data quality."
        )

    # ---------- Per-field DataFrame ----------
    # One row per field_id. Aggregate over the FULL df (not teff-
    # filtered) so even fields with no valid observations still
    # appear; they'll have a target row of zeros and be excluded from
    # masks. Since field_id was factorized from `object`, all rows in
    # a group share the same name (taking iloc[0] is unambiguous).
    fields_rows = []
    for field_id, g in df.groupby("field_id"):
        fields_rows.append({
            "field_id": int(field_id),
            "field": g["field"].iloc[0],
            "ra": float(_get_mean_ra(g["ra"].values / units.deg) * units.deg),
            "dec": float(g["dec"].mean()),
        })
    fields = pd.DataFrame(fields_rows).set_index("field_id").sort_index()
    fields.index.name = "field_id"
    logger.info(" [+] Constructed Fields Lookup")

    # ---------- Per-(field, filter) matrices ----------
    # target_fidfilt_counts: total VALID observations per (field, filter).
    target_fidfilt_counts = np.zeros((num_fields, nfilters), dtype=np.int32)
    np.add.at(
        target_fidfilt_counts,
        (valid_df["field_id"].values, valid_df["filt_idx"].values),
        1,
    )
    logger.info(" [+] Constructed Target Counts Lookup")

    # fidfilt_exptime: most-common exptime per (field, filter).
    fidfilt_exptime = (
        df.pivot_table(
            index="field_id", columns="filt_idx", values="exptime",
            aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else 0,
        )
        .reindex(index=range(num_fields), columns=range(nfilters), fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    logger.info(" [+] Constructed Exposure Time Lookup")


    # ---------- Per-night visit history & last-visit timestamps ----------
    # Running counts and most-recent-visit timestamps, snapshotted at the
    # START of each night. Iterate over ALL nights in df (not just nights
    # with valid observations) so every observed night has a seedable
    # state, but only VALID observations contribute to the running totals
    # — keeping history consistent with the target-completion semantics.
    #
    # last_visit uses NaN as "no recorded visit yet"; we update with
    # `np.fmax` so existing recorded values win over NaN as new
    # observations come in. (np.maximum would propagate NaN.)
    field_running = np.zeros(num_fields, dtype=np.int32)
    fidfilt_running = np.zeros((num_fields, nfilters), dtype=np.int32)
    field_last_visit_ts = np.full(num_fields, np.nan, dtype=np.float64)
    fidfilt_last_visit_ts = np.full((num_fields, nfilters), np.nan, dtype=np.float64)
    field_last_visit_ot = np.full(num_fields, np.nan, dtype=np.float64)
    fidfilt_last_visit_ot = np.full((num_fields, nfilters), np.nan, dtype=np.float64)

    night2fid_visit_hist = {}
    night2fidfilt_visit_hist = {}
    night2fid_last_visit_ts = {}
    night2fidfilt_last_visit_ts = {}
    night2fid_last_visit_ot = {}
    night2fidfilt_last_visit_ot = {}
    night2ot_clock_seconds = {}     # night -> OT(sunset_n)

    cum_ot = 0.0

    for i, (night, night_df) in enumerate(df.groupby("night")):
        sunset_ts, sunrise_ts = get_night_boundaries(night, sun_el_limit=sun_el_limit)
        night2ot_clock_seconds[night] = cum_ot
        night_dur = sunrise_ts - sunset_ts
        # night2idx[night] = i

        # Snapshot start-of-night state BEFORE adding this night's
        # contributions. Matches existing visit_hist semantics so
        # downstream consumers can pair (visit_hist[n], last_visit[n])
        # safely.
        night2fid_visit_hist[night] = field_running.copy()
        night2fidfilt_visit_hist[night] = fidfilt_running.copy()
        night2fid_last_visit_ts[night] = field_last_visit_ts.copy()
        night2fidfilt_last_visit_ts[night] = fidfilt_last_visit_ts.copy()
        night2fid_last_visit_ot[night] = field_last_visit_ot.copy()
        night2fidfilt_last_visit_ot[night] = fidfilt_last_visit_ot.copy()


        valid_night = night_df[night_df["teff"] > valid_teff_threshold]
        if len(valid_night):
            # Visit counts
            field_running += np.bincount(
                valid_night["field_id"].values, minlength=num_fields
            )
            np.add.at(
                fidfilt_running,
                (valid_night["field_id"].values, valid_night["filt_idx"].values),
                1,
            )

            # Last-visit timestamps: per (field) and per (field, filter),
            # we want the MAXIMUM timestamp across this night's valid
            # observations for that key.
            #
            # Per-field: groupby max is O(n log n) but n is small per
            # night; cleaner than np.maximum.at + temp array.
            fid_max = valid_night.groupby("field_id")["timestamp"].max()
            fid_ids = fid_max.index.to_numpy()
            fid_ts = fid_max.to_numpy(dtype=np.float64)
            # np.fmax: NaN-aware max — incoming value wins over existing NaN.
            field_last_visit_ts[fid_ids] = np.fmax(
                field_last_visit_ts[fid_ids], fid_ts
            )

            # Per (field, filter): same idea, 2-D index.
            ff_max = valid_night.groupby(["field_id", "filt_idx"])["timestamp"].max()
            if len(ff_max):
                ff_keys = np.array(ff_max.index.tolist(), dtype=np.int64)
                ff_ts = ff_max.to_numpy(dtype=np.float64)
                rows, cols = ff_keys[:, 0], ff_keys[:, 1]
                fidfilt_last_visit_ts[rows, cols] = np.fmax(
                    fidfilt_last_visit_ts[rows, cols], ff_ts
                )

            # Last-visit time: in unit of observing time seconds
            #       is per (field) and per (field, filter),
            valid_night = valid_night.assign(
                ot=cum_ot + (valid_night["timestamp"] - sunset_ts)
            )
            fid_max = valid_night.groupby("field_id")["ot"].max()
            fid_ids = fid_max.index.to_numpy()
            fid_ot = fid_max.to_numpy(dtype=np.float64)
            # np.fmax: NaN-aware max — incoming value wins over existing NaN.
            field_last_visit_ot[fid_ids] = np.fmax(
                field_last_visit_ot[fid_ids], fid_ot
            )

            # Per (field, filter): same idea, 2-D index.
            ff_max = valid_night.groupby(["field_id", "filt_idx"])["ot"].max()
            if len(ff_max):
                ff_keys = np.array(ff_max.index.tolist(), dtype=np.int64)
                ff_ot = ff_max.to_numpy(dtype=np.float64)
                rows, cols = ff_keys[:, 0], ff_keys[:, 1]
                fidfilt_last_visit_ot[rows, cols] = np.fmax(
                    fidfilt_last_visit_ot[rows, cols], ff_ot
                )

        cum_ot += night_dur

    # total_observing_seconds = cum_ot

    logger.info(" [+] Constructed start-of-night 'Snapshots' Lookup (required for history-based feature construction and per-night rollout) ")

    # ---------- Construct LookupTable and save to disk ----------
    lookups = TrainLookupTables(
        fields=fields,
        target_fidfilt_counts=target_fidfilt_counts,
        fidfilt_exptime=fidfilt_exptime,
        dir=outdir,
        night2fid_visit_hist=night2fid_visit_hist,
        night2fidfilt_visit_hist=night2fidfilt_visit_hist,
        night2fid_last_visit_ts=night2fid_last_visit_ts,
        night2fidfilt_last_visit_ts=night2fidfilt_last_visit_ts,
        night2fid_last_visit_ot=night2fid_last_visit_ot,
        night2fidfilt_last_visit_ot=night2fidfilt_last_visit_ot,
        night2ot_clock_seconds=night2ot_clock_seconds,
        # total_ot_sec=total_observing_seconds,
    )
    lookups.write_to_disk(outdir)
    logger.info(f" [+] Successfully generated all lookup tables in {outdir}")

    return lookups
