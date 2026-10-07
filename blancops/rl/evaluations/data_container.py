"""Data containers that wrangle expert/agent dataframes for evaluation.

The original `DataContainer.eval_method` branch is split into two subclasses,
`SingleStepDataContainer` and `MultiStepDataContainer`, so each method has a
single, explicit code path. Shared logic lives on the base class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from blancops.data.dataset import TransitionDataset
from blancops.data.features.normalizations import StateNormalizer, inverse_cyclical_norm
from blancops.data.lookup_tables import LookupTables
from blancops.data.norm_stats import NormStats
from blancops.ephemerides import ephemerides
from blancops.math import units
from blancops.math.geometry import angular_separation
import logging
logger = logging.getLogger(__name__)

from .helpers import (
    calc_airmass,
    calc_moon_dist,
    calc_moon_phase,
    calc_slew_distance,
    calc_sun_and_moon_pos,
)


# Time-gap thresholds for masking "next states" that aren't actually adjacent.
# Expert: small gaps mean a real cadence break we should drop from comparisons.
EXPERT_MAX_GAP = 5 # minutes
# Agent: larger window because the offline runner may emit longer apparent gaps
# across wait-actions. Kept explicit + named so the discrepancy is visible.
AGENT_MAX_GAP = 60 * 5 # minutes


# Substrings (within underscore-tokenized column names) that imply a radian
# value needing conversion to degrees. Anything ending in _sin/_cos is kept as-is.
_ANGLE_TOKENS = frozenset({
    'ra', 'dec', 'az', 'el',
    'slew', 'distance', 'separation',
})
_CIRC_SUFFIXES = ('_sin', '_cos')


def _is_angle_column(key: str) -> bool:
    if key.endswith(_CIRC_SUFFIXES):
        return False
    return bool(_ANGLE_TOKENS.intersection(key.split('_')))


def _convert_df_to_deg(df: pd.DataFrame) -> pd.DataFrame:
    """Convert in-place any column whose name implies a radian angle to degrees."""
    for key in df.columns:
        if _is_angle_column(key):
            df[key] = df[key] / units.deg
    return df


class DataContainer(ABC):
    """Base class. Subclasses populate `expert_df` and `agent_df`."""

    def __init__(self, val_dataset, action_space: str, lookups: LookupTables,
                 global_normalizer: StateNormalizer):
        self.dataset = val_dataset
        self.candidate_grid = val_dataset.candidate_grid
        self.is_azel = val_dataset.candidate_grid.is_azel
        self.action_space = action_space
        self.lookups = lookups
        self.global_normalizer = global_normalizer
        self.cyclical_feature_names = (
            global_normalizer.cyclical_feature_names if global_normalizer is not None else []
        )

        self.expert_df: pd.DataFrame = pd.DataFrame()
        self.agent_df:  pd.DataFrame = pd.DataFrame()
        self.errors_df: pd.DataFrame = pd.DataFrame()

        self._populate_expert_df()
        self.segments = self._get_expert_idx_segments()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _populate_expert_df(self) -> None: ...

    @abstractmethod
    def populate_agent_df(self, *args, **kwargs) -> None: ...

    # ------------------------------------------------------------------
    # Shared extraction
    # ------------------------------------------------------------------

    def _extract_expert_data(self, desired_indices) -> pd.DataFrame:
        filtered = self.dataset._df.iloc[desired_indices].reset_index(drop=True)
        out = pd.DataFrame()

        candidate_idxs = filtered['candidate_idx'].values.copy()
        z_mask = candidate_idxs == -1
        if z_mask.any() and self.is_azel:
            zenith_bin = self.candidate_grid.ang2idx(lon=0, lat=np.pi / 2)
            candidate_idxs[z_mask] = zenith_bin

        out['candidate_idx'] = candidate_idxs
        out['timestamp']  = filtered['timestamp'].values
        out['night'] = filtered['night'].values
        out['datetime'] = pd.to_datetime(out['timestamp'].values, unit='s')
        _has_filter_idx = 'filter_idx' in filtered.columns
        _has_filter_onehot = 'is_filter' in filtered.columns
        _has_filter_name = 'filter' in filtered.columns
        if not (_has_filter_idx or _has_filter_onehot or _has_filter_name):
            raise ValueError('no filter info found in expert data')
        elif _has_filter_name and not _has_filter_idx:
            filtered['filter_idx'] = filtered['filter'].map(self.lookups.survey.filter2idx).fillna(-1)
        elif not _has_filter_name and _has_filter_idx:
            filtered['filter'] = filtered['filter_idx'].map(self.lookups.survey.idx2filter).fillna(-1)
        elif _has_filter_name and _has_filter_idx:
            pass
        elif not _has_filter_name and not _has_filter_idx:
            for filt, idx in self.lookups.survey.filter2idx.items():
                mask = filtered[f'is_filter_{filt}'].astype(bool).values
                filtered.loc[mask, 'filter_idx'] = idx
                filtered.loc[mask, 'filter'] = filt
        else:
            raise ValueError('Missing if/else condition -- this should never happen')
        out['filter_idx'] = filtered['filter_idx'].values
        out['filter'] = out['filter_idx'].map(self.lookups.survey.idx2filter).fillna(-1)

        out['candidate_az'], out['candidate_el'], out['candidate_ra'], out['candidate_dec'] = self._get_candidate_coords(
            out['candidate_idx'].values, timestamps=out['timestamp'].values,
        )
        out['ra'], out['dec'] = filtered['ra'].values, filtered['dec'].values
        out['az'], out['el'] = filtered['az'].values, filtered['el'].values
        out['airmass'] = filtered['airmass'].values

        # Hex name of the observed field -- used to get fields center (not dithers)
        # for apples-to-apples comparison to learned scheduler
        if 'field' in filtered.columns:
            out['field'] = filtered['field'].values

        # Pull through all configured global features. Missing ones get NaN so
        # downstream math doesn't silently break on None.
        for feat in self.dataset.global_feature_names:
            if feat in out.columns:
                continue
            out[feat] = filtered[feat] if feat in filtered.columns else np.nan

        for feat in self.dataset.global_feature_names:
            if feat in out.columns:
                continue
            out[feat] = filtered[feat] if feat in filtered.columns else np.nan

        # Cyclical pairs (_cos, _sin) carried through above represent the same
        # angle as the raw 'lst'/'sun_ra'/etc. columns the env never stored.
        # Recover those raw angle columns so downstream code (plots, convert_to_deg,
        # error metrics) can use them uniformly with the agent_df.
        if self.cyclical_feature_names:
            inverse_cyclical_norm(
                target=None,
                df=out,
                drop_cyclical_components=False,
                cyclical_feature_names=self.cyclical_feature_names,
            )

        return out

    def _populate_expert_derived(self, expert_df: pd.DataFrame,
                                  prev_expert_df: Optional[pd.DataFrame]) -> None:
        """Compute moon distance, airmass, and slew distances on the expert df."""
        cand_radecs = expert_df[['candidate_ra', 'candidate_dec']].to_numpy()
        radecs     = expert_df[['ra', 'dec']].to_numpy()
        timestamps = expert_df['timestamp'].values

        expert_df['candidate_moon_distance'] = calc_moon_dist(cand_radecs, timestamps)
        expert_df['moon_distance']     = calc_moon_dist(radecs, timestamps)
        expert_df['candidate_airmass']       = calc_airmass(expert_df['candidate_el'].to_numpy())

    # ------------------------------------------------------------------
    # Candidate/field coordinate lookups
    # ------------------------------------------------------------------

    def _get_candidate_coords(self, candidate_idxs, timestamps):
        if self.is_azel:
            az_arr = np.array([self.candidate_grid.lon[b] for b in candidate_idxs])
            el_arr = np.array([self.candidate_grid.lat[b] for b in candidate_idxs])
            ra_arr = np.zeros(len(candidate_idxs))
            dec_arr = np.zeros(len(candidate_idxs))
            for i, (t, az, el) in enumerate(zip(timestamps, az_arr, el_arr)):
                ra_arr[i], dec_arr[i] = ephemerides.topographic_to_equatorial(az=az, el=el, time=float(t))
        else:
            ra_arr = np.array([self.candidate_grid.lon[b] for b in candidate_idxs])
            dec_arr = np.array([self.candidate_grid.lat[b] for b in candidate_idxs])
            az_arr = np.zeros(len(candidate_idxs))
            el_arr = np.zeros(len(candidate_idxs))
            for i, (t, b) in enumerate(zip(timestamps, candidate_idxs)):
                az_arr[i], el_arr[i] = ephemerides.equatorial_to_topographic(
                    ra=self.candidate_grid.lon[b], dec=self.candidate_grid.lat[b], time=float(t),
                )
        return az_arr, el_arr, ra_arr, dec_arr

    def _field_center_radecs(self, df: pd.DataFrame, pointing_radecs: np.ndarray) -> np.ndarray:
        """Lookup-table center of each row's observed field, in radians.

        Parameters
        ----------
        df : pd.DataFrame
            Expert rows, optionally carrying the hex `field` name.
        pointing_radecs : np.ndarray
            [n_rows, 2] dithered (ra, dec) pointing in radians; used where the
            field is missing or not in the lookups.

        Returns
        -------
        np.ndarray
            [n_rows, 2] (ra, dec) in radians.
        """
        out = pointing_radecs.copy()
        if 'field' not in df.columns:
            return out
        fields = self.lookups.fields
        fid = df['field'].map(dict(zip(fields['field'], fields.index)))
        ok = fid.notna().to_numpy()
        out[ok] = fields.loc[fid[ok].astype(int), ['ra', 'dec']].to_numpy()
        return out

    def _get_field_coords(self, field_ids, timestamps):
        ra_arr  = np.array([self.lookups.fields['ra'][f] for f in field_ids])
        dec_arr = np.array([self.lookups.fields['dec'][f] for f in field_ids])
        az_arr  = np.zeros(len(field_ids))
        el_arr  = np.zeros(len(field_ids))
        for i, (t, ra, dec) in enumerate(zip(timestamps, ra_arr, dec_arr)):
            az_arr[i], el_arr[i] = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=float(t))
        return ra_arr, dec_arr, az_arr, el_arr

    def _get_expert_idx_segments(self):
        diffs = np.diff(self.dataset.current_state_idxs)
        breaks = np.where(diffs > 1)[0] + 1
        return np.split(self.dataset.next_state_idxs, breaks)

    @staticmethod
    def _get_valid_state_mask(timestamps, max_time_diff_min):
        """True for indices that are within `max_time_diff_min` of their predecessor."""
        max_diff_sec = max_time_diff_min * 60
        diffs = np.diff(timestamps).astype(float)
        valid = diffs <= max_diff_sec
        return np.insert(valid, 0, False)

    # ------------------------------------------------------------------
    # Base agent df + shared errors
    # ------------------------------------------------------------------

    def _get_base_agent_df(self, candidate_idxs, filter_idxs, timestamps) -> pd.DataFrame:
        df = pd.DataFrame()
        df['candidate_idx'] = candidate_idxs
        df['timestamp'] = timestamps.astype(np.int64)
        df['datetime'] = pd.to_datetime(df['timestamp'].values, unit='s')
        df['candidate_az'], df['candidate_el'], df['candidate_ra'], df['candidate_dec'] = self._get_candidate_coords(candidate_idxs, timestamps)
        df['filter_idx'] = filter_idxs
        df['filter'] = df['filter_idx'].map(self.lookups.survey.idx2filter)
        df['az'] = df['el'] = df['ra'] = df['dec'] = np.nan
        return df

    def populate_errors_df(self):
        """Angular separation between expert and agent candidate choices, in radians.

        Assumes both dataframes have candidate_ra/candidate_dec already in degrees (call
        `convert_to_deg()` on both first). Output is stored as radians and is
        converted to degrees by the standard `convert_to_deg()` pass.
        """

        expert_radec_rad_cand = self.expert_df[['candidate_ra', 'candidate_dec']].to_numpy() * units.deg
        agent_radec_rad_cand  = self.agent_df[['candidate_ra', 'candidate_dec']].to_numpy() * units.deg

        expert_radec_rad = self.expert_df[['ra', 'dec']].to_numpy() * units.deg
        agent_radec_rad = self.agent_df[['ra', 'dec']].to_numpy() * units.deg

        cand_angseps = np.fromiter(
            (angular_separation(p1, p2) for p1, p2 in zip(expert_radec_rad_cand, agent_radec_rad_cand)),
            dtype=float, count=len(expert_radec_rad_cand),
        )
        angseps = np.fromiter(
            (angular_separation(p1, p2) for p1, p2 in zip(expert_radec_rad, agent_radec_rad)),
            dtype=float, count=len(expert_radec_rad),
        )

        self.errors_df = pd.DataFrame({
            'timestamp': self.expert_df['timestamp'].values,
            'candidate_angular_separation': cand_angseps,
            'angular_separation': angseps
        })

    def convert_to_deg(self, df: pd.DataFrame) -> pd.DataFrame:
        return _convert_df_to_deg(df)


# ----------------------------------------------------------------------
# Single-step
# ----------------------------------------------------------------------

class SingleStepDataContainer(DataContainer):
    """Expert vs agent on one-step-ahead predictions from the validation set."""

    def __init__(self, val_dataset, action_space: str, lookups: LookupTables,
                 global_normalizer: StateNormalizer):
        self.prev_expert_df: pd.DataFrame = pd.DataFrame()
        # Previous telescope pointing in radians, cached by _populate_expert_df
        # before the frames are converted to degrees.
        self._prev_radecs_rad = None
        self._prev_cand_radecs_rad = None
        super().__init__(val_dataset, action_space, lookups, global_normalizer)

    def _populate_expert_df(self) -> None:
        self.expert_df = self._extract_expert_data(self.dataset.next_state_idxs)
        self.prev_expert_df = self._extract_expert_data(self.dataset.current_state_idxs)

        self._populate_expert_derived(self.expert_df, self.prev_expert_df)

        # Previous-state derived
        prev_cand_radecs = self.prev_expert_df[['candidate_ra', 'candidate_dec']].to_numpy()
        prev_radecs     = self.prev_expert_df[['ra', 'dec']].to_numpy()
        prev_ts         = self.prev_expert_df['timestamp'].values
        prev_cand_els    = self.prev_expert_df['candidate_el'].to_numpy()

        self.prev_expert_df['candidate_moon_distance'] = calc_moon_dist(prev_cand_radecs, prev_ts)
        self.prev_expert_df['moon_distance']     = calc_moon_dist(prev_radecs, prev_ts)
        self.prev_expert_df['candidate_airmass']       = calc_airmass(prev_cand_els)

        # Transitions
        cand_radecs = self.expert_df[['candidate_ra', 'candidate_dec']].to_numpy()
        radecs     = self.expert_df[['ra', 'dec']].to_numpy()
        self.expert_df['candidate_slew_dist'] = calc_slew_distance(prev_cand_radecs, cand_radecs)
        self.expert_df['slew_dist']     = calc_slew_distance(prev_radecs, radecs)

        # Agent slews start at the previous field's center, not its dithered pointing
        self._prev_radecs_rad     = self._field_center_radecs(self.prev_expert_df, prev_radecs)
        self._prev_cand_radecs_rad = prev_cand_radecs.copy()

        self.convert_to_deg(self.expert_df)
        self.convert_to_deg(self.prev_expert_df)

    def populate_agent_df(self, candidate_idxs, filter_idxs, timestamps, field_ids=None) -> None:
        df = self._get_base_agent_df(candidate_idxs, filter_idxs, timestamps)
        df['night'] = self.expert_df['night'].values
        df['candidate_airmass'] = calc_airmass(df['candidate_el'].to_numpy())

        # Borrow environmental params from expert (same timestamps).
        for col in ('moon_phase', 'fwhm', 'sun_az', 'sun_el', 'moon_az', 'moon_el'):
            df[col] = self.expert_df[col].values if col in self.expert_df.columns else np.nan

        cand_radecs = df[['candidate_ra', 'candidate_dec']].to_numpy()
        df['candidate_moon_distance'] = calc_moon_dist(cand_radecs, timestamps)

        df['candidate_slew_dist'] = calc_slew_distance(self._prev_cand_radecs_rad, cand_radecs)

        if field_ids is not None:
            ra, dec, az, el = self._get_field_coords(field_ids, timestamps)
            df['ra'], df['dec'], df['az'], df['el'] = ra, dec, az, el
            df['airmass'] = calc_airmass(el)
            df['ha'] = np.fromiter(
                (ephemerides.equatorial_to_hour_angle(ra=r, dec=d, time=float(t))
                 for r, d, t in zip(ra, dec, timestamps)),
                dtype=float, count=len(timestamps),
            )
            radecs = np.column_stack([ra, dec])
            df['moon_distance'] = calc_moon_dist(radecs, timestamps)
            df['slew_dist'] = calc_slew_distance(self._prev_radecs_rad, radecs)
            df['field_id'] = field_ids

        self.agent_df = df


# ----------------------------------------------------------------------
# Multi-step
# ----------------------------------------------------------------------

class MultiStepDataContainer(DataContainer):
    """Expert vs agent across whole-episode rollouts from the offline runner."""

    def __init__(self, val_dataset, action_space: str, lookups: LookupTables, norm_stats: NormStats,
                 global_normalizer: StateNormalizer):
        self.expert_valid_mask: np.ndarray = np.array([], dtype=bool)
        self.agent_valid_mask:  np.ndarray = np.array([], dtype=bool)
        self.agent_candidate_feat_dict: dict = {}
        self.norm_stats = norm_stats
        super().__init__(val_dataset, action_space, lookups, global_normalizer)

    def _populate_expert_df(self) -> None:
        self.expert_df = self._extract_expert_data(self.dataset.state_idxs)
        self.expert_valid_mask = self._get_valid_state_mask(
            self.expert_df['timestamp'].values, max_time_diff_min=EXPERT_MAX_GAP,
        )
        self._populate_expert_derived(self.expert_df, prev_expert_df=None)

        prev_df_slew    = self._extract_expert_data(self.dataset.current_state_idxs)
        prev_cand_radecs = prev_df_slew[['candidate_ra', 'candidate_dec']].to_numpy()
        prev_radecs     = prev_df_slew[['ra', 'dec']].to_numpy()

        # Locate each next_state_idx in state_idxs to get its row in expert_df.
        next_positions  = np.searchsorted(self.dataset.state_idxs, self.dataset.next_state_idxs)
        nxt_cand_radecs  = self.expert_df[['candidate_ra', 'candidate_dec']].to_numpy()[next_positions]
        nxt_radecs      = self.expert_df[['ra', 'dec']].to_numpy()[next_positions]

        pair_cand_slews  = calc_slew_distance(prev_cand_radecs, nxt_cand_radecs)
        pair_slews      = calc_slew_distance(prev_radecs, nxt_radecs)

        cand_slew = np.full(len(self.expert_df), np.nan)
        slew     = np.full(len(self.expert_df), np.nan)
        cand_slew[next_positions] = pair_cand_slews
        slew[next_positions]     = pair_slews

        self.expert_df['candidate_slew_dist'] = cand_slew
        self.expert_df['slew_dist']     = slew

        self.convert_to_deg(self.expert_df)

    def populate_agent_df(self, candidate_idxs, filter_idxs, timestamps,
                          field_ids, glob_df, candidate_feat_dict) -> None:
        df = self._get_base_agent_df(candidate_idxs, filter_idxs, timestamps)
        df['night'] = glob_df['night'].values
        df['ra'], df['dec'], df['az'], df['el'] = self._get_field_coords(field_ids, timestamps)

        df['moon_phase'] = calc_moon_phase(timestamps)
        df['sun_az'], df['sun_el'], df['moon_az'], df['moon_el'] = calc_sun_and_moon_pos(timestamps)
        df['candidate_airmass'] = calc_airmass(df['candidate_el'].to_numpy())
        df['airmass']     = calc_airmass(df['el'].to_numpy())

        radecs     = df[['ra', 'dec']].to_numpy()
        cand_radecs = df[['candidate_ra', 'candidate_dec']].to_numpy()
        df['candidate_moon_distance'] = calc_moon_dist(cand_radecs, timestamps)
        df['moon_distance']     = calc_moon_dist(radecs, timestamps)

        # Compute valid-state mask BEFORE we use it on slew distances.
        self.agent_valid_mask = self._get_valid_state_mask(
            df['timestamp'].values, max_time_diff_min=AGENT_MAX_GAP,
        )

        cand_slew = calc_slew_distance(cand_radecs[:-1], cand_radecs[1:])
        slew     = calc_slew_distance(radecs[:-1], radecs[1:])
        cand_slew = np.insert(cand_slew, 0, np.nan)
        slew     = np.insert(slew, 0, np.nan)
        cand_slew[~self.agent_valid_mask] = np.nan
        slew[~self.agent_valid_mask]     = np.nan
        df['candidate_slew_dist'] = cand_slew
        df['slew_dist']     = slew

        # # Carry through remaining globals from the runner output (normalized form).
        # for feat in self.dataset.global_feature_names:
        #     if feat not in df.columns:
        #         df[feat] = glob_df[feat].values if feat in glob_df.columns else np.nan

        # Carry through remaining globals from the runner output (normalized form).
        glob_feats_carried = []
        for feat in self.dataset.global_feature_names:
            if feat not in df.columns:
                if feat in glob_df.columns:
                    df[feat] = glob_df[feat].values
                    glob_feats_carried.append(feat)
                else:
                    df[feat] = np.nan

        # Inverse-normalize ONLY the columns just pulled from the runner.
        # Everything computed fresh above is already in raw units.
        self.global_normalizer.inverse_transform_df(
            df,
            feature_names=glob_feats_carried,
            **self.norm_stats.normalizer_kwargs('global_features'),
        )

        self.agent_candidate_feat_dict = candidate_feat_dict
        self.agent_df = self.convert_to_deg(df)
