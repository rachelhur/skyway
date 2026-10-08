from blancops.configs.paths import OfflineRunPaths
from blancops.io.file_io import read_schedule_csv

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import pickle
from pathlib import Path

from blancops.ephemerides import ephemerides
from blancops.ephemerides.time_utils import unix_to_datetime
from blancops.configs.constants import *
import logging

from blancops.plotting.plotting import plot_schedule_movie, plot_schedule_whole
from blancops.configs.enums import grid_is_azel, is_field_level

logger = logging.getLogger(__name__)


class OfflineRunner:
    def __init__(self, agent, policy, cfg, lookups, telescope,
                 outdir=None, save_observing_script=False, observing_script_kwargs=None,
                 save_state_features=False, save_movie=False, save_mollweide=False, plot_bins=False,
                 dump_moonset_q=False):
        self.agent = agent
        self.cfg = cfg
        self.policy = policy
        self.lookups = lookups
        self.field_level = is_field_level(cfg.data.action_space)
        # Bin-level models always draw bins; field-level models draw them in movies only when plot_bins
        self._movie_bins = plot_bins or not self.field_level
        self._mollweide_bins = not self.field_level
        self.field_choice_method = self.agent.field_choice_method
        self.outdir = Path(outdir) # XXX fix if None branch
        self.paths = OfflineRunPaths(self.outdir)
        self.save_movie = save_movie
        self.save_mollweide = save_mollweide
        self.telescope = telescope
        if save_observing_script and telescope.observing_script_writer is None:
            raise ValueError(f"Telescope {telescope.key!r} has no observing script writer.")
        self.save_observing_script = save_observing_script
        self.observing_script_kwargs = observing_script_kwargs or {}
        # When True, also saves glob/candidate observation arrays as .npz per night
        # for use with diagnostic plot functions. Off by default to protect memory.
        self.save_state_features = save_state_features
        # One-shot per-filter Q breakdown at the first post-moonset step.
        self.dump_moonset_q = dump_moonset_q
        self._moonset_dumped = False
        self._prev_moon_el = None

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.paths.nights.mkdir(exist_ok=True)
        if self.save_observing_script:
            self.paths.observing_scripts.mkdir(exist_ok=True)
        if self.save_movie or self.save_mollweide:
            self.paths.plots.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Per-night CSV streaming
    # ------------------------------------------------------------------

    def _flush_night_csv(self, rows, night_label):
        """Write lightweight schedule rows for one night to `nights/<night_label>.csv`. Returns path."""
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Filter out zenith and wait sentinels before saving
        real_mask = (df['bin_id'] != ZENITH_BIN_NUM) & (df['bin_id'] != WAIT_SIGNAL)
        df = df[real_mask].copy()
        if df.empty:
            return None
        df['filter'] = df['filter_idx'].map(self.lookups.survey.idx2filter)
        path = self.paths.nights / f'{night_label}.csv'
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Per-night / full-survey observing script writing
    # ------------------------------------------------------------------

    def _write_observing_script(self, df, name):
        """Write a schedule DataFrame as the telescope's observing script. Returns path or None."""
        if df is None or df.empty:
            return None
        return self.telescope.observing_script_writer(
            df, name, self.paths.observing_scripts, self.lookups, **self.observing_script_kwargs
        )

    @staticmethod
    def _load_full_survey_df(night_csv_paths):
        """Concatenate all per-night CSVs in time order. Returns None when there are no rows."""
        frames = [read_schedule_csv(p) for p in night_csv_paths if p is not None]
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True).sort_values('timestamp')
        return None if df.empty else df

    # ------------------------------------------------------------------
    # Optional obs-feature flushing (for diagnostic plots)
    # ------------------------------------------------------------------

    def _flush_obs_features(self, obs_dict, night_label):
        """Save glob/candidate observation arrays as compressed .npz."""
        arr_dict = {}
        for key, values in obs_dict.items():
            arr = np.asarray(values)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32, copy=False)
            arr_dict[key] = arr
        path = self.paths.nights / f'{night_label}_obs.npz'
        np.savez_compressed(path, **arr_dict)
        return path

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _field_pos_from_df(self, df):
        """Return list of (ra_rad, dec_rad) tuples for each row in df."""
        fids = df['field_id'].values
        return [
            (float(self.lookups.fields['ra'].values[fid]),
             float(self.lookups.fields['dec'].values[fid]))
            for fid in fids
        ]

    def _bin_idxs_from_df(self, df, draw_bins):
        """HEALPix bin per schedule row when the plot draws bins, else None (fields only).

        Parameters
        ----------
        df : pd.DataFrame
            Schedule rows with a bin_id column.
        draw_bins : bool
            Whether the plot draws the bin layer.

        Returns
        -------
        np.ndarray or None
            Bin index per row, shape (n_rows,), or None.
        """
        return df['bin_id'].values if draw_bins else None

    def _movie_path(self, night_label):
        return self.paths.plots / f'{night_label}_movie.gif'

    def _save_movie(self, df, night_label):
        plot_schedule_movie(
            outfile=str(self._movie_path(night_label)),
            times=df['timestamp'].values,
            field_pos=self._field_pos_from_df(df),
            bin_idxs=self._bin_idxs_from_df(df, self._movie_bins),
            nside=self.cfg.data.nside,
            is_azel=grid_is_azel(self.cfg.data.action_space),
        )

    def save_missing_movies(self, manifest: dict) -> None:
        """Render a movie for each night CSV in the manifest that has no movie yet.

        Movies are named after the CSV, `plots/<csv stem>_movie.gif`.

        Parameters
        ----------
        manifest : dict
            Night key -> path of that night's schedule CSV (None for an empty night).
        """
        self.paths.plots.mkdir(parents=True, exist_ok=True)
        n_drawn, n_skipped = 0, 0
        for csv_path in manifest.values():
            if csv_path is None or self._movie_path(Path(csv_path).stem).exists():
                n_skipped += 1
                continue
            self._save_movie(read_schedule_csv(csv_path), Path(csv_path).stem)
            n_drawn += 1
        logger.info(f'Movies: drew {n_drawn}, skipped {n_skipped} (empty or already in {self.paths.plots})')

    def _save_mollweide(self, df):
        plot_schedule_whole(
            outfile=self.paths.plots / 'all_nights_mollweide.png',
            times=df['timestamp'].values,
            field_pos=self._field_pos_from_df(df),
            bin_idxs=self._bin_idxs_from_df(df, self._mollweide_bins),
            nside=self.cfg.data.nside,
        )

    @staticmethod
    def _restore_nans(arr, mask):
        if arr is None or mask is None or not mask.any():
            return arr.copy() if arr is not None else arr
        out = arr.astype(np.float32, copy=True)
        out[mask] = np.nan
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _maybe_dump_moonset_q(self, obs, info):
        """Print the per-filter Q breakdown once, at the first post-moonset step."""
        if not self.dump_moonset_q or self._moonset_dumped:
            return
        # Local import breaks the offline_runner <-> evaluations package cycle.
        from blancops.rl.evaluations.helpers import dump_filter_q_breakdown
        t = info.get('timestamp')
        if t is None:
            return
        moon_radec = ephemerides.get_source_ra_dec('moon', time=t)
        _, moon_el = ephemerides.equatorial_to_topographic(moon_radec[0], moon_radec[1], time=t)
        if self._prev_moon_el is not None and self._prev_moon_el > 0 and moon_el <= 0:
            logger.info(
                f"[moonset q-dump] ts={t} moon el {self._prev_moon_el:.4f}->{moon_el:.4f} rad"
            )
            dump_filter_q_breakdown(self.policy, obs, info, self.lookups.survey.idx2filter)
            self._moonset_dumped = True
        self._prev_moon_el = moon_el

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    def run(self, env):
        self.policy.eval()

        hpGrid = ephemerides.HealpixGrid(nside=self.cfg.data.nside, is_azel=grid_is_azel(self.cfg.data.action_space))

        obs, info = env.reset()
        self._moonset_dumped = False
        self._prev_moon_el = None
        running_reward = 0
        terminated = False
        truncated = False

        # Lightweight per-step schedule records — only ~5 scalars each
        per_night_rows = []
        # Optional obs feature buffers (only populated when save_state_features=True)
        per_night_obs = self._empty_obs_buffer() if self.save_state_features else None

        manifest = {}  # night_key -> csv path
        dispersion = {}  # night_key -> per-filter (D_b, baseline_b) at the night's last in-night step
        steps = {}  # night_key -> per-exposure dead time and predicted teff (NaN when not computed)
        night_dispersion = info['survey_progress_tracker'].dispersion_index()
        reward = 0
        night_idx = 0
        current_night_key = f'night-{night_idx}'
        night_label = info['night_label']

        i = 0
        last_bin_idx = ZENITH_BIN_NUM
        field_id = ZENITH_FIELD_ID
        filter_idx = ZENITH_FILTER_IDX

        # Approximate...
        pbar = tqdm(total=200 * env.get_wrapper_attr('max_nights'), dynamic_ncols=True,
                    desc=f"Rolling out policy for night {night_idx} step {i}")

        while not (terminated or truncated):
            with torch.no_grad():
                action_mask = info.get('action_mask', None)

                if not action_mask.any():
                    if last_bin_idx != WAIT_SIGNAL:
                        logger.warning(f"No observable, incomplete field at {unix_to_datetime(info['timestamp'])}; waiting.")
                    bin_idx = WAIT_SIGNAL
                else:
                    bin_idx, filter_idx, field_id = self.agent.choose_bin_filter_field(obs, info, hpGrid, epsilon=None)

                self._maybe_dump_moonset_q(obs, info)

                obs_timestamp = info.get('timestamp')
                pre_step_glob = obs['global_state']
                pre_step_glob_nan_mask = info.get('glob_nan_mask')
                pre_step_cand = obs['candidate_state']
                pre_step_cand_nan_mask = info.get('candidate_nan_mask')

                obs, reward, terminated, truncated, info = env.step(
                    self.agent.command_to_env_action(bin_idx, filter_idx, field_id)
                )

                is_first_wait = (bin_idx == WAIT_SIGNAL) and (last_bin_idx != WAIT_SIGNAL)
                is_real_obs = bin_idx >= 0
                if is_first_wait or is_real_obs:
                    per_night_rows.append({
                        'timestamp':  obs_timestamp,
                        'field_id':   int(field_id),
                        'filter_idx': int(filter_idx),
                        'bin_id':     int(bin_idx),
                        'reward':     float(reward),
                    })
                if self.save_state_features and is_real_obs:
                    per_night_obs['glob_observations'].append(
                        self._restore_nans(pre_step_glob, pre_step_glob_nan_mask)
                    )
                    per_night_obs['candidate_observations'].append(
                        self._restore_nans(pre_step_cand, pre_step_cand_nan_mask)
                    )
                    per_night_obs['action_masks'].append(
                        np.asarray(action_mask, dtype=bool)
                    )

                rec = info['step_record']
                if is_real_obs and rec is not None:
                    steps.setdefault(current_night_key, []).append(
                        {'dead_time': float(rec['dead_time']), 'teff_pred': float(rec.get('teff_pred', np.nan))}
                    )
                running_reward += reward
                last_bin_idx = bin_idx

                # Night boundary: flush current night and open next
                if info.get('night_idx') == night_idx:
                    night_dispersion = info['survey_progress_tracker'].dispersion_index()
                else:
                    dispersion[current_night_key] = self._dispersion_record(night_dispersion)
                    manifest[current_night_key] = self._finish_night(per_night_rows, per_night_obs, night_label)
                    per_night_rows = []
                    if self.save_state_features:
                        per_night_obs = self._empty_obs_buffer()

                    night_idx = info.get('night_idx')
                    current_night_key = f'night-{night_idx}'
                    night_label = info['night_label']

                i += 1
                pbar.update(1)
                pbar.set_description(f"Rolling out policy for night {night_idx} step {i}")

        logger.info(f'terminated at step {i}')

        # Flush the final night
        dispersion[current_night_key] = self._dispersion_record(night_dispersion)
        manifest[current_night_key] = self._finish_night(per_night_rows, per_night_obs, night_label)

        if self.save_observing_script or self.save_mollweide:
            full_df = self._load_full_survey_df([Path(p) for p in manifest.values() if p is not None])
            if full_df is not None:
                if self.save_observing_script:
                    self._write_observing_script(full_df, 'all_nights')
                if self.save_mollweide:
                    self._save_mollweide(full_df)

        pbar.close()

        rollout_info = self._construct_diagnostics(running_reward, manifest, dispersion, steps)
        self._write_diagnostics_to_file(rollout_info)
        return rollout_info

    @staticmethod
    def _empty_obs_buffer() -> dict:
        return {'glob_observations': [], 'candidate_observations': [], 'action_masks': []}

    def _finish_night(self, rows, obs_buffer, night_label):
        """Write one night's CSV and its optional outputs. Returns the CSV path as str, or None for an empty night.

        Avoids OOM."""
        csv_path = self._flush_night_csv(rows, night_label)
        if csv_path is None:
            logger.warning(f"Night {night_label}: no exposures scheduled (no field was observable and incomplete); "
                           f"no schedule files written for this night.")
        else:
            night_df = read_schedule_csv(csv_path)
            logger.info(f"Night {night_label}: {len(night_df)} exposures scheduled.")
            if self.save_observing_script:
                self._write_observing_script(night_df, night_label)
            if self.save_movie:
                self._save_movie(night_df, night_label)
        if self.save_state_features and obs_buffer:
            self._flush_obs_features(obs_buffer, night_label)
            gc.collect()
        return str(csv_path) if csv_path else None

    def _construct_diagnostics(self, total_reward, manifest, dispersion, steps):
        # Readers index the single rollout as 'ep-0'.
        return {
            'ep-0': {
                'manifest': dict(manifest),
                'total_reward': float(total_reward),
                'dispersion': dict(dispersion),
                'steps': dict(steps),
            },
        }

    @staticmethod
    def _dispersion_record(dispersion) -> dict:
        """Per-filter dispersion index D_b = Var_b / m_b and its random-visit baseline, as lists."""
        d, baseline = dispersion
        return {'dispersion': d.tolist(), 'baseline': baseline.tolist()}

    def _write_diagnostics_to_file(self, rollout_info):
        with open(self.paths.rollout_info, 'wb') as handle:
            pickle.dump(rollout_info, handle)
            logger.info(f'Rollout info saved to {self.paths.rollout_info}')
