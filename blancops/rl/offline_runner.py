from blancops.io.file_io import SCHEDULE_KEYS, write_SISPI_from_df

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import pickle
from pathlib import Path

from blancops.ephemerides import ephemerides
from blancops.configs.constants import *
import logging

from blancops.plotting.plotting import plot_bins_movie, plot_schedule_whole

logger = logging.getLogger(__name__)


_PROPID_PLACEHOLDER = 'XXXX-XXXX'
_PROPOSER_PLACEHOLDER = 'ai-scheduler'
_PROGRAM_PLACEHOLDER = 'ai-scheduler-test'


class OfflineRunner:
    def __init__(self, agent, policy, cfg, lookups, num_episodes=1,
                 outdir=None, save_SISPI=True, SISPI_fn="sispi.json",
                 save_state_features=False, save_movie=False, save_mollweide=False,
                 reset_counts_on_exhaustion=False, # whether or not to reset all counts after all targets complete
                 dump_moonset_q=False):
        self.agent = agent
        self.cfg = cfg
        self.policy = policy
        self.num_episodes = num_episodes
        self.lookups = lookups
        self.field_choice_method = self.agent.field_choice_method
        self.outdir = Path(outdir)
        self.save_movie = save_movie
        self.save_mollweide = save_mollweide
        self.save_SISPI = save_SISPI
        self.SISPI_fn = SISPI_fn
        # When True, also saves glob/bin observation arrays as .npz per night
        # for use with diagnostic plot functions. Off by default to protect memory.
        self.save_state_features = save_state_features
        # When every action is masked because the survey targets reachable at
        # this moment are complete, zero the visit counts and carry on rather
        # than idling to the end of the night. Mirrors live operation, where
        # the scheduler was restarted against a fresh history once it ran out.
        self.reset_counts_on_exhaustion = reset_counts_on_exhaustion
        # One-shot per-filter Q breakdown at the first post-moonset step.
        self.dump_moonset_q = dump_moonset_q
        self._moonset_dumped = False
        self._prev_moon_el = None

        self.outdir.mkdir(parents=True, exist_ok=True)
        self._nights_dir = self.outdir / 'nights'
        self._nights_dir.mkdir(exist_ok=True)
        self._sispi_dir = self.outdir / 'sispi'
        if self.save_SISPI:
            self._sispi_dir.mkdir(exist_ok=True)
        self._plots_dir = self.outdir / 'plots'
        if self.save_movie or self.save_mollweide:
            self._plots_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Per-night CSV streaming
    # ------------------------------------------------------------------

    def _flush_night_csv(self, rows, ep_num, night_key):
        """Write lightweight schedule rows for one night to CSV. Returns path."""
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Filter out zenith and wait sentinels before saving
        real_mask = (df[SCHEDULE_KEYS['bin_id']] != ZENITH_BIN_NUM) & (df[SCHEDULE_KEYS['bin_id']] != WAIT_SIGNAL)
        df = df[real_mask].copy()
        if df.empty:
            return None
        df[SCHEDULE_KEYS['filter']] = df[SCHEDULE_KEYS['filter_idx']].map(IDX2FILTER)
        path = self._nights_dir / f'ep-{ep_num}_{night_key}.csv'
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Per-night / full-survey SISPI writing
    # ------------------------------------------------------------------

    def _write_night_sispi(self, df, ep_num, night_key):
        """Write a SISPI JSON for one night from a schedule DataFrame. Returns path or None."""
        if df is None or df.empty:
            return None
        dt_series = pd.Series(
            pd.to_datetime(df[SCHEDULE_KEYS['timestamp']], utc=True, unit='s') - pd.Timedelta(12, 'h')
        )
        out_fn = f'ep-{ep_num}_{night_key}_{self.SISPI_fn}'
        write_SISPI_from_df(
            schedule_df=df,
            out_fn=out_fn,
            save_dir=self._sispi_dir,
            lookups=self.lookups,
            filter_override_val=None,
            proposer=_PROPOSER_PLACEHOLDER,
            program=_PROGRAM_PLACEHOLDER,
            dt_series=dt_series,
            use_date_prefix=False,
            propid=_PROPID_PLACEHOLDER,
        )
        return self._sispi_dir / out_fn

    def _write_full_survey_sispi(self, ep_num, night_csv_paths):
        """Concatenate all per-night CSVs and write a single full-survey SISPI JSON.

        Returns (path, df) so the caller can reuse df for plotting without re-reading.
        """
        frames = [pd.read_csv(p) for p in night_csv_paths if p is not None]
        if not frames:
            return None, None
        df = pd.concat(frames, ignore_index=True).sort_values(SCHEDULE_KEYS['timestamp'])
        if df.empty:
            return None, None
        dt_series = pd.Series(
            pd.to_datetime(df[SCHEDULE_KEYS['timestamp']], utc=True, unit='s') - pd.Timedelta(12, 'h')
        )
        out_fn = f'ep-{ep_num}_full_survey_{self.SISPI_fn}'
        write_SISPI_from_df(
            schedule_df=df,
            out_fn=out_fn,
            save_dir=self._sispi_dir,
            lookups=self.lookups,
            filter_override_val=None,
            dt_series=dt_series,
            use_date_prefix=False,
            proposer=_PROPOSER_PLACEHOLDER,
            program=_PROGRAM_PLACEHOLDER,
            propid=_PROPID_PLACEHOLDER,
        )
        return self._sispi_dir / out_fn, df


    # ------------------------------------------------------------------
    # Optional obs-feature flushing (for diagnostic plots)
    # ------------------------------------------------------------------

    def _flush_obs_features(self, obs_dict, ep_num, night_key):
        """Save glob/bin observation arrays as compressed .npz."""
        arr_dict = {}
        for key, values in obs_dict.items():
            arr = np.asarray(values)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32, copy=False)
            arr_dict[key] = arr
        path = self._nights_dir / f'ep-{ep_num}_{night_key}_obs.npz'
        np.savez_compressed(path, **arr_dict)
        return path

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _field_pos_from_df(self, df):
        """Return list of (ra_rad, dec_rad) tuples for each row in df."""
        fids = df[SCHEDULE_KEYS['field_id']].values
        return [
            (float(self.lookups.fields['ra'].values[fid]),
             float(self.lookups.fields['dec'].values[fid]))
            for fid in fids
        ]

    def _save_movie(self, df, ep_num, night_key):
        outfile = self._plots_dir / f'ep-{ep_num}_{night_key}_movie.gif'
        plot_bins_movie(
            outfile=outfile,
            nside=self.cfg.data.nside,
            times=df[SCHEDULE_KEYS['timestamp']].values,
            idxs=df[SCHEDULE_KEYS['bin_id']].values,
            field_pos=self._field_pos_from_df(df),
            is_azel='azel' in self.cfg.data.action_space,
        )

    def _save_mollweide(self, df, ep_num):
        outfile = self._plots_dir / f'ep-{ep_num}_full_survey_mollweide.png'
        plot_schedule_whole(
            outfile=outfile,
            times=df[SCHEDULE_KEYS['timestamp']].values,
            field_pos=self._field_pos_from_df(df),
            bin_idxs=df[SCHEDULE_KEYS['bin_id']].values,
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
            dump_filter_q_breakdown(self.policy, obs, info, IDX2FILTER)
            self._moonset_dumped = True
        self._prev_moon_el = moon_el

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    def run(self, env):
        self.policy.eval()
        episode_rewards = []

        hpGrid = ephemerides.HealpixGrid(nside=self.cfg.data.nside, is_azel=('azel' in self.cfg.data.action_space))

        for ep_num in tqdm(range(self.num_episodes)):
            obs, info = env.reset()
            self._moonset_dumped = False
            self._prev_moon_el = None
            running_reward = 0
            terminated = False
            truncated = False
            num_nights = env.unwrapped.max_nights

            # Lightweight per-step schedule records — only ~5 scalars each
            per_night_rows = []
            # Optional obs feature buffers (only populated when save_obs_features=True)
            per_night_obs = {'glob_observations': [], 'bin_observations': [], 'action_masks': []} if self.save_state_features else None

            episode_manifest = {}  # night_key -> csv path
            reward = 0
            night_idx = 0
            current_night_key = f'night-{night_idx}'

            i = 0
            last_bin_idx = ZENITH_BIN_NUM
            field_id = ZENITH_FIELD_ID
            filter_idx = ZENITH_FILTER_IDX

            pbar = tqdm(total=250 * num_nights, dynamic_ncols=True,
                        desc=f"Rolling out policy for night {night_idx} step {i}")

            while not (terminated or truncated):
                with torch.no_grad():
                    action_mask = info.get('action_mask', None)

                    if not action_mask.any() and self.reset_counts_on_exhaustion:
                        base_env = env.unwrapped
                        logger.warning(
                            f"Survey targets exhausted at step {i}; zeroing visit "
                            f"counts and continuing."
                        )
                        base_env._survey_progress_tracker.zero_counts()
                        base_env.compute_action_mask()
                        info = base_env.get_info()
                        action_mask = info.get('action_mask', None)

                    if not action_mask.any():
                        logger.warning(f"No valid fields available at step {i} (mask is all zeros).")
                        bin_idx = WAIT_SIGNAL
                    else:
                        bin_idx, filter_idx, field_id = self.agent.choose_bin_filter_field(obs, info, hpGrid, epsilon=None)

                    self._maybe_dump_moonset_q(obs, info)

                    obs_timestamp = info.get('timestamp')
                    pre_step_glob = obs['global_state']
                    pre_step_glob_nan_mask = info.get('glob_nan_mask')
                    pre_step_bin  = obs['bin_state']
                    pre_step_bin_nan_mask  = info.get('bin_nan_mask')

                    obs, reward, terminated, truncated, info = env.step({
                        'bin': np.int32(bin_idx),
                        'field_id': np.int32(field_id),
                        'filter_idx': np.int32(filter_idx),
                    })

                    is_first_wait = (bin_idx == WAIT_SIGNAL) and (last_bin_idx != WAIT_SIGNAL)
                    is_real_obs = bin_idx >= 0
                    if is_first_wait or is_real_obs:
                        per_night_rows.append({
                            SCHEDULE_KEYS['timestamp']: obs_timestamp,
                            SCHEDULE_KEYS['field_id']:  int(field_id),
                            SCHEDULE_KEYS['filter_idx']: int(filter_idx),
                            SCHEDULE_KEYS['bin_id']:    int(bin_idx),
                            'reward':          float(reward),
                        })
                    if self.save_state_features and is_real_obs:
                        per_night_obs['glob_observations'].append(
                            self._restore_nans(pre_step_glob, pre_step_glob_nan_mask)
                        )
                        per_night_obs['bin_observations'].append(
                            self._restore_nans(pre_step_bin, pre_step_bin_nan_mask)
                        )
                        per_night_obs['action_masks'].append(
                            np.asarray(action_mask, dtype=bool)
                        )

                    running_reward += reward
                    last_bin_idx = bin_idx

                    # Night boundary: flush current night and open next
                    if info.get('night_idx') != night_idx:
                        csv_path = self._flush_night_csv(per_night_rows, ep_num, current_night_key)
                        episode_manifest[current_night_key] = str(csv_path) if csv_path else None
                        if csv_path is not None:
                            night_df = pd.read_csv(csv_path)
                            if self.save_SISPI:
                                self._write_night_sispi(night_df, ep_num, current_night_key)
                            if self.save_movie:
                                self._save_movie(night_df, ep_num, current_night_key)
                        if self.save_state_features and per_night_obs:
                            self._flush_obs_features(per_night_obs, ep_num, current_night_key)

                        per_night_rows = []
                        if self.save_state_features:
                            per_night_obs = {'glob_observations': [], 'bin_observations': [], 'action_masks': []}
                            gc.collect()

                        night_idx = info.get('night_idx')
                        current_night_key = f'night-{night_idx}'

                    i += 1
                    pbar.update(1)
                    pbar.set_description(f"Rolling out policy for night {night_idx} step {i}")

            logger.info(f'terminated at step {i}')

            # Flush the final night
            csv_path = self._flush_night_csv(per_night_rows, ep_num, current_night_key)
            episode_manifest[current_night_key] = str(csv_path) if csv_path else None
            if csv_path is not None:
                night_df = pd.read_csv(csv_path)
                if self.save_SISPI:
                    self._write_night_sispi(night_df, ep_num, current_night_key)
                if self.save_movie:
                    self._save_movie(night_df, ep_num, current_night_key)
            if self.save_state_features and per_night_obs:
                self._flush_obs_features(per_night_obs, ep_num, current_night_key)
                gc.collect()

            if self.save_SISPI or self.save_mollweide:
                valid_paths = [Path(p) for p in episode_manifest.values() if p is not None]
                _, full_df = self._write_full_survey_sispi(ep_num, valid_paths)
                if self.save_mollweide and full_df is not None:
                    self._save_mollweide(full_df, ep_num)

            episode_rewards.append(running_reward)
            pbar.close()

        rollout_info = self._construct_diagnostics(episode_rewards, episode_manifest, ep_num)
        self._write_diagnostics_to_file(rollout_info)
        return rollout_info

    def _construct_diagnostics(self, episode_rewards, episode_manifest, ep_num):
        diagnostics = {
            f'ep-{ep_num}': {
                'manifest': dict(episode_manifest),
                'total_reward': float(episode_rewards[ep_num]) if episode_rewards else 0.0,
            },
            'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            'std_reward':  float(np.std(episode_rewards))  if episode_rewards else 0.0,
            'min_reward':  float(np.min(episode_rewards))  if episode_rewards else 0.0,
            'max_reward':  float(np.max(episode_rewards))  if episode_rewards else 0.0,
            'episode_rewards': episode_rewards,
        }
        return diagnostics

    def _write_diagnostics_to_file(self, rollout_info):
        with open(self.outdir / 'rollout_info.pkl', 'wb') as handle:
            pickle.dump(rollout_info, handle)
            logger.info(f'eval_metrics.pkl saved in {self.outdir}')
