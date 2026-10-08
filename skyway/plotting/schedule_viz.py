
from datetime import datetime, timedelta
import math
import os
from venv import logger

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import timezone

from skyway.configs.constants import *
from skyway.ephemerides.ephemerides import HealpixGrid
from skyway.data.features.glob_features import calc_twilight
from skyway.plotting.plotting import plot_schedule_from_file
from collections import defaultdict
from skyway.ephemerides.ephemerides import topographic_to_equatorial
from skyway.math import units
from skyway.configs.enums import grid_is_azel

def save_gifs(schedule_path, save_dir, do_fieldbin, do_bin, do_mollefield, do_ortho, action_space, nside, fid2radec_filepath):
    if do_fieldbin:
        plot_schedule_from_file(
            outfile=save_dir / "agent_fieldbin_schedule.gif",
            schedule_file=schedule_path,
            plot_type='fieldbin',
            nside=nside,
            fields_file=fid2radec_filepath,
            whole=False,
            compare=False,
            expert=False,
            is_azel=grid_is_azel(action_space),
            mollweide=False,
        )
    if do_bin:
        plot_schedule_from_file(
            outfile=save_dir + 'agent_bin_schedule.gif',
            schedule_file=schedule_path,
            plot_type='bin',
            nside=nside,
            fields_file=fid2radec_filepath,
            whole=False,
            compare=False,
            expert=False,
            is_azel=grid_is_azel(action_space),
            mollweide=False,
        )

        if action_space == 'radec':
            if do_mollefield:
                # Mollefield
                logger.info("Creating static plots")
                plot_schedule_from_file(
                    outfile=save_dir / "mollweide.png",
                    schedule_file=schedule_path,
                    plot_type='bin',
                    nside=nside,
                    fields_file=fid2radec_filepath,
                    whole=True,
                    compare=True,
                    expert=True,
                    is_azel=grid_is_azel(action_space),
                    mollweide=True,
                )
            if do_ortho:
                plot_schedule_from_file(
                    outfile=save_dir / "ortho.png",
                    schedule_file=schedule_path,
                    plot_type='bin',
                    nside=nside,
                    fields_file=fid2radec_filepath,
                    whole=True,
                    compare=True,
                    expert=True,
                    is_azel=grid_is_azel(action_space),
                    mollweide=False,
                )

def save_survey_diagnostics(eval_metrics, save_dir, field_lookup, nside, action_space, ep_num=0):
    eval_metrics = eval_metrics[f'ep-{ep_num}']
    _preflat_metrics = defaultdict(list)
    hpGrid = HealpixGrid(nside=nside, is_azel=grid_is_azel(action_space))

    # Extract the arrays from each night
    for night_key, metrics_dict in eval_metrics.items():
        for metric_name, array_values in metrics_dict.items():
            _preflat_metrics[metric_name].append(array_values)

    # Concatenate the collected arrays for each metric
    survey_metrics = {}
    for k, list_of_arrays in _preflat_metrics.items():
        survey_metrics[k] = np.concatenate(list_of_arrays)

    # Filter out zenith and wait states
    sel_valid_obs = survey_metrics['bin'] != ZENITH_BIN_NUM
    sel_valid_obs &= survey_metrics['bin'] != WAIT_SIGNAL
    for k, v in survey_metrics.items():
        survey_metrics[k] = v[sel_valid_obs]

    field_ids = survey_metrics['field_id']
    bin_nums = survey_metrics['bin']
    timestamps = survey_metrics['timestamp']

    # --- Plot bin and field radecs --- #

    # Get bin radecs
    bin2coord = {int(i): (lon, lat) for i, (lon, lat) in enumerate(zip(hpGrid.lon, hpGrid.lat))}
    if hpGrid.is_azel:
        bin_azels = np.array([bin2coord[bid] for bid in bin_nums])
        bin_radecs = np.zeros(shape=bin_azels.shape)
        for i, ts in enumerate(timestamps):
            bin_radecs[i] = topographic_to_equatorial(bin_azels[i, 0], bin_azels[i, 1], time=ts)
    else:
        bin_radecs = np.array([bin2coord[bid] for bid in bin_nums])
    bin_radecs /= units.deg

    # Get field radecs
    field_radecs = np.array([[field_lookup['ra'][field_id], field_lookup['dec'][field_id]] for field_id in field_ids])
    field_radecs /= units.deg

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
    axs[0].scatter(bin_radecs[:, 0], bin_radecs[:, 1], cmap='Blues', c=np.arange(len(bin_radecs)))
    axs[0].set_xlabel('ra ')
    axs[0].set_ylabel('dec')
    axs[0].set_title('Bins')

    axs[1].scatter(field_radecs[:, 0], field_radecs[:, 1], label='agent', cmap='Purples', c=np.arange(len(field_radecs)), s=10)
    axs[1].set_xlabel('ra ')
    axs[1].set_title('Fields')

    fig.savefig(save_dir / "survey_ra_vs_dec.png")

def save_nightly_diagnostics(eval_metrics, observing_night_strs, schedule_outdir, action_architecture, env, nside, lookup_dirpath, num_actions, action_space, ep_num=0, do_gifs=False):
    eval_metrics = eval_metrics[f'ep-{ep_num}']
    night_info = []
    for obs_n_str in observing_night_strs:
        str_split = obs_n_str.split('-', maxsplit=3)
        night_str = '-'.join(str_split[:3])
        night_portion = str_split[-1]
        night_dt = datetime.strptime(night_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        midnight_dt = night_dt + (timedelta(days=1) - pd.Timedelta(nanoseconds=1))
        night_info.append((midnight_dt, night_portion))

    for night_idx in range(len(eval_metrics.keys())):
        observing_night_str = observing_night_strs[night_idx]
        night_dt = night_info[night_idx][0]
        date_str = f"{night_dt.year}-{night_dt.month}-{night_dt.day}"
        logger.info(f'Drawing plots for night {date_str}')
        night_dir = schedule_outdir / date_str
        if not os.path.exists(night_dir):
            os.makedirs(night_dir)

        metrics = eval_metrics[f'night-{night_idx}']
        if len(metrics['timestamp']) < 1:
            logger.info(f"Night {night_idx} had no viable observations")
            continue

        # Mask zenith observations in plotting
        real_obs_mask = np.array(metrics['field_id']) != ZENITH_FIELD_ID
        real_obs_mask &= np.array(metrics['field_id']) != WAIT_SIGNAL

        timestamps = metrics['timestamp']
        field_ids = metrics['field_id']
        bin_nums = metrics['bin']

        night_ts = night_dt.timestamp()
        sunset_time = math.ceil(calc_twilight(night_ts, 'set', env.unwrapped.horizon))
        timestamps = (timestamps - sunset_time) / 3600

        # Plot bins vs timestamp
        fig_b, axb = plt.subplots()
        axb.plot(timestamps[real_obs_mask],
                bin_nums[real_obs_mask],
                    marker='o', label='pred', alpha=.5)
        axb.legend()
        axb.set_xlabel('Hours since sunset \n (-10 deg)')
        axb.set_ylabel('bin')
        fig_b.suptitle(observing_night_str)
        fig_b.tight_layout()
        fig_b.savefig(night_dir / 'bin_vs_step.png')
        plt.close()

        # Plot state features vs timestamp for first episode
        fig, axs = plt.subplots(len(env.unwrapped.global_feature_names), figsize=(10, len(env.unwrapped.global_feature_names)*5))
        for i, feature_row in enumerate(np.array(metrics['glob_observations']).T[:len(env.unwrapped.global_feature_names)]):
            feat_name = env.unwrapped.global_feature_names[i]
            if feat_name == 'airmass':
                feature_row = 1 / feature_row
            elif 'dec' in feat_name or 'el' in feat_name:
                feature_row = feature_row * (np.pi/2)
            elif 'distance' in feat_name:
                feature_row = feature_row * np.pi

            axs[i].plot(timestamps[real_obs_mask], feature_row[real_obs_mask], label='policy roll out', marker='o')
            axs[i].set_title(feat_name)
            axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
            axs[i].legend()
        fig.tight_layout()
        fig.savefig(night_dir / 'state_features_vs_time.png')
        plt.close()

        # Plot most frequently visited bin features vs timestamp
        if action_architecture is not None:
            _bins_vis_tonight = np.array(bin_nums).astype(int)
            _bincounts = np.bincount(_bins_vis_tonight[real_obs_mask], minlength=num_actions)
            _most_common_bin = np.argmax(_bincounts)
            normed_feature_names = env.unwrapped.candidate_feature_names
            fig, axs = plt.subplots(len(normed_feature_names), figsize=(10, len(normed_feature_names)* 5))
            for i, feat_row in enumerate(np.array(metrics['candidate_observations']).T[:, _most_common_bin, :]):
                feat_name = normed_feature_names[i]
                # unnormalize observations to compare to expert values
                if feat_name == 'airmass':
                    feat_row = 1 / feat_row
                elif 'dec' in feat_name or 'el' in feat_name:
                    feat_row = feat_row * (np.pi/2)
                elif 'distance' in feat_name:
                    feat_row = feat_row * np.pi
                axs[i].plot(timestamps[real_obs_mask], feat_row[real_obs_mask], label='policy roll out', marker='o')
                axs[i].set_title(feat_name)
                axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
                axs[i].legend()
            fig.tight_layout()
            fig.savefig(night_dir / 'bin_features_vs_time.png')

        logger.info(f'Creating schedule gif for {night_idx}th night')
        save_nightly_schedule(night_metrics=metrics, save_dir=night_dir)
        if do_gifs:
            save_gifs(night_dir / "schedule.csv", night_dir, do_fieldbin=True, do_bin=False, do_mollefield=False, do_ortho=False, action_space=action_space, nside=nside,
                      fid2radec_filepath=lookup_dirpath / 'fid2radec.json')

        night_dt += timedelta(days=1)

def save_nightly_schedule(night_metrics, save_dir):
    timestamps = night_metrics['timestamp']
    bins = night_metrics['bin']
    fids = night_metrics['field_id']
    if len(timestamps) < 1:
        return

    real_obs_mask = (bins != ZENITH_BIN_NUM) & (bins != WAIT_SIGNAL)

    schedule_full = {
        'agent_timestamp': timestamps[real_obs_mask],
        'agent_field_id': fids[real_obs_mask],
        'agent_bin_id': bins[real_obs_mask],
    }

    df = pd.DataFrame(data={k: pd.Series(v) for k, v in schedule_full.items()}).fillna(0).astype(int)
    df.to_csv(save_dir / "schedule.csv", index=False)
