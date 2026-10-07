"""Orchestration: Evaluator hierarchy + factory.

`Evaluator` is the abstract base. `SingleStepEvaluator` runs one-step-ahead
inference; `MultiStepEvaluator` drives the offline runner.

`build_evaluators` wires everything together from a config path or object.
"""
from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops import rearrange
import logging
from datetime import date

from blancops.math import units
logger = logging.getLogger(__name__)

from collections import defaultdict

from blancops.configs.constants import FILTER2IDX, _NUM_FILTERS
from blancops.configs.paths import RunPaths, feature_cache_dir, field_feature_cache_dir, lookups_dir, resolve_data_dir
from blancops.ephemerides import ephemerides as _ephemerides
from blancops.math.interpolate import interpolate_on_sphere
from blancops.configs.experiment_schema import ActionConstraints, load_and_validate
from blancops.data.dataset import TransitionDataset, TransitionDatasetCache
from blancops.data.feature_cache import FieldFeatureCache, BinFeatureCache
from blancops.data.norm_stats import NormStats
from blancops.data.splits import NightSplit
from blancops.data.features.normalizations import build_normalizer
from blancops.data.lookup_tables import LookupTables, TrainLookupTables
from blancops.environment.historic_env import HistoricBlancoEnv
from blancops.rl.agent import filter_first_decode
from blancops.rl.agent_factory import AgentFactory
from blancops.rl.checkpointer import get_checkpoint
from blancops.rl.offline_runner import OfflineRunner
from blancops.telescope import get_telescope
from blancops.io.file_io import SCHEDULE_KEYS

from .data_container import (
    DataContainer,
    MultiStepDataContainer,
    SingleStepDataContainer,
)
from .plotters import FILTER_COLORS, EvaluationPlotter, PlotStyle
from blancops.configs.enums import grid_is_azel, has_filter, is_field_level


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def build_evaluators(
    cfg_or_cfg_path,
    device,
    eval_outdir: str = None,
    style: PlotStyle = None,
    save_movie=False,
    save_mollweide=False,
    plot_bins=False,
    data_dir=None,
    action_decoding='joint',
    split: str = 'val',
) -> Tuple['SingleStepEvaluator', 'MultiStepEvaluator']:
    """Build SS and MS evaluators for one split from a config.

    Args:
        cfg_or_cfg_path: An ExperimentConfig or a path to one.
        device: Torch device.
        eval_outdir: Output subdirectory name. Defaults to 'holdout_eval' for
            the val split and 'test_eval' for the test split.
        style: Plot style.
        save_movie: Whether the multi-step runner saves movies.
        save_mollweide: Whether the multi-step runner saves Mollweide frames.
        plot_bins: Whether movies of field-level models also draw HEALPix bins (bin-level models always do).
        data_dir: Override for the feature cache root.
        action_decoding: 'joint' or 'filter_first'.
        split: Which split to evaluate, 'val' or 'test'.

    Returns:
        The single-step and multi-step evaluators.
    """
    cfg = (
        load_and_validate(cfg_or_cfg_path)
        if isinstance(cfg_or_cfg_path, str)
        else cfg_or_cfg_path
    )
    style = style or PlotStyle(agent_label=f"{cfg.model.algorithm.name} Agent")

    # Resolve the model dir from where the config was loaded
    run_paths = RunPaths.from_config(cfg)
    outdir = run_paths.root
    eval_dir = run_paths.eval_dir(split) if eval_outdir is None else outdir / eval_outdir
    ss_outdir = eval_dir / 'ss'
    ms_outdir = eval_dir / 'ms'

    # Checkpoint + normalizers
    checkpoint = get_checkpoint(outdir, device=device)
    norm_stats = NormStats.from_dict(checkpoint['norm_stats'])

    telescope = get_telescope("blanco")

    # Load val dataset from cache or reconstruct from feature cache
    _data_dir = resolve_data_dir(data_dir if data_dir is not None else cfg.data.data_dir)
    lookups = TrainLookupTables.load_from_dir(lookups_dir(_data_dir), acceptance=cfg.data.acceptance)
    logger.info(f"Loaded lookups from {lookups_dir(_data_dir)} (acceptance '{cfg.data.acceptance}')")
    val_cache_path = run_paths.dataset_cache(split)
    field_level_cache = is_field_level(cfg.data.action_space)
    cache_cls = FieldFeatureCache if field_level_cache else BinFeatureCache
    cache_dir = (field_feature_cache_dir(_data_dir) if field_level_cache
                 else feature_cache_dir(_data_dir, cfg.data.nside, is_azel=grid_is_azel(cfg.data.action_space)))

    if TransitionDatasetCache.exists(val_cache_path):
        val_dataset = TransitionDatasetCache.load(val_cache_path)
    else:
        if not cache_cls.exists(cache_dir):
            raise FileNotFoundError(
                f"Neither {split} dataset cache ({val_cache_path}) nor feature cache "
                f"({cache_dir}) found."
            )
        split_json = run_paths.split_json
        if NightSplit.exists(split_json):
            split_nights = NightSplit.load(split_json).nights_for(split)
        else:
            split_nights = cfg.data.val_nights if split == 'val' else cfg.data.test_nights
        if not split_nights:
            raise ValueError(
                f"No {split} nights found in {split_json} or in the config; "
                f"cannot reconstruct the {split} dataset."
            )
        full_cache = (FieldFeatureCache.load(cache_dir, mmap=True, acceptance=cfg.data.acceptance)
                      if field_level_cache
                      else BinFeatureCache.load(cache_dir, mmap_bin=True, acceptance=cfg.data.acceptance))
        val_raw_cache = full_cache.filter_nights(split_nights)
        val_dataset = TransitionDataset(
            cache=val_raw_cache, cfg=cfg, lookups=lookups,
            norm_stats=norm_stats, split_role=split, telescope=telescope,
        )
        TransitionDatasetCache.from_transition_dataset(val_dataset, split=split).save(val_cache_path)

    # Build with the dataset's expanded names so filter-dependent features
    # (sky_brightness_g, urgency_r, ...) appear in active_features and can be inverted.
    global_normalizer = build_normalizer(
        state_feature_names=val_dataset.global_feature_names, cfg=cfg,
    )

    # Agent + runner
    factory = AgentFactory(base_model_dir=outdir)
    agent, cfg, _ = factory.build_agent(
        model_path_or_alias=outdir,
        lookups=lookups,
        field_choice_method='interp',
        device=device,
        action_decode=action_decoding
    )
    runner = OfflineRunner(
        agent=agent, policy=agent.policy, cfg=cfg,
        lookups=lookups, num_episodes=1, outdir=ms_outdir,
        save_SISPI=False, save_state_features=True,
        save_movie=save_movie, save_mollweide=save_mollweide, plot_bins=plot_bins,
    )

    # Environment for MS evaluator
    field_level = is_field_level(cfg.data.action_space)
    nightgroup = val_dataset._df.groupby('night')
    if not field_level:
        # Bin runs start each night at the expert's first field; field runs start at the zenith row.
        nightgroup = nightgroup.apply(lambda x: x.iloc[1:], include_groups=False).reset_index(level=0).reset_index(drop=True).groupby('night')

    night_start_candidate_states = None
    if cfg.data.candidate_state_dim > 0 and not field_level:
        cur = val_dataset._df.iloc[val_dataset.current_state_idxs].reset_index(drop=True)
        night_start_indices = cur.index[cur['field'] == 'zenith'].values + 1
        night_start_candidate_states = val_dataset._prenorm_candidate_states[night_start_indices].detach().numpy()

    env = HistoricBlancoEnv(
        cfg=cfg, constraints_cfg=ActionConstraints(), lookups=lookups,
        global_pd_nightgroup=nightgroup, night_start_candidate_states=night_start_candidate_states,
        norm_stats=norm_stats, telescope=telescope, zenith_start=field_level,
    )

    # Containers + plotters + evaluators
    action_space = cfg.data.action_space
    ss_data = SingleStepDataContainer(val_dataset, action_space, lookups,
                                     global_normalizer=global_normalizer)

    ms_data = MultiStepDataContainer(val_dataset, action_space, lookups, norm_stats=norm_stats,
                                     global_normalizer=global_normalizer)

    ss_plotter = EvaluationPlotter(ss_outdir, style=style)
    ms_plotter = EvaluationPlotter(ms_outdir, style=style)

    # Precompute per-sample visible-bin masks for filter_first single-step
    # decoding, aligned to the SS timestamps used for field placement.
    ss_visible_masks = None
    if action_decoding == 'filter_first' and is_field_level(action_space):
        raise ValueError("action_decoding 'filter_first' is not supported for action_space 'field_filter'.")
    if action_decoding == 'filter_first' and 'filter' in action_space:
        ss_ts = ss_data.expert_df['timestamp'].to_numpy(dtype=float)
        ss_visible_masks = np.stack(
            [env._compute_visible_bin_mask(ts) for ts in ss_ts]
        )

    ss_eval = SingleStepEvaluator(policy=agent.policy, data_container=ss_data,
                                  plotter=ss_plotter, device=device,
                                  action_decode=action_decoding,
                                  visible_bin_masks=ss_visible_masks)
    ms_eval = MultiStepEvaluator(runner=runner, env=env, policy=agent.policy,
                                 data_container=ms_data, plotter=ms_plotter, device=device)
    return ss_eval, ms_eval


# ----------------------------------------------------------------------
# Evaluator base
# ----------------------------------------------------------------------

class Evaluator(ABC):
    """Holds a policy, plotter, and data container to run inference.
    Can plot using convenience methods (wrapper around Plotter methods)
    or plot manually using data in self.data.expert_df and self.data.agent_df.
    """

    def __init__(self, policy, data_container: DataContainer,
                 plotter: EvaluationPlotter, device: str):
        self.policy = policy
        self.data = data_container
        self.plotter = plotter
        self.device = device

    @abstractmethod
    def run(self, *args, **kwargs) -> None: ...

    # ---- Diagnostics on the policy ----------------------------------

    def run_saliency_check(self):
        n = len(self.data.dataset.curr_compact_idxs)
        idx = np.random.randint(low=0, high=n)
        compact_idx = self.data.dataset.curr_compact_idxs[idx]

        x_glob = self.data.dataset.states[compact_idx].unsqueeze(0).to(self.device)
        x_cand = self.data.dataset.candidate_states[compact_idx].unsqueeze(0).to(self.device)
        self.policy.core_net.to(self.device)
        x_glob.requires_grad_(True)
        x_cand.requires_grad_(True)

        scores = self.policy.core_net(x_glob, x_cand)
        target = scores[0].max()
        self.policy.core_net.zero_grad()
        target.backward()

        cand_grads = x_cand.grad[0].abs().mean(dim=0)
        peak = cand_grads.max().item()
        for i, name in enumerate(self.data.dataset.candidate_feature_names):
            print(f'Feature: {name:30} | Gradient: {cand_grads[i].item() / peak:.6f}')

    # def plot_layer1_weights(self, ax=None):
    #     if ax is None:
    #         _, ax = plt.subplots(figsize=(20, 5))
    #     names = self.data.dataset.global_feature_names + self.data.dataset.candidate_feature_names
    #     weights = self.policy.core_net.net[0].weight.data.cpu().detach().numpy()
    #     means = weights.mean(axis=0)
    #     stds  = weights.std(axis=0)
    #     xs = np.arange(len(names))
    #     ax.errorbar(xs, means[:len(names)], yerr=stds[:len(names)], color='black', fmt='none')
    #     ax.scatter(xs, means[:len(names)], color='C0')
    #     ax.set_xticks(xs)
    #     ax.set_xticklabels(names, rotation=45)
    #     return ax


    # ---- Common plot pass-throughs ----------------------------------

    def plot_mollweide_res(self):
        expert_df, agent_df = self.data.expert_df, self.data.agent_df
        times = dict(expert_times=expert_df['timestamp'].to_numpy(), agent_times=agent_df['timestamp'].to_numpy())
        if not is_field_level(self.data.action_space):
            self.plotter.plot_mollweide_res(
                **times,
                expert_bin_idxs=expert_df['candidate_idx'].to_numpy(),
                agent_bin_idxs=agent_df['candidate_idx'].to_numpy(),
                nside=self.data.candidate_grid.nside,
            )
            return
        self.plotter.plot_mollweide_res(
            **times,
            expert_field_pos=self._field_radec(expert_df['candidate_idx'].to_numpy()),
            agent_field_pos=self._field_radec(agent_df['candidate_idx'].to_numpy()),
        )

    def _field_radec(self, field_ids):
        """Field centers in radians, NaN for the zenith sentinel (the DataFrame ra/dec columns are in degrees).

        Parameters
        ----------
        field_ids : np.ndarray
            Field id per observation, ZENITH_FIELD_ID (-1) for zenith rows.

        Returns
        -------
        np.ndarray
            (n_obs, 2) (ra, dec) in radians.
        """
        fields = self.data.lookups.fields
        ids = np.asarray(field_ids, dtype=int)                 # [n_obs]
        valid = ids >= 0
        pos = np.full((len(ids), 2), np.nan)                   # [n_obs, 2]
        pos[valid, 0] = fields['ra'].to_numpy()[ids[valid]]
        pos[valid, 1] = fields['dec'].to_numpy()[ids[valid]]
        return pos

    def plot_hist_comparison(self, feature_name, density=True, bins=20, use_weights=False, ax=None):
        return self.plotter.plot_hist_comparison(
            feature_name,
            expert_arr=self.data.expert_df[feature_name],
            agent_arr=self.data.agent_df[feature_name],
            density=density, bins=bins, use_weights=use_weights, ax=ax,
        )


    def plot_2dhist(self, feature_x: str, feature_y: str, bins=25, normalization='density'):
        return self.plotter.plot_2dhist(
            feature_x, feature_y,
            self.data.expert_df[feature_x].values, self.data.expert_df[feature_y].values,
            self.data.agent_df[feature_x].values,  self.data.agent_df[feature_y].values,
            bins=bins,
            normalization=normalization,
        )

    def plot_2dhist_res(self, feature_x, feature_y, bins=25, label_fontsize=20, normalization='counts'):
        return self.plotter.plot_2dhist_res(
            feature_x, feature_y,
            self.data.expert_df[feature_x], self.data.expert_df[feature_y],
            self.data.agent_df[feature_x],  self.data.agent_df[feature_y],
            bins=bins,
            label_fontsize=label_fontsize,
            normalization=normalization,
        )

    def plot_2dhist_per_filter(self, feature_x, feature_y, bins=25, normalization='density'):
        """Plot one expert/agent 2D histogram figure per filter.

        Args:
            feature_x: Column plotted on the x axis.
            feature_y: Column plotted on the y axis.
            bins: Number of bins per axis.
            normalization: One of 'counts', 'density' or 'probability', applied
                within each filter so panels are comparable across filters.

        Returns:
            Dict mapping filter name to the figure drawn for that filter.
        """
        figs = {}
        for filt in FILTER2IDX.keys():
            exp_f_mask = self.data.expert_df['filter'].values == filt
            agent_f_mask = self.data.agent_df['filter'].values == filt
            fig, _ = self.plotter.plot_2dhist(
                feature_x=feature_x,
                feature_y=feature_y,
                expert_x=self.data.expert_df[feature_x][exp_f_mask],
                expert_y=self.data.expert_df[feature_y][exp_f_mask],
                agent_x=self.data.agent_df[feature_x][agent_f_mask],
                agent_y=self.data.agent_df[feature_y][agent_f_mask],
                bins=bins,
                normalization=normalization
            )
            fig.suptitle(f'{filt}-band', fontsize=16)
            figs[filt] = fig
        return figs

    # ---- Plotter independent plots ----------------------------------

    def plot_scalar_metrics(self):
        expert_metrics = self._get_scalar_metrics_from_df(self.data.expert_df)
        agent_metrics = self._get_scalar_metrics_from_df(self.data.agent_df)

        metrics_keys = list(expert_metrics.keys())
        num_metrics = len(metrics_keys)

        fig, axs = plt.subplots(num_metrics, 1, figsize=(8, 2 * num_metrics), sharex=False)

        exp_y = 1
        ag_y = 2

        for i, metric in enumerate(metrics_keys):
            ax = axs[i]
            if not isinstance(expert_metrics[metric], dict):
                exp_val = expert_metrics[metric]
                ag_val = agent_metrics[metric]

                # Plot just the dots
                ax.plot(exp_val, exp_y, 'o', color=self.plotter.style.expert_color, markersize=8)
                ax.plot(ag_val, ag_y, 'o', color=self.plotter.style.agent_color, markersize=8)

                # Add padding for scalar values so they don't sit on the plot edges
                x_min_all = min(exp_val, ag_val)
                x_max_all = max(exp_val, ag_val)

                # Ensure there is visible padding even if the expert and agent values are identical
                spread = x_max_all - x_min_all
                padding = spread * 0.2 if spread > 0 else max(x_min_all * 0.1, 1.0)
                ax.set_xlim(x_min_all - padding, x_max_all + padding)

            else:
                # Extract Expert Data
                exp_mean = expert_metrics[metric]['mean']
                exp_p10 = expert_metrics[metric]['p10']
                exp_p90 = expert_metrics[metric]['p90']

                # Extract Agent Data
                ag_mean = agent_metrics[metric]['mean']
                ag_p10 = agent_metrics[metric]['p10']
                ag_p90 = agent_metrics[metric]['p90']

                # Plot Expert (y=1)
                ax.hlines(y=exp_y, xmin=exp_p10, xmax=exp_p90, color=self.plotter.style.expert_color, linewidth=2)
                ax.plot(exp_mean, exp_y, 'o', color=self.plotter.style.expert_color, markersize=8, label='Expert' if i==0 else "")

                # Plot Agent (y=2)
                ax.hlines(y=ag_y, xmin=ag_p10, xmax=ag_p90, color=self.plotter.style.agent_color, linewidth=2)
                ax.plot(ag_mean, ag_y, 'o', color=self.plotter.style.agent_color, markersize=8, label='Agent' if i==0 else "")

                # Add a little padding to the x-axis so dots don't hit the edges
                x_min_all = min(exp_p10, ag_p10)
                x_max_all = max(exp_p90, ag_p90)
                padding = (x_max_all - x_min_all) * 0.1
                ax.set_xlim(x_min_all - padding, x_max_all + padding)

            # Formatting
            ax.set_yticks([0, ag_y, exp_y, 3])
            ax.set_yticklabels([None, 'Agent', 'Expert', None])
            ax.set_title(metric, fontsize=14, loc='left')
            ax.grid(True, alpha=0.3)

        fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
        plt.tight_layout()
        return fig, axs

    def plot_violin_per_filter(self, key_metric='moon_el'):
        expert_df = self.data.expert_df.assign(source='Expert')
        agent_df  = self.data.agent_df.assign(source=self.plotter.style.agent_label)

        combined_df = pd.concat(
            [expert_df[[key_metric, 'filter', 'source']],
            agent_df[[key_metric, 'filter', 'source']]],
            ignore_index=True,
        )
        self.plotter.plot_violin_per_filter(combined_df, key_metric=key_metric)

    def plot_metric_distributions(self):
        metrics = ['airmass', 'slew_dist']

        expert_df = self.data.expert_df.copy()
        agent_df = self.data.agent_df.copy()
        for _df in (expert_df, agent_df):
            if 'ha' in _df.columns:
                _df['ha'] /= units.deg

        # Remove slew distances > 35 degrees (arbitrary cutoff) # XXX need to check train data construction
        expert_df['slew_dist'] = expert_df['slew_dist'].where(expert_df['slew_dist'] < 10, np.nan)
        agent_df['slew_dist'] = agent_df['slew_dist'].where(agent_df['slew_dist'] < 10, np.nan)

        expert_df = expert_df[metrics].assign(source='DES')
        agent_df  = agent_df[metrics].assign(source=self.plotter.style.agent_label)

        # 3. Combine into a single long-format DataFrame
        combined_df = pd.concat([expert_df, agent_df], ignore_index=True)

        # 4. Pass the combined data and metrics list to your plotter
        fig, axs = self.plotter._plot_metric_distributions(combined_df, metrics)

        return fig, axs


# ----------------------------------------------------------------------
# Single-step
# ----------------------------------------------------------------------

class SingleStepEvaluator(Evaluator):
    """One-step-ahead inference."""

    N_INFERENCE_SLICES = 8
    _INTERP_NEIGHBORS = 16  # RBF neighbors for interpolate_on_sphere; controls speed vs accuracy

    def __init__(self, policy, data_container: SingleStepDataContainer,
                 plotter: EvaluationPlotter, device: str = 'cuda',
                 field_choice_method: str = 'interp',
                 action_decode: str = 'joint', visible_bin_masks=None):
        super().__init__(policy, data_container, plotter, device)
        self.field_choice_method = field_choice_method
        self.action_decode = action_decode
        # [n_states, n_bins] bool, aligned to dataset.curr_compact_idxs order;
        # required for filter_first decoding, unused otherwise.
        self.visible_bin_masks = visible_bin_masks

    def run(self) -> None:
        agent_cand_idxs, agent_filter_idxs, agent_field_ids = self._batch_single_step()
        timestamps = self.data.expert_df['timestamp'].astype(int)

        self.data.populate_agent_df(agent_cand_idxs, agent_filter_idxs, timestamps,
                                    field_ids=agent_field_ids)
        # IMPORTANT: convert agent_df to deg BEFORE computing errors so both
        # dataframes share units inside populate_errors_df.
        self.data.convert_to_deg(self.data.agent_df)
        self.data.populate_errors_df()
        self.data.convert_to_deg(self.data.errors_df)

    def _batch_single_step(self):
        n_slices = self.N_INFERENCE_SLICES
        compact_idxs = self.data.dataset.curr_compact_idxs
        chunk = len(compact_idxs) // n_slices
        action_outputs = []
        score_outputs  = []
        ff_bin_outputs = []
        ff_filter_outputs = []

        dataset = self.data.dataset

        do_filter = has_filter(self.data.action_space)
        field_level = is_field_level(self.data.action_space)
        filter_first = (self.action_decode == 'filter_first' and do_filter
                        and self.visible_bin_masks is not None)
        need_scores = (self.field_choice_method == 'interp' and not field_level) or filter_first

        for i in range(n_slices):
            sl = slice(i * chunk, None if i == n_slices - 1 else (i + 1) * chunk)
            idxs = compact_idxs[sl]
            glob  = dataset.states[idxs].to(self.device)
            cands = dataset.candidate_states[idxs].to(self.device) if dataset.include_candidate_features else None
            masks = dataset.action_masks[idxs].to(self.device)
            with torch.no_grad():
                scores = self.policy.core_net(glob, cands) if need_scores else None
                if filter_first:
                    visible = torch.as_tensor(
                        self.visible_bin_masks[sl], device=self.device, dtype=torch.bool
                    )
                    b_idx, f_idx = filter_first_decode(scores, masks, visible, _NUM_FILTERS)
                    ff_bin_outputs.append(b_idx.cpu())
                    ff_filter_outputs.append(f_idx.cpu())
                else:
                    action_outputs.append(self.policy.select_action(glob, cands, masks))
                if self.field_choice_method == 'interp' and not field_level:
                    score_outputs.append(scores.cpu())

        if filter_first:
            cand_idxs   = torch.cat(ff_bin_outputs).numpy()
            filter_idxs = torch.cat(ff_filter_outputs).numpy()
        else:
            cand_idxs = torch.cat(action_outputs).cpu().detach().numpy()
            if do_filter:
                filter_idxs = cand_idxs % _NUM_FILTERS
                cand_idxs   = cand_idxs // _NUM_FILTERS
            else:
                filter_idxs = None

        # Log active-candidate coverage diagnostic (fraction of candidates with no sentinel features)
        active_candidate_mask = getattr(dataset, 'active_candidate_mask', None)
        if active_candidate_mask is not None:
            logger.debug(
                f"Active candidate coverage: {float(active_candidate_mask.float().mean()):.3f}"
            )

        if field_level:
            # Candidates are fields: the chosen candidate index is the field id.
            return cand_idxs, filter_idxs, cand_idxs
        if self.field_choice_method != 'interp':
            return cand_idxs, filter_idxs, None

        field_ids = self._choose_fields_interp(
            cand_idxs, filter_idxs, torch.cat(score_outputs).numpy(),
        )
        return cand_idxs, filter_idxs, field_ids

    def _choose_fields_interp(self, bin_idxs, filter_idxs, all_scores):
        """For each chosen bin, pick the best field via Q-value interpolation."""
        n_bins    = self.data.dataset.n_candidates
        n_filters = all_scores.shape[-1] // n_bins
        lon_data  = self.data.candidate_grid.lon
        lat_data  = self.data.candidate_grid.lat
        is_azel   = self.data.candidate_grid.is_azel

        fids_all = self.data.lookups.fields.index.values
        ra_all   = self.data.lookups.fields['ra'].values
        dec_all  = self.data.lookups.fields['dec'].values
        timestamps = self.data.expert_df['timestamp'].values

        _filter_idxs = filter_idxs if filter_idxs is not None else np.zeros(len(bin_idxs), dtype=int)

        # Precompute static bin→fields map for equatorial grids.
        static_map = None
        if not is_azel:
            bids = self.data.candidate_grid.ang2idx(lon=ra_all, lat=dec_all)
            static_map = defaultdict(list)
            for fid, bid in zip(fids_all, bids):
                static_map[int(bid)].append(int(fid))

        field_ids_out = np.empty(len(bin_idxs), dtype=int)

        for j, (bid, filt_idx, q_row, ts) in enumerate(
            zip(bin_idxs, _filter_idxs, all_scores, timestamps)
        ):
            q_map = q_row.reshape(n_bins, n_filters)[:, int(filt_idx)]

            if is_azel:
                # Project all field RA/Dec to az/el at this observation's timestamp.
                az_all, el_all = _ephemerides.equatorial_to_topographic(ra_all, dec_all, time=float(ts))
                bids_j = self.data.candidate_grid.ang2idx(lon=az_all, lat=el_all)
                bin_map_j = defaultdict(list)
                for fid, b in zip(fids_all, bids_j):
                    if b is not None:  # field not observable at this timestamp
                        bin_map_j[int(b)].append(int(fid))
                valid_fids = bin_map_j.get(int(bid), [])
            else:
                az_all = el_all = None
                valid_fids = static_map.get(int(bid), [])

            if not valid_fids:
                # Fallback: nearest field to chosen bin center by angular distance.
                blon, blat = lon_data[int(bid)], lat_data[int(bid)]
                lons_f = az_all if is_azel else ra_all
                lats_f = el_all if is_azel else dec_all
                dists = (lons_f - blon) ** 2 + (lats_f - blat) ** 2
                field_ids_out[j] = int(fids_all[np.argmin(dists)])
                continue

            target_lons = (az_all if is_azel else ra_all)[np.array(valid_fids)]
            target_lats = (el_all if is_azel else dec_all)[np.array(valid_fids)]

            q_interp = interpolate_on_sphere(
                az=target_lons, el=target_lats,
                az_data=lon_data, el_data=lat_data,
                values=q_map, neighbors=self._INTERP_NEIGHBORS,
            )
            field_ids_out[j] = valid_fids[int(np.argmax(q_interp))]

        return field_ids_out

    # ---- Convenience pass-throughs ----------------------------------

    def plot_line_comparison(self, feature_name, ax=None):
        self._check_input_features(feature_name, None)
        return self.plotter.plot_line_comparison(
            feature_name, self.data.expert_df[feature_name].values,
            self.data.agent_df[feature_name], ax=ax,
        )

    def plot_scatter_comparison(self, feature_y, feature_x=None, ax=None):
        self._check_input_features(feature_x, feature_y)
        expert_y = self.data.expert_df[feature_y].values
        agent_y  = self.data.agent_df[feature_y].values
        if feature_x is None:
            expert_x = np.arange(len(expert_y))
            agent_x  = np.arange(len(agent_y))
        else:
            expert_x = self.data.expert_df[feature_x].values
            agent_x  = self.data.agent_df[feature_x].values
        return self.plotter.plot_scatter_comparison(
            feature_y, expert_x, expert_y, agent_x, agent_y,
            feature_x=feature_x, ax=ax,
        )

    def plot_residual(self, feature_y, feature_x=None, plot_type='hist', bins=20, alpha=0.2,
                      density=True, ax=None):
        return self.plotter.plot_residual(
            feature_y,
            self.data.expert_df[feature_y], self.data.agent_df[feature_y],
            feature_x=feature_x,
            expert_x=self.data.expert_df.get(feature_x, None),
            agent_x=self.data.agent_df.get(feature_x, None),
            plot_type=plot_type, bins=bins, alpha=alpha, density=density, ax=ax,
        )

    def plot_cdf_pointing_error(self, per_filter=False, use_candidate=False):
        return self.plotter.plot_cdf_pointing_error(self.data.expert_df, self.data.errors_df, per_filter=per_filter,
                                                    use_candidate=use_candidate)

    def plot_quiver(self, feature_x, feature_y, ax=None):
        self._check_input_features(feature_x, feature_y)
        # prev_expert_df holds the PREVIOUS state; expert_df holds the NEXT state.
        return self.plotter.plot_quiver(
            feature_x, feature_y,
            expert_prev_x=self.data.prev_expert_df[feature_x],
            expert_prev_y=self.data.prev_expert_df[feature_y],
            expert_next_x=self.data.expert_df[feature_x],
            expert_next_y=self.data.expert_df[feature_y],
            agent_next_x=self.data.agent_df[feature_x],
            agent_next_y=self.data.agent_df[feature_y],
            ax=ax,
        )

    def plot_hist_comparison_per_filter(self, feature_name, bins=20):
        return self.plotter.plot_hist_comparison_per_filter(
            feature_name,
            expert_feature_arr=self.data.expert_df[feature_name].values,
            expert_filters=self.data.expert_df['filter'].values,
            agent_filters=self.data.agent_df['filter'].values,
            bins=bins,
        )


    def calculate_filter_confusion(self) -> np.ndarray:
        n = len(FILTER2IDX)
        conf_mat = np.zeros((n, n))
        for filt, idx in FILTER2IDX.items():
            mask = self.data.expert_df['filter_idx'].values == idx
            ag = self.data.agent_df['filter'].values[mask]
            total = mask.sum()
            if total == 0:
                continue
            for _fname, _fidx in FILTER2IDX.items():
                conf_mat[idx, _fidx] = (ag == _fname).sum() / total
        return conf_mat

    def plot_filter_confusion(self, ax=None):
        return self.plotter.plot_filter_confusion(self.calculate_filter_confusion(), ax=ax)

    def _check_input_features(self, feature_x, feature_y):
        assert feature_x in self.data.agent_df.columns, f"{feature_x} not in agent_df"
        assert feature_y in self.data.agent_df.columns, f"{feature_y} not in agent_df"

# ----------------------------------------------------------------------
# Multi-step
# ----------------------------------------------------------------------

class MultiStepEvaluator(Evaluator):
    """Whole-episode rollout via the offline runner."""

    def __init__(self, runner: OfflineRunner, env, policy,
                 data_container: MultiStepDataContainer,
                 plotter: EvaluationPlotter, device: str = 'cuda'):
        super().__init__(policy, data_container, plotter, device)
        self.runner = runner
        self.env = env
        self.eval_metrics = None
        self.outdir: Path = Path(runner.outdir)

    def run(self, outdir: Union[str, Path, None] = None, overwrite=False) -> None:
        self.outdir = Path(outdir or self.runner.outdir)
        metrics_path = self.outdir / 'eval_metrics.pkl'

        if metrics_path.exists() and not overwrite:
            with open(metrics_path, 'rb') as f:
                self.eval_metrics = pickle.load(f)
                logger.info(f"Results already exist in {metrics_path}. \
                            Pass overwrite=True to re-run.")
            if self.runner.save_movie:
                self._save_missing_movies()
        else:
            self.eval_metrics = self.runner.run(env=self.env)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'wb') as f:
                pickle.dump(self.eval_metrics, f)

        ts, bin_idxs, filter_idxs, field_ids, glob_df, candidate_feat_dict = \
            self._process_eval_metrics(self.eval_metrics)

        # Field candidates are indexed by field id; the schedule's bin is grid bookkeeping only.
        candidate_idxs = field_ids if is_field_level(self.data.action_space) else bin_idxs
        self.data.populate_agent_df(
            candidate_idxs=candidate_idxs, filter_idxs=filter_idxs, timestamps=ts,
            field_ids=field_ids, glob_df=glob_df, candidate_feat_dict=candidate_feat_dict,
        )

    def _process_eval_metrics(self, eval_metrics):
        # Deterministic eval: single episode.
        episode = eval_metrics['ep-0']

        # Detect format. The streaming runner stores {'manifest': {...}, ...};
        # the legacy in-memory runner stores {night_key: {arr_dict}, ...}.
        if isinstance(episode, dict) and 'manifest' in episode:
            return self._process_eval_metrics_from_manifest(episode['manifest'])
        return self._process_eval_metrics_from_memory(episode)

    def _process_eval_metrics_from_memory(self, episode):
        """Legacy path: per-night arrays already in RAM.

        Memory note: the original implementation built `bin_feats` (a list of
        rearranged per-night views), concatenated it into a single full-size
        array, AND held the per-feature dict simultaneously — ~3x peak.
        Here we concatenate per feature and drop intermediates as we go.
        """
        # Build the small flat columns first; these are cheap.
        night_keys = list(episode.keys())
        bin_arr = np.concatenate([np.asarray(episode[n]['bin']) for n in night_keys])
        field_arr = np.concatenate([np.asarray(episode[n]['field_id']) for n in night_keys])
        ts_arr = np.concatenate([np.asarray(episode[n]['timestamp']) for n in night_keys])
        filter_arr = np.concatenate([np.asarray(episode[n]['filter_idx']) for n in night_keys])
        night_col = np.concatenate(
            [np.full(len(episode[n]['bin']), n) for n in night_keys]
        )

        # Drop zenith/wait states.
        valid = (bin_arr != -1) & (bin_arr != -2)

        # Global features: build the DataFrame once from a single concatenated
        # array rather than concat-of-per-night-DataFrames.
        glob_arr = np.concatenate(
            [np.asarray(episode[n]['glob_observations'], dtype=np.float32)
             for n in night_keys],
            axis=0,
        )
        glob_df = pd.DataFrame(glob_arr, columns=self.data.dataset.global_feature_names)
        glob_df['night'] = night_col
        del glob_arr

        # Candidate features: stream per feature to avoid holding a full
        # (nfeats, total_rows, n_candidates) intermediate AND a full per-feature
        # dict at the same time.
        feat_names = self.data.dataset.candidate_feature_names
        candidate_feat_dict = {}
        # Materialize the full (total_rows, nbins, nfeats) array once in float32
        # — we still need it, but at least at fp32 not fp64.
        bin_obs_all = np.concatenate(
            [np.asarray(episode[n]['bin_observations'], dtype=np.float32)
             for n in night_keys],
            axis=0,
        )
        # Drop the per-night references from the upstream dict to free
        # ~50% of duplicated memory before slicing.
        for n in night_keys:
            episode[n].pop('bin_observations', None)
        import gc as _gc
        _gc.collect()

        for i, name in enumerate(feat_names):
            # arr[valid, :, i] copies; assign and move on.
            candidate_feat_dict[name] = bin_obs_all[valid, :, i].copy()
        del bin_obs_all
        _gc.collect()

        return (
            ts_arr[valid],
            bin_arr[valid],
            filter_arr[valid],
            field_arr[valid],
            glob_df[valid],
            candidate_feat_dict,
        )

    def _process_eval_metrics_from_manifest(self, manifest):
        """Streaming path: scalar columns from per-night CSVs, obs features
        from companion ``_obs.npz`` files written by the runner.

        CSVs are pre-filtered (real observations only, no zenith/wait rows),
        so no valid mask is needed. The companion .npz files contain the same
        rows in the same order (the runner accumulates obs features only for
        real observations since save_state_features=True is set in build_evaluators).
        """
        import gc as _gc

        night_keys = [k for k in manifest if manifest[k] is not None]

        # The manifest may store absolute paths from the training machine; the
        # night CSV/npz files always live in <outdir>/nights with these
        # basenames, so resolve by basename to stay portable across machines.
        nights_dir = self.outdir / 'nights'
        manifest = {k: (nights_dir / Path(v).name) if v is not None else None
                    for k, v in manifest.items()}

        # ---- Pass 1: scalars from per-night CSVs ----
        frames = []
        for n in night_keys:
            df = pd.read_csv(manifest[n])
            df['_night_key'] = n
            frames.append(df)

        full_df    = pd.concat(frames, ignore_index=True)
        ts_arr     = full_df[SCHEDULE_KEYS['timestamp']].values
        bin_arr    = full_df[SCHEDULE_KEYS['bin_id']].values
        filter_arr = full_df[SCHEDULE_KEYS['filter_idx']].values
        field_arr  = full_df[SCHEDULE_KEYS['field_id']].values
        night_col  = full_df['_night_key'].values
        n_rows     = len(ts_arr)
        del frames, full_df

        feat_names_glob = self.data.dataset.global_feature_names
        feat_names_cand = self.data.dataset.candidate_feature_names

        # ---- Pass 2: obs features from companion _obs.npz files ----
        # Path convention: nights/ep-N_night-K.csv → nights/ep-N_night-K_obs.npz
        first_npz = Path(manifest[night_keys[0]])
        first_npz = first_npz.parent / (first_npz.stem + '_obs.npz')

        # ---- Pass 2a: glob_observations (single concat) ----
        glob_parts = []
        for n in night_keys:
            npz_path = Path(manifest[n])
            npz_path = npz_path.parent / (npz_path.stem + '_obs.npz')
            with np.load(npz_path) as npz:
                glob_parts.append(np.asarray(npz['glob_observations'], dtype=np.float32))
        glob_arr = np.concatenate(glob_parts, axis=0)
        del glob_parts
        glob_df = pd.DataFrame(glob_arr, columns=feat_names_glob)
        glob_df['night'] = night_col
        del glob_arr

        # ---- Pass 2b: candidate_observations, streamed per night ----
        # Runs written before the bin -> candidate rename store them as bin_observations
        with np.load(first_npz, mmap_mode='r') as npz:
            cand_key = 'candidate_observations' if 'candidate_observations' in npz else 'bin_observations'
            n_candidates = npz[cand_key].shape[1]
        candidate_feat_dict = {
            name: np.empty((n_rows, n_candidates), dtype=np.float32)
            for name in feat_names_cand
        }
        row_offset = 0
        for n in night_keys:
            npz_path = Path(manifest[n])
            npz_path = npz_path.parent / (npz_path.stem + '_obs.npz')
            with np.load(npz_path) as npz:
                cand_obs = np.asarray(npz[cand_key], dtype=np.float32)
            n_night = cand_obs.shape[0]
            for i, name in enumerate(feat_names_cand):
                candidate_feat_dict[name][row_offset:row_offset + n_night] = cand_obs[:, :, i]
            del cand_obs
            row_offset += n_night
            _gc.collect()

        return ts_arr, bin_arr, filter_arr, field_arr, glob_df, candidate_feat_dict

    # ---- MS-specific plots ------------------------------------------

    def plot_kde_and_scatter(self, feature_x, feature_y, agent_alpha=0.2,
                             use_filter_coloring=True, s=5, ax=None):
        color_mapping = (
            self.data.agent_df['filter'].map(FILTER_COLORS)
            if use_filter_coloring else None
        )
        return self.plotter.plot_kde_and_scatter(
            feature_x, feature_y,
            self.data.expert_df[feature_x], self.data.expert_df[feature_y],
            self.data.agent_df[feature_x],  self.data.agent_df[feature_y],
            agent_alpha=agent_alpha, color_mapping=color_mapping, s=s, ax=ax,
        )

    def plot_hist_comparison_per_filter(self, feature_name, bins=20):
        return self.plotter.plot_hist_comparison_per_filter(
            feature_name,
            expert_feature_arr=self.data.expert_df[feature_name].values,
            agent_feature_arr=self.data.agent_df[feature_name].values,
            expert_filters=self.data.expert_df['filter'].values,
            agent_filters=self.data.agent_df['filter'].values,
            bins=bins,
        )

    def plot_hist_comparison(self, feature_name, density=True, bins=20, use_weights=False, ax=None):
        # MS expert_df has gaps masked via expert_valid_mask.
        expert_arr = self.data.expert_df[feature_name][self.data.expert_valid_mask]
        return self.plotter.plot_hist_comparison(
            feature_name, expert_arr=expert_arr,
            agent_arr=self.data.agent_df[feature_name],
            density=density, bins=bins, use_weights=use_weights, ax=ax,
        )

    def _calculate_performance_metrics(self):
        """Placeholder for episode-level metrics (reward, coverage, etc.)."""
        metrics = {}

        pass


# ----------------------------------------------------------------------
# Cross-evaluator plots
# ----------------------------------------------------------------------

def plot_metric_distributions_with_ss_overlay(
    ms_evaluator: 'MultiStepEvaluator',
    ss_evaluator: 'SingleStepEvaluator',
):
    """Plot MS metric distributions, overlaying SS distributions as black dashed lines (no fill).

    Metrics absent from either SS dataframe are skipped on the overlay (e.g.
    'ha' is not populated on the SS agent_df by default).
    """
    import seaborn as sns
    from matplotlib.lines import Line2D

    fig, axs = ms_evaluator.plot_metric_distributions()

    metrics = ['airmass', 'slew_dist']
    ss_agent_df  = ss_evaluator.data.agent_df.copy()

    if 'ha' in ss_agent_df.columns:
        ss_agent_df['ha'] = ss_agent_df['ha'] / units.deg
    if 'slew_dist' in ss_agent_df.columns:
        ss_agent_df['slew_dist']  = ss_agent_df['slew_dist'].where(ss_agent_df['slew_dist']  < 10, np.nan)
    SS_COLOR = 'black'
    SS_STYLE = 'solid'
    SS_LW = 2
    for ax, metric in zip(axs, metrics):
        if metric not in ss_agent_df.columns or ss_agent_df[metric].dropna().empty:
            continue
        sns.kdeplot(
            data=ss_agent_df, x=metric,
            color=SS_COLOR, linestyle=SS_STYLE, fill=False, linewidth=SS_LW,
            cut=0, bw_adjust=0.8, ax=ax,
        )
        # ax.axvline(
        #     ss_agent_df[metric].median(),
        #     color=SS_COLOR, linestyle='--', linewidth=1.0, alpha=0.8,
        # )
        ax.set_xlabel('')  # Keeping x-axis clear as the metric title explains the values


    existing = axs[0].get_legend()
    handles = list(existing.legend_handles) if existing else []
    handles.append(Line2D([0], [0], color=SS_COLOR, linestyle=SS_STYLE, linewidth=SS_LW, label='Single Step'))
    axs[0].legend(handles=handles, fontsize=20*(3/4), framealpha=0.9, loc='upper right')

    return fig, axs
