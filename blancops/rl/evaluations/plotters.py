"""Plotting utilities for comparing expert vs agent behavior.

`EvaluationPlotter` methods accept an optional `ax` and return the matplotlib
artist(s) they create, so plots are composable into subplot grids.

All style choices live in `PlotStyle`. Pass one instance everywhere instead of
threading six color/cmap arguments through every constructor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from matplotlib import patches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from blancops.configs.constants import FILTER2IDX
from blancops.plotting.plotting import plot_schedule_whole
from blancops.rl.evaluations.data_container import _ANGLE_TOKENS


import logging
logger = logging.getLogger(__name__)

FILTER_COLORS = {
    'g': '#00b25d',  # Green
    'r': '#f97306',  # Orange
    'i': '#e50000',  # Red
    'z': '#8b0000',  # Dark Red
    'Y': '#3b0000',  # Near-IR (Darkest)
}
FILTER_PATCHES = [Patch(color=v, label=k) for k, v in FILTER_COLORS.items()]


@dataclass(frozen=True)
class PlotStyle:
    agent_color: str = 'purple'
    agent_cmap: str = 'Purples'
    expert_color: str = 'green'
    expert_cmap: str = 'Greens'
    res_cmap: str = 'PRGn_r'
    res_color: str = 'slateblue'


def _wrapped_ra(ra):
    return (ra + 180) % 360 - 180


def _wrap_if_ra(feature_name: Optional[str], arr):
    """Wrap RA-like features to [-180, 180]. Leaves everything else untouched."""
    if feature_name is None or arr is None:
        return arr
    if feature_name == 'ra' or '_ra' in feature_name:
        return _wrapped_ra(arr)
    return arr


class EvaluationPlotter:
    def __init__(self, outdir, style: Optional[PlotStyle] = None):
        self.outdir = Path(outdir)
        self.style = style or PlotStyle()

    # ------------------------------------------------------------------
    # 2D histograms
    # ------------------------------------------------------------------

    def plot_2dhist(self, feature_x, feature_y, expert_x, expert_y, agent_x, agent_y,
                    norm=None, bins=25, label_fontsize=20, return_plt_objects=False,
                    normalization='density'):
        """Plot expert and agent 2D histograms side-by-side.

        Args:
            feature_x: Column plotted on the x axis.
            feature_y: Column plotted on the y axis.
            expert_x: Expert x values.
            expert_y: Expert y values.
            agent_x: Agent x values.
            agent_y: Agent y values.
            norm: Colour normalization; defaults to a log norm.
            bins: Number of bins per axis.
            label_fontsize: Unused; kept for signature parity with the other plots.
            return_plt_objects: Whether to also return the two histogram arrays.
            normalization: One of 'counts', 'density' or 'probability'. 'density'
                divides by the total count and the bin area; 'probability'
                divides by the total count only, so cells read as the fraction
                of that panel's observations.

        Returns:
            The figure and axes, plus the expert and agent histograms when
            ``return_plt_objects`` is set.
        """
        if norm is None:
            norm = mcolors.LogNorm()
        density = normalization == 'density'
        cbar_label = {
            'counts': 'Counts',
            'density': 'Density',
            'probability': 'Fraction of observations',
        }[normalization]
        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)

        x_min = min(np.min(expert_x), np.min(agent_x))
        x_max = max(np.max(expert_x), np.max(agent_x))
        y_min = min(np.min(expert_y), np.min(agent_y))
        y_max = max(np.max(expert_y), np.max(agent_y))


        x_edges = np.linspace(x_min, x_max, bins + 1)
        y_edges = np.linspace(y_min, y_max, bins + 1)

        bins=[x_edges, y_edges]

        use_weights = normalization == 'probability'
        exp_weights = np.ones_like(expert_x) / len(expert_x) if use_weights else None
        ag_weights  = np.ones_like(agent_x)  / len(agent_x)  if use_weights else None

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        exp_counts, _, _, im1 = axs[0].hist2d(expert_x, expert_y, bins=bins,
                                              cmap=self.style.expert_cmap, norm=norm,
                                              density=density, weights=exp_weights)
        fig.colorbar(im1, ax=axs[0], location='right', label=cbar_label)
        axs[0].set(xlabel=feature_x, ylabel=feature_y, title='Expert')

        ag_counts, _, _, im2 = axs[1].hist2d(agent_x, agent_y, bins=bins,
                                             cmap=self.style.agent_cmap, norm=norm,
                                             density=density, weights=ag_weights)
        fig.colorbar(im2, ax=axs[1], location='right', label=cbar_label)
        axs[1].set(xlabel=feature_x, ylabel=feature_y, title='Agent')

        if return_plt_objects:
            return fig, axs, exp_counts, ag_counts
        return fig, axs

    def plot_2dhist_res(self, feature_x, feature_y, expert_x, expert_y, agent_x, agent_y,
                        bins=25, label_fontsize=20, return_plt_objects=False,
                        normalization='counts', ax=None):

        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)

        xmin = min(expert_x.min(), agent_x.min())
        xmax = max(expert_x.max(), agent_x.max())
        ymin = min(expert_y.min(), agent_y.min())
        ymax = max(expert_y.max(), agent_y.max())
        hrange = [[xmin, xmax], [ymin, ymax]]
        nbins = [bins, bins]

        if normalization == 'density':
            exp_hist, _, _ = np.histogram2d(expert_x, expert_y, bins=nbins, range=hrange, density=True)
            agent_hist, _, _ = np.histogram2d(agent_x, agent_y, bins=nbins, range=hrange, density=True)
            cbar_label = 'Residual density\n(agent - expert)'

        elif normalization == 'probability':
            exp_hist, _, _ = np.histogram2d(expert_x, expert_y, bins=nbins, range=hrange, density=False)
            agent_hist, _, _ = np.histogram2d(agent_x, agent_y, bins=nbins, range=hrange, density=False)

            exp_hist = exp_hist / len(expert_x)
            agent_hist = agent_hist / len(agent_x)
            # Updated label to explicitly state percentage
            cbar_label = 'Residual percentage\n(agent - expert)'

        else: # 'counts'
            exp_hist, _, _ = np.histogram2d(expert_x, expert_y, bins=nbins, range=hrange, density=False)
            agent_hist, _, _ = np.histogram2d(agent_x, agent_y, bins=nbins, range=hrange, density=False)
            cbar_label = 'Residual counts\n(agent - expert)'

        res = agent_hist - exp_hist
        lim = np.max(np.abs(res))

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure
        im = ax.imshow(res.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
                       cmap=self.style.res_cmap, aspect='auto', vmin=-lim, vmax=lim)
        unit_x = 'deg' if feature_x in _ANGLE_TOKENS else ''
        unit_y = 'deg' if feature_y in _ANGLE_TOKENS else ''

        ax.set_xlabel(feature_x + ' (' + unit_x + ')', fontsize=label_fontsize)
        ax.set_ylabel(feature_y + ' (' + unit_y + ')', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=label_fontsize*(3/4))

        cbar_label = 'Residual Relative Density \n(agent - expert)' if normalization == 'probability' else cbar_label
        cbar = fig.colorbar(im, ax=ax, label=cbar_label)

        # Format ticks as percentages if normalization is 'probability'
        if normalization == 'probability':
            from matplotlib.ticker import PercentFormatter
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

        # Adjust the font sizes for colorbar elements
        cbar.ax.tick_params(labelsize=label_fontsize * (3/4))   # Scale ticks to match plot ticks
        cbar.set_label(cbar_label, fontsize=label_fontsize)      # Match main label font size

        cbar.ax.tick_params(labelsize=label_fontsize * (3/4))
        cbar.set_label(cbar_label, fontsize=label_fontsize, labelpad=20)


        if return_plt_objects:
            return fig, ax, exp_hist, agent_hist
        return fig, ax


    # ------------------------------------------------------------------
    # Mollweide / line / scatter / hist / residual
    # ------------------------------------------------------------------

    def plot_mollweide_res(self, timestamps, expert_bin_idxs, agent_bin_idxs, field_pos, nside):
        plot_schedule_whole(
            outfile=self.outdir / 'mollweide_residuals',
            times=timestamps,
            field_pos=None,
            bin_idxs=expert_bin_idxs,
            alternate_bin_idxs=agent_bin_idxs,
            nside=nside,
            sky_bin_mapping=None,
            projection='mollweide',
            center_pos=(None, None),
            schedule_label='',
        )

    def plot_line_comparison(self, feature_name, expert_arr, agent_arr, ax=None):
        ax = ax or plt.gca()
        expert_arr = _wrap_if_ra(feature_name, expert_arr)
        agent_arr  = _wrap_if_ra(feature_name, agent_arr)
        ax.plot(expert_arr, label='expert', color=self.style.expert_color)
        ax.plot(agent_arr, label='agent', color=self.style.agent_color)
        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel(feature_name, fontsize=16)
        ax.legend(fontsize=16)
        return ax

    def plot_scatter_comparison(self, feature_y, expert_x, expert_y, agent_x, agent_y,
                                feature_x=None, ax=None):
        ax = ax or plt.gca()
        ax.scatter(expert_x, expert_y, label='expert', color=self.style.expert_color)
        ax.scatter(agent_x,  agent_y,  label='agent',  color=self.style.agent_color)
        if feature_x is not None:
            ax.set_xlabel(feature_x, fontsize=16)
        ax.set_ylabel(feature_y, fontsize=16)
        ax.legend(fontsize=16)
        return ax

    def plot_hist_comparison(self, feature_name, expert_arr, agent_arr,
                             density=False, bins=20, alpha=0.2, use_weights=False, ax=None):
        ax = ax or plt.gca()
        if isinstance(bins, int) and feature_name != 'filter':
            lo = min(np.min(expert_arr), np.min(agent_arr))
            hi = max(np.max(expert_arr), np.max(agent_arr))
            shared_bins = np.linspace(lo, hi, bins + 1)
        else:
            shared_bins = bins

        ag_weights  = np.ones_like(agent_arr)  / len(agent_arr)  if use_weights else None
        exp_weights = np.ones_like(expert_arr) / len(expert_arr) if use_weights else None
        if use_weights:
            density = False

        # Outlines
        ax.hist(expert_arr, bins=shared_bins, density=density, histtype='step',
                color=self.style.expert_color, weights=exp_weights)
        ax.hist(agent_arr,  bins=shared_bins, density=density, histtype='step',
                color=self.style.agent_color, weights=ag_weights)
        # Fills
        ax.hist(expert_arr, bins=shared_bins, label='expert', histtype='stepfilled',
                color=self.style.expert_color, alpha=alpha, density=density, weights=exp_weights)
        ax.hist(agent_arr,  bins=shared_bins, label='agent',  histtype='stepfilled',
                color=self.style.agent_color,  alpha=alpha, density=density, weights=ag_weights)

        ax.set_xlabel(feature_name, fontsize=16)
        ylabel = 'Probability Density' if density else ('Fraction of total' if use_weights else 'Raw counts')
        ax.set_ylabel(ylabel, fontsize=16)
        ax.legend(fontsize=16)
        return ax

    def plot_residual(self, feature_y, expert_y, agent_y, feature_x=None,
                      expert_x=None, agent_x=None, plot_type='hist',
                      bins=20, alpha=0.2, density=False, ax=None):
        assert plot_type in ('hist', 'line')
        ax = ax or plt.gca()

        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)

        res_y = np.asarray(agent_y) - np.asarray(expert_y)
        if expert_x is None:
            expert_x = np.arange(len(res_y))

        if plot_type == 'hist':
            ax.hist(res_y, bins=bins, density=density, histtype='step',       color=self.style.res_color)
            ax.hist(res_y, bins=bins, density=density, histtype='stepfilled', color=self.style.res_color, alpha=alpha)
            ax.set_xlabel(feature_y)
            ax.set_ylabel('Normalized (residual) counts' if density else 'Raw (residual) counts', fontsize=16)
        else:  # line
            ax.plot(expert_x, res_y, color=self.style.res_color, marker='o')
        return ax

    # ------------------------------------------------------------------
    # Filter-specific plots
    # ------------------------------------------------------------------

    def plot_filter_confusion(self, conf_mat, ax=None, label_fontsize=20):
        # Slightly enlarged figure size to accommodate the square aspect ratio and colorbar label
        FIG_SIZE = (6.0, 5.0)

        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZE)

        sns.heatmap(conf_mat,
                    annot=True,
                    fmt=".2f",           # Limits annotations to 2 decimal places
                    cmap=self.style.agent_cmap,
                    xticklabels=FILTER2IDX.keys(),
                    yticklabels=FILTER2IDX.keys(),
                    ax=ax,
                    square=True,         # Forces cells to be perfectly square
                    cbar_kws={'label': 'Fraction of Observations'}, # Adds context to the colorbar
                    annot_kws={"size": label_fontsize*(3/4)}
                    )

        ax.set_xlabel('Agent', fontsize=label_fontsize)
        ax.set_ylabel('Expert', fontsize=label_fontsize)

        # Ensures y-tick labels are horizontal and easy to read
        ax.tick_params(axis='x', labelsize=label_fontsize*(3/4))
        ax.tick_params(axis='y', labelsize=label_fontsize*(3/4), labelrotation=0)

        # --- Colorbar Formatting ---
        cbar = ax.collections[0].colorbar

        cbar.set_label('Fraction of Observations', size=label_fontsize*(3/4))

        # Set the colorbar tick font size
        cbar.ax.tick_params(labelsize=label_fontsize*(3/4))

        return ax

    def plot_cdf_pointing_error(self, expert_df, errors_df, tolerance_deg=5.0,
                                per_filter=False, use_bin=False, label_fontsize=20):
        FIG_SIZE = (5.5, 3.8)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        max_x = 0.0
        error_key = 'angular_separation'
        if use_bin:
            error_key = 'bin_' + error_key
        if per_filter:
            for i, filt in enumerate(FILTER2IDX.keys()):
                mask = (expert_df['filter'] == filt).values
                sorted_errors = np.sort(errors_df[error_key][mask])
                if len(sorted_errors) == 0:
                    continue
                y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                f_color = FILTER_COLORS[filt]

                ax.plot(sorted_errors, y, lw=4, alpha=0.7, color=f'C{i}')

                pct = np.searchsorted(sorted_errors, tolerance_deg) / len(sorted_errors)
                ax.axhline(pct, color=f_color, linestyle='dotted', lw=1.5,
                        label=f'{filt}: {pct*100:.1f}% within average \nHEALPix bin width')
                max_x = max(max_x, sorted_errors.max())
            ax.axvline(tolerance_deg, color='red', lw=1.5)
        else:
            sorted_errors = np.sort(errors_df[error_key].values)
            if len(sorted_errors) > 0:
                y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                ax.plot(sorted_errors, y, lw=4, alpha=0.7, color="#3f007d")
                pct = np.searchsorted(sorted_errors, tolerance_deg) / len(sorted_errors)
                ax.axhline(pct, color='red', linestyle='dotted', lw=1.5,
                           label=f'{pct*100:.1f}% within average \nHEALPix bin width')
                ax.axvline(tolerance_deg, color='red', linestyle='dotted', lw=1.5)
                max_x = sorted_errors.max()
        ax.set(xlabel='Pointing error (deg)', ylabel='CDF',
               xlim=(0, max_x or 1), ylim=(0, 1.05))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.tick_params(axis='both', labelsize=label_fontsize*(3/4))
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=14)
        return fig, ax

    def plot_hist_comparison_per_filter(self, feature_name, expert_feature_arr, expert_filters,
                               agent_filters, agent_feature_arr=None, density=True,
                               alpha=0.2, bins=20):
        if agent_feature_arr is None:
            agent_feature_arr = expert_feature_arr

        nrows, ncols = 2, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()
        for ax, filt in zip(axes, FILTER2IDX.keys()):
            exp_m = expert_filters == filt
            ag_m  = agent_filters  == filt
            ax.hist(agent_feature_arr[ag_m],  bins=bins, density=density,
                    color=self.style.agent_color,  histtype='step', lw=1.5)
            ax.hist(expert_feature_arr[exp_m], bins=bins, density=density,
                    color=self.style.expert_color, histtype='step', lw=1.5)
            ax.hist(agent_feature_arr[ag_m],  bins=bins, label=f'Agent chose {filt}',
                    histtype='stepfilled', color=self.style.agent_color,  alpha=alpha, density=density)
            ax.hist(expert_feature_arr[exp_m], bins=bins, label=f'Expert chose {filt}',
                    histtype='stepfilled', color=self.style.expert_color, alpha=alpha, density=density)
            ax.set_xlabel(feature_name)
            ax.legend()
        # Hide any unused subplots (only 5 filters but a 2x3 grid).
        for ax in axes[len(FILTER2IDX):]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    # KDE + scatter, quiver
    # ------------------------------------------------------------------

    def plot_kde_and_scatter(self, feature_x, feature_y, expert_x, expert_y,
                             agent_x, agent_y, agent_alpha=0.2, color_mapping=None,
                             s=5, ax=None):
        ax = ax or plt.gca()
        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)

        if color_mapping is None:
            color_mapping = self.style.agent_color
            agent_label = 'agent'
            handles = None
        else:
            agent_label = None
            handles = FILTER_PATCHES

        sns.kdeplot(x=expert_x, y=expert_y, thresh=0, levels=10,
                    cmap=self.style.expert_cmap, alpha=.5, label='expert', fill=False,
                    norm=mcolors.CenteredNorm(), ax=ax)
        ax.scatter(agent_x, agent_y, marker='*', c=color_mapping, label=agent_label,
                   alpha=agent_alpha, s=s)
        ax.set_xlabel(feature_x, fontsize=16)
        ax.set_ylabel(feature_y, fontsize=16)
        ax.legend(handles=handles, fontsize=16)
        return ax

    def plot_quiver(self, feature_x, feature_y, expert_prev_x, expert_prev_y,
                    expert_next_x, expert_next_y, agent_next_x, agent_next_y, ax=None):
        ax = ax or plt.gca()
        expert_prev_x = _wrap_if_ra(feature_x, expert_prev_x)
        expert_prev_y = _wrap_if_ra(feature_y, expert_prev_y)
        expert_next_x = _wrap_if_ra(feature_x, expert_next_x)
        expert_next_y = _wrap_if_ra(feature_y, expert_next_y)
        agent_next_x  = _wrap_if_ra(feature_x, agent_next_x)
        agent_next_y  = _wrap_if_ra(feature_y, agent_next_y)

        expert_u = expert_next_x - expert_prev_x
        expert_v = expert_next_y - expert_prev_y
        agent_u  = agent_next_x  - expert_prev_x
        agent_v  = agent_next_y  - expert_prev_y

        ax.quiver(expert_prev_x, expert_prev_y, expert_u, expert_v,
                  color=self.style.expert_color, label='expert')
        ax.quiver(expert_prev_x, expert_prev_y, agent_u, agent_v,
                  color=self.style.agent_color, label='agent')
        ax.set_xlabel(feature_x, fontsize=14)
        ax.set_ylabel(feature_y, fontsize=14)
        ax.legend()
        return ax

    def plot_violin_per_filter(self, combined_dfs, key_metric, label_fontsize=20):
        FILTER_ORDER = ['g', 'r', 'i', 'z', 'Y']   # left→right: dark→bright time

        # Sized for a half-column slot on a 24×36 inch portrait poster.
        # Adjust if your column width differs.
        FIG_SIZE = (6, 3.8)

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        key_ylabel_mapping = {
            'moon_el': 'Moon elevation (deg)',
            'moon_distance': 'Moon distance (deg)',
            'moon_phase': 'Moon phase (%)',
            'sky_brightness_g': "Sky brightness (g')",
        }

        # Split violin: Expert = left half, BC Agent = right half of each violin.
        # Requires seaborn >= 0.12.
        # If you see a DeprecationWarning on 'split', upgrade seaborn or swap to the
        # side-by-side fallback at the bottom of this file.
        sns.violinplot(
            data      = combined_dfs,
            x         = 'filter',
            y         = key_metric,
            hue       = 'source',
            order     = FILTER_ORDER,
            hue_order = ['Expert', 'BC Agent'],
            palette   = {'Expert': self.style.expert_color, 'BC Agent': self.style.agent_color},
            split     = True,       # mirror both distributions within one violin body
            inner     = 'quartile', # show median + IQR as dashed lines inside violin
            linewidth = 0.7,
            cut       = 0,          # clip KDE at data range (no phantom tails)
            bw_adjust = 0.8,        # mild smoothing; increase if distributions look spiky
            ax        = ax,
        )

        if 'moon' in key_metric:
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', zorder=0)


        ax.set_xlabel('Filter', fontsize=label_fontsize)
        ax.set_ylabel(key_ylabel_mapping[key_metric], fontsize=label_fontsize)
        # ax.set_title('Filter strategy vs lunar conditions', fontsize=11, pad=5)
        # ax.set_ylim(-82, 82)
        ax.tick_params(labelsize=label_fontsize*(3/4))

        # Compact legend: plain patches, no seaborn extras
        ax.legend(
            handles=[
                patches.Patch(color=self.style.expert_color, label='Expert'),
                patches.Patch(color=self.style.agent_color,  label='BC Agent'),
            ],
            fontsize=label_fontsize*(3/4), framealpha=0.9,
        )

        sns.despine(ax=ax)
        plt.tight_layout(pad=0.5)

    def _plot_metric_distributions(self, combined_df, metrics, label_fontsize=20):
        # One stacked panel per metric
        FIG_SIZE = (9, 2.93 * len(metrics))
        COLORS = {'DES': self.style.expert_color, 'BC Agent': self.style.agent_color}

        fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=FIG_SIZE)

        title_mapping = {
            'airmass': 'Airmass',
            'ha': 'Hour angle (deg)',
            'slew_dist': 'Slew Distance (deg)'
        }

        xlim_mapping = {
            'airmass': (None, 1.65),
        }

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Using smooth KDE density plots to match the visual fidelity of your violins
            sns.kdeplot(
                data        = combined_df,
                x           = metric,
                hue         = 'source',
                palette     = COLORS,
                hue_order   = ['DES', 'BC Agent'],
                fill        = True,
                alpha       = 0.25,
                common_norm = False,
                cut         = 0,          # Clip KDE at data range (no phantom tails)
                bw_adjust   = 0.8,        # Match your reference smoothing setting
                ax          = ax
            )

            for source, color in COLORS.items():
                mean_val = combined_df[combined_df['source'] == source][metric].mean()
                ax.axvline(
                    mean_val,
                    color=color,
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.8
                )

            if metric in xlim_mapping:
                ax.set_xlim(*xlim_mapping[metric])
            ax.set_title(title_mapping[metric], loc='left', fontsize=label_fontsize, pad=4, fontweight='semibold')
            ax.set_ylabel('Density', fontsize=label_fontsize)
            ax.set_xlabel('')  # Keeping x-axis clear as the metric title explains the values
            ax.tick_params(labelsize=label_fontsize*(3/4))
            ax.grid(True, alpha=0.2, linestyle=':')

            # Clean up default seaborn legend behavior for clean subplots
            if ax.get_legend():
                ax.get_legend().remove()

        # Places a single clean legend at the top right of the overall figure
        axes[0].legend(
            handles=[
                patches.Patch(color=self.style.expert_color, label='DES'),
                patches.Patch(color=self.style.agent_color,  label='BC Agent'),
            ],
            fontsize=label_fontsize,
            framealpha=0.9,
            loc='upper right',
        )

        sns.despine(fig=fig)
        plt.tight_layout(pad=1.0)
        return fig, axes
