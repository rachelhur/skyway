import argparse
import numpy as np
from pathlib import Path
from blancops.data.preprocessing import build_DES_lookups
from blancops.data.lookup_tables import TrainLookupTables
from blancops.math import units

from blancops.configs.paths import lookups_dir, resolve_data_dir, workspace
from blancops.configs.enums import AcceptanceRule, LookupKeys
import matplotlib.pyplot as plt
import warnings
import logging
logger = logging.getLogger(__name__)

from blancops.io.logger_utils import configure_logger

def main():
    parser = argparse.ArgumentParser(
        description="Generate train-data lookup tables from raw DECam observations."
    )
    parser.add_argument(
        "--fits_path", type=Path, default=workspace().des_fits,
        help="Path to the raw DECam exposures FITS file",
    )
    parser.add_argument(
        "-o", "--out_parent_dir", type=Path, default=workspace().des_data,
        help="Directory to save the generated lookup tables; relative paths are under the workspace root. Defaults to workspace().des_data",
    )
    parser.add_argument(
        "--acceptance", choices=[r.value for r in AcceptanceRule], default=AcceptanceRule.UNIFORM.value,
        help="Which exposures count toward the survey: one 0.3 teff threshold, or DES's per-band "
             "minimum teff (Morganson et al. 2018, Table 4). Must match data.acceptance of the runs using these lookups.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing existing lookups that were built with a different --acceptance.",
    )
    parser.add_argument(
        '-p', '--save_plots', action="store_true",
        help="Whether or not to save plots. (Will be saved to outdir)"
    )
    args = parser.parse_args()


    # --------------------------------------
    # SETUP LOGGER
    # --------------------------------------

    logger = configure_logger(
        level="info",
        log_to_stdout=True,
        log_to_file=False
    )

    # --------------------------------------
    # SETUP OUTDIR
    # --------------------------------------
    out_parent_dir = resolve_data_dir(args.out_parent_dir)
    lookups_outdir = lookups_dir(out_parent_dir)
    built = ((lookups_outdir / LookupKeys.ACCEPTANCE.value).exists()
             and TrainLookupTables.load_acceptance(lookups_outdir).value)
    if built and built != args.acceptance and not args.overwrite:
        raise SystemExit(f"{lookups_outdir} holds lookups built with acceptance '{built}'; pass a different "
                         f"-o for '{args.acceptance}', or --overwrite to replace them.")
    figures_outdir = out_parent_dir / "figures"
    lookups_outdir.mkdir(parents=True, exist_ok=True)
    figures_outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting DES lookup table generation...")

    # --------------------------------------
    # BUILD LOOKUPS
    # --------------------------------------
    lookups = build_DES_lookups(fits_path=args.fits_path, outdir=lookups_outdir, acceptance=args.acceptance)
    filter2idx = lookups.survey.filter2idx
    save = args.save_plots

    # --------------------------------------
    # PLOTTING
    # --------------------------------------
    if save:
        _FIGSIZE = (6,6)
        _ra = (lookups.fields.ra.values + np.pi) % (2 * np.pi) - np.pi
        _dec = lookups.fields.dec.values

        # --------------------------------------
        # Fields in ra/dec, colored by field id
        # --------------------------------------
        fig, ax  = plt.subplots(figsize=_FIGSIZE)
        ax.scatter(_ra/units.deg, _dec/units.deg, cmap="plasma", c=lookups.fields.index.values, alpha=.5, s=5)
        ax.set_xlabel('ra (deg)')
        ax.set_ylabel('dec (deg)')
        if save:
            fig.savefig(figures_outdir / "field_radecs.png")

        # Plot target counts
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        for filt, fidx in filter2idx.items():
            ax.scatter(np.arange(len(lookups.fields)), lookups.target_fidfilt_counts[:, fidx], label=filt, s=5, alpha=.5)
        ax.set_xlabel('Field id')
        ax.set_ylabel('Counts')
        if save:
            fig.savefig(figures_outdir / "target_counts_per_field_filter.png")

        # --------------------------------------
        # Average Accumulated Visits over bins vs night
        # --------------------------------------
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        visits = np.array(list(lookups.night2fidfilt_visit_hist.values()))
        mean_visits = visits.mean(axis=1)
        std_visits = visits.std(axis=1)
        for filt, fidx in filter2idx.items():
            ax.plot(np.arange(len(mean_visits)), mean_visits[:, fidx], label=filt, color=f"C{fidx}")
            ax.fill_between(
                np.arange(len(lookups.night2fidfilt_visit_hist)),
                y1=mean_visits[:, fidx] - std_visits[:, fidx],
                y2=mean_visits[:, fidx] + std_visits[:, fidx],
                alpha=.3,
                color=f"C{fidx}"
                )
        ax.set_xlabel('Night of survey')
        ax.set_ylabel("Visits")
        ax.legend()
        if save:
            fig.savefig(figures_outdir / "average_visits_over_time.png")

        # --------------------------------------
        # Average Time Since Last Visit over bins vs night FOR INCOMPLETE FIELDS ONLY
        # --------------------------------------

        fig, ax = plt.subplots(figsize=_FIGSIZE)
        last_visit_times = np.array(list(lookups.night2fidfilt_last_visit_ot.values())) / 60 / 60
        ot_clock_hour = np.array(list(lookups.night2ot_clock_seconds.values())) / 60 / 60
        t_since_last_visit = ot_clock_hour[:, None, None] - last_visit_times  # (n_nights, n_fields, n_filters)

        visit_hist = np.array(list(lookups.night2fidfilt_visit_hist.values()))   # (n_nights, n_fields, n_filters)
        incomplete_mask = visit_hist < lookups.target_fidfilt_counts[None, :, :]
        t_since_last_visit = np.where(incomplete_mask, t_since_last_visit, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_times = np.nanmean(t_since_last_visit, axis=1)
            std_times = np.nanstd(t_since_last_visit, axis=1)

        for filt, fidx in filter2idx.items():
            night_idxs = np.arange(len(mean_times))
            _filt_means = mean_times[:, fidx]
            _filt_stds = std_times[:, fidx]
            ax.plot(_filt_means, label=filt, color=f"C{fidx}")
            ax.fill_between(
                x=night_idxs,
                y1=_filt_means - _filt_stds,
                y2=_filt_means + _filt_stds,
                alpha=.3,
                color=f"C{fidx}"
            )
        ax.set_xlabel('Night of survey')
        ax.set_ylabel("Time since last visit (hour)\n(incomplete fields only)")
        ax.legend()
        if save:
            fig.tight_layout()
            fig.savefig(figures_outdir / "average_time_since_last_visit_incomplete.png")


        # --------------------------------------
        # Median Time Since Last Visit over bins vs night FOR INCOMPLETE FIELDS ONLY
        # Includes number of (field, filter) pair contributes per night, filter
        # --------------------------------------

        fig, axes = plt.subplots(
            nrows=len(filter2idx), ncols=1,
            figsize=(8, 2 * len(filter2idx)),
            sharex=True,
        )

        last_visit_times = np.array(list(lookups.night2fidfilt_last_visit_ot.values())) / 60 / 60
        ot_clock_hour = np.array(list(lookups.night2ot_clock_seconds.values())) / 60 / 60
        t_since_last_visit = ot_clock_hour[:, None, None] - last_visit_times   # (n_nights, n_fields, n_filters)

        # Mask out (field, filter) cells that were already complete at the
        # start of each night — same logic as the bin-feature change in
        # compute_bin_progress_features. The np.where converts masked cells
        # to NaN so nanpercentile / nanmean skip them.
        visit_hist = np.array(list(lookups.night2fidfilt_visit_hist.values()))
        incomplete_mask = visit_hist < lookups.target_fidfilt_counts[None, :, :]
        t_since_last_visit = np.where(incomplete_mask, t_since_last_visit, np.nan)

        # Percentile reduction over the fields axis. nanpercentile returns NaN
        # for slices that are all-NaN (a filter with no recorded visits yet,
        # or all-complete late in the survey); matplotlib will leave gaps,
        # which is the intended visual.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pct = np.nanpercentile(t_since_last_visit, [10, 50, 90], axis=1)
            # pct shape: (3, n_nights, n_filters)

        # Contributing-(field, filter) count per (night, filter) — sanity check for the
        # noisy tail. Counts (field, filter) that are both incomplete AND have a recorded
        # visit (i.e., contributed a finite value to the percentile).
        contributing = (incomplete_mask & ~np.isnan(t_since_last_visit)).sum(axis=1)
        # contributing shape: (n_nights, n_filters)

        night_idxs = np.arange(pct.shape[1])
        for (filt, fidx), ax in zip(filter2idx.items(), axes):
            ax2 = ax.twinx()
            p10, p50, p90 = pct[0, :, fidx], pct[1, :, fidx], pct[2, :, fidx]
            ax.plot(night_idxs, p50, color=f"C{fidx}", label=f"{filt} median")
            ax.fill_between(night_idxs, p10, p90, alpha=0.25, color=f"C{fidx}")
            ax2.plot(
                night_idxs, contributing[:, fidx],
                color="gray", linestyle="--", alpha=0.6, linewidth=0.8,
            )
            ax.set_ylabel(f"{filt}\nhour since visit")
            ax2.set_ylabel("n (field, filter) pairs", color="gray")
            ax.set_ylim(bottom=0)
            ax2.tick_params(axis="y", colors="gray")

        axes[-1].set_xlabel("Night of survey")
        fig.suptitle("Time since last visit (incomplete fields only)\nmedian with 10–90 percentile band")
        fig.tight_layout()

        if save:
            fig.tight_layout()
            fig.savefig(figures_outdir / "median_time_since_last_visit_incomplete.png")

if __name__ == "__main__":
    main()
