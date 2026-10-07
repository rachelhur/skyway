"""Offline RL rollout entry point (``run-offline-scheduler``).

Builds the agent/policy from a trained or deployable model, constructs a
multi-night forward-simulation ``OfflineBlancoEnv``, and runs the policy to
generate an observing schedule. The first night's survey state can be seeded
from a prior observing history via ``--obs_history_filename``.
"""
import pandas as pd
import numpy as np
import gymnasium as gym

from blancops.configs.paths import OfflineRunPaths, RunPaths, workspace
from blancops.configs.experiment_schema import ActionConstraints
from blancops.rl.agent_factory import AgentFactory
from blancops.rl.offline_runner import OfflineRunner
from blancops.data.lookup_tables import LookupTables
from blancops.data.obs_history import load_seed_state_from_obs_history
from blancops.data.seeing_trajectory import extract_night_seeing_trajectory
from blancops.utils.sys_utils import seed_everything
from blancops.io.logger_utils import configure_logger
from blancops.utils.sys_utils import get_system_device
from blancops.environment.offline_env import OfflineBlancoEnv
from blancops.environment.field_mask_schedule import FieldMaskSchedule

import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model choice
    parser.add_argument('-m', '--model_path_or_alias', type=str, default="bc_v1", help='Model alias or relative path to trained model directory')

    # Fields
    parser.add_argument('--field_lookup_dir', type=Path, required=True, help='Relative path to field lookup directory')
    parser.add_argument('--obs_history_filename', type=str, default=None,
                        help='If provided, seed the first night from a prior observing history. '
                             'Accepts a schedule CSV (.csv) or a live observing log (.jsonl/.json).')

    # Observing nights
    parser.add_argument('--start_time', type=float, default=None,
                        help="Unix timestamp at which every simulated night begins."
                        "A different start time for each night is not supported.")
    parser.add_argument('--stop_time', type=float, default=None,
                        help="Unix timestamp at which every simulated night ends."
                        "A different stop time for each night is not supported.")
    parser.add_argument('-d', '--observing_nights', type=str, nargs='*', default=['2026-06-23-half2', '2026-06-24-half2'],
                        help="List of observing nights. Format [YY-MM-DD-NIGHT, ...] where NIGHT is one of " +
                        "'full', 'half1', 'half2' (e.g. 2026-06-23-full)"
                        )

    # Output
    parser.add_argument('-o', '--outdir', type=Path, required=True, help='Relative path to output directory')
    parser.add_argument('--schedule_prefix', type=str, default='schedule',
                        help='Base filename prefix for the generated schedule output')
    parser.add_argument('-s', '--save_observing_script', action='store_true',
                        help="Whether to save schedules as the telescope's observing script (SISPI JSON for Blanco).")
    parser.add_argument('--propid', type=str, default=None,
                        help='Proposal id written to observing scripts. Required with --save_observing_script.')
    parser.add_argument('--proposer', type=str, default='ai-scheduler', help='Proposer written to observing scripts.')
    parser.add_argument('--program', type=str, default=None,
                        help='Program name written to observing scripts. Required with --save_observing_script.')
    parser.add_argument('--save_state_features', action='store_true',
                        help="Whether to save per-night glob/bin observation arrays as _obs.npz files.")

    # Plotting
    parser.add_argument('--save_movie', action='store_true', help='Whether to save gif files.')
    parser.add_argument('--save_mollweide', action='store_true', help='Whether to save png files.')
    parser.add_argument('--plot_bins', action='store_true',
                        help='Also draw HEALPix bins in movies of field-level models (bin-level models always draw them).')

    # Logging
    parser.add_argument('-l', '--logging_level', type=str, default='info', choices=['info', 'debug', 'warning', 'error'], help='Logging level.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing schedule if already exists.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for schedule generation')

    # Scheduling parameters
    parser.add_argument('--sun_el_limit', type=float, default=-12, help="Highest sun elevation (in deg) for observing. Default is -12.")
    parser.add_argument('--airmass_limit', type=float, default=1.8,
                        help="Only fields with airmass below this limit can be scheduled.")
    parser.add_argument('--initial_fwhm', type=float, default=0.9,
                        help="Assumed zenith delivered seeing (arcsec, r-band) for the forward sim, "
                             "projected per pointing by airmass/filter. Default 0.9 is the CTIO Blanco/DECam "
                             "ignored when --seeing_val_night is given.")

    # Field masking (time-windowed field-id masks). Omit --mask_baseline_field_ids to disable.
    parser.add_argument('--mask_baseline_field_ids', type=int, nargs='*', default=None,
                        help='Field ids masked outside any mask window (baseline). If omitted, no masking is applied.')
    parser.add_argument('--mask_baseline_mode', type=str, choices=['mask', 'keep_only'], default='mask',
                        help="Baseline mask mode: 'mask' hides these field ids; 'keep_only' hides all others.")
    parser.add_argument('--mask_window_start', type=float, default=None, help='Unix ts (UTC) start of the mask window.')
    parser.add_argument('--mask_window_end', type=float, default=None, help='Unix ts (UTC) end of the mask window.')
    parser.add_argument('--mask_window_field_ids', type=int, nargs='*', default=None,
                        help='Field ids for the mask window rule.')
    parser.add_argument('--mask_window_mode', type=str, choices=['mask', 'keep_only'], default='keep_only',
                        help="Window mask mode: 'keep_only' hides all field ids except these during the window.")

    # Diagnostics/legacy
    parser.add_argument('-c', '--field_choice_method', type=str, default='interp', choices=['random', 'interp'], help="Field selection method within a chosen bin.")
    parser.add_argument('--action_decode', type=str, default='joint', choices=['joint', 'filter_first'], help="Bin/filter decode: 'joint' argmax, or 'filter_first' (choose filter over all visible bins, then best available bin).")
    parser.add_argument('--dump_moonset_q', action='store_true', help="Print a one-shot per-filter Q breakdown at the first post-moonset step (diagnostic).")
    parser.add_argument('--val_seeing_cache', type=Path,
                        default=RunPaths(workspace().deployable_models / 'bc_v1_max_feature_set').dataset_cache('val'),
                        help="Path to a val_dataset_cache.pt holding the validation-night DataFrame, "
                             "used with --seeing_val_night to replay a real night's measured seeing.")
    parser.add_argument('--downtime_csv', type=Path, default=None,
                        help="CSV with columns start, end (unix timestamps) giving intervals "
                             "in which the telescope was not observing. The replay idles "
                             "through them instead of slewing straight on, so that a simulated "
                             "night covers the same observing time a real one did.")
    parser.add_argument('--reset_counts_on_exhaustion', action='store_true',
                        help="When every reachable survey target is complete, zero the visit "
                             "counts and keep observing instead of idling to the end of the "
                             "night. Matches live operation, where the scheduler was restarted "
                             "against a fresh history once it ran out.")
    parser.add_argument('--seeing_trajectory_csv', type=Path, default=None,
                        help="CSV with columns sec_since_sunset, fwhm (arcsec), band, el (rad) "
                             "to replay as the night's measured seeing. Use for nights that are "
                             "not in a validation cache, e.g. a deployment night scored from its "
                             "own telemetry. Overrides --initial_fwhm and --seeing_val_night.")
    parser.add_argument('--seeing_val_night', type=str, default=None,
                        help="Validation night key (date string in the cache's 'night' column) whose "
                             "measured seeing trajectory to replay each sim night. Overrides --initial_fwhm. "
                             "Omit to use a constant --initial_fwhm.")

    args = parser.parse_args()
    if args.save_observing_script and not args.propid:
        parser.error("--propid is required with --save_observing_script")
    return args


def main():
    # Parse args
    args = get_args()

    # ------------------------------
    # LOAD TARGET FIELDS
    # ------------------------------

    lookup_dir = Path(args.field_lookup_dir)
    lookups = LookupTables.load_from_dir(data_dir=lookup_dir)

    # ---------------------------------
    # SETUP LOGGER AND OUTDIR
    # ---------------------------------
    device = get_system_device()
    seed_everything(args.seed)

    outdir = Path(args.outdir)

    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=outdir,
        filename=OfflineRunPaths.LOG,
        use_tqdm=True
    )

    logger.info("Arguments:")
    for key, value in vars(args).items():
        logger.info(
            "\t" + f"{key}: {value}"
            )

    logger.info(f"Using {outdir} as output directory.")

    # ---------------------------------
    # LOAD AGENT, MODEL, AND OFFLINE RUNNER
    # ---------------------------------
    logger.info("Loading agent...")
    factory = AgentFactory()

    agent, model_cfg, norm_stats = factory.build_agent(
        model_path_or_alias=args.model_path_or_alias,
        lookups=lookups,
        field_choice_method=args.field_choice_method,
        device=device,
        action_decode=args.action_decode,
    )
    runner = OfflineRunner(
        agent=agent, policy=agent.policy, cfg=model_cfg,
        lookups=lookups, telescope=lookups.survey.telescope,
        outdir=outdir,
        save_observing_script=args.save_observing_script, save_movie=args.save_movie,
        observing_script_kwargs={'propid': args.propid, 'proposer': args.proposer, 'program': args.program},
        save_mollweide=args.save_mollweide,
        plot_bins=args.plot_bins,
        save_state_features=args.save_state_features,
        dump_moonset_q=args.dump_moonset_q
    )

    # ---------------------------------
    # CREATE ENVIRONMENT
    # ---------------------------------
    logger.info("Setting up environment...")
    env_name = 'OfflineBlanco-v0'
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=OfflineBlancoEnv,
    )

    # norm_stats come from the exact weights file loaded for the policy
    # (returned by build_agent), so normalization always matches the policy.

    # Seed the first night's survey state, either from a prior observing
    # history or from a cold start (no prior visits, OT clock at 0).
    if args.obs_history_filename:
        logger.info(f"Seeding initial state from observing history: {args.obs_history_filename}")
        initial_counts, initial_last_visit_ot, initial_ot_at_sunset = (
            load_seed_state_from_obs_history(
                Path(args.obs_history_filename), lookups, args.sun_el_limit
            )
        )
    else:
        initial_counts = np.zeros_like(lookups.target_fidfilt_counts)
        initial_last_visit_ot = np.full(shape=lookups.target_fidfilt_counts.shape, fill_value=np.nan)
        initial_ot_at_sunset = 0.0

    # Replay a real validation night's measured seeing when requested. The
    # extracted trajectory (keyed by seconds-since-sunset) is saved to the run
    # outdir for provenance and re-aligned to each sim night inside the env.
    seeing_trajectory = None
    if args.seeing_trajectory_csv is not None:
        logger.info(f"Replaying seeing trajectory from {args.seeing_trajectory_csv}")
        seeing_trajectory = pd.read_csv(args.seeing_trajectory_csv)
    elif args.seeing_val_night is not None:
        logger.info(
            f"Extracting seeing trajectory for night {args.seeing_val_night} from "
            f"{args.val_seeing_cache}"
        )
        seeing_trajectory = extract_night_seeing_trajectory(
            cache_path=args.val_seeing_cache,
            val_night=args.seeing_val_night,
            sun_el_limit=args.sun_el_limit,
        )
        seeing_csv_path = outdir / 'val_night_seeing.csv'
        seeing_trajectory.to_csv(seeing_csv_path, index=False)
        logger.info(f"Saved seeing trajectory to {seeing_csv_path}")

    # Build the time-windowed field-mask schedule (None when no masking args given).
    field_mask_schedule = FieldMaskSchedule.build(
        baseline_field_ids=args.mask_baseline_field_ids,
        baseline_mode=args.mask_baseline_mode,
        window_start=args.mask_window_start,
        window_end=args.mask_window_end,
        window_field_ids=args.mask_window_field_ids,
        window_mode=args.mask_window_mode,
    )

    downtime_windows = None
    if args.downtime_csv is not None:
        dt = pd.read_csv(args.downtime_csv)
        downtime_windows = list(zip(dt["start"].astype(float),
                                    dt["end"].astype(float)))
        logger.info(f"Loaded {len(downtime_windows)} downtime intervals from "
                    f"{args.downtime_csv}")

    env = gym.make(
        id=f"gymnasium_env/{env_name}",
        cfg=model_cfg,
        constraints_cfg=ActionConstraints(sun_el_limit=args.sun_el_limit,
                                          airmass_limit=args.airmass_limit,
                                          airmass_failsafe=args.airmass_limit), # extra failsafe for live scheduler
        lookups=lookups,
        norm_stats=norm_stats,
        observing_night_strs=args.observing_nights,
        initial_counts=initial_counts,
        initial_last_visit_ot=initial_last_visit_ot,
        initial_ot_at_sunset=initial_ot_at_sunset,
        initial_fwhm=args.initial_fwhm,
        seeing_trajectory=seeing_trajectory,
        start_time=args.start_time,
        stop_time=args.stop_time,
        downtime_windows=downtime_windows,
        field_mask_schedule=field_mask_schedule,
        reset_counts_on_exhaustion=args.reset_counts_on_exhaustion,
    )

    # ---------------------------------
    # RUN POLICY
    # ---------------------------------
    logger.info("Running policy rollout...")
    runner.run(env=env)

    logger.info(f"Done. Output written to: {outdir}")


if __name__ == "__main__":
    main()
