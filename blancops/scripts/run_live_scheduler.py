"""Command-line entrypoint for the live scheduler."""

import argparse
import sys
import os
from pathlib import Path
import yaml

# Ensure blancops is in the path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blancops.live_scheduler.client import MockTelescopeClient, BlancoSCLTelescopeClient
from blancops.live_scheduler.model_runner import MockModelRunner, AIModelRunner
from blancops.live_scheduler.interface import CLIInterface
from blancops.live_scheduler.progress_manager import ProgressManager
from blancops.live_scheduler.orchestrator import SchedulerOrchestrator
from blancops.math import units
from blancops.ephemerides import time_utils

# Use logging module -> stdout and/or file out instead of lieu statements
from blancops.io.logger_utils import configure_logger
from blancops.configs.paths import LIVE_SCHEDULER_DEFAULT_CONFIG

DEFAULT_CONFIG_PATH = LIVE_SCHEDULER_DEFAULT_CONFIG

def load_yaml_defaults(config_path):
    """Load scheduler defaults from a YAML config file."""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file at {config_path} does not exist.")

    with config_path.open("r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}

    if not isinstance(defaults, dict):
        raise ValueError(
            f"Config file at {config_path} must contain a mapping of defaults."
        )

    return defaults


def parse_args():
    """Parse CLI arguments, using YAML defaults unless overridden on the command line."""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file containing scheduler defaults.",
    )

    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the blancops live scheduler.",
    )

    # ==================================================================================
    # Client Settings
    # ==================================================================================
    client_group = parser.add_argument_group("Client Settings")
    client_group.add_argument(
        "--client-mode",
        choices=("mock", "blanco", "blanco_test"),
        default=defaults.get("client_mode", "mock"),
        help=(
            'Scheduler mode to run. Choose "mock" for offline testing. Any other mode '
            'containing "test" (e.g. "blanco_test") enables day-time testing on the '
            "real Blanco client, submitting harmless dark exposures."
        ),
    )
    client_group.add_argument(
        "--mock-exposure-duration",
        type=float,
        default=defaults.get("mock_exposure_duration", 90),
        help="Simulated exposure duration in seconds when running in mock mode.",
    )
    client_group.add_argument(
        "--scl-server-ip",
        type=str,
        default=defaults.get("scl_server_ip", "observer4.ctio.noao.edu"),
        help=(
            "IP address of the SCL server for telescope control. Default is for "
            "running on the Observer6 computer."
        ),
    )
    client_group.add_argument(
        "--scl-server-port",
        type=int,
        default=defaults.get("scl_server_port", 20000),
        help="TCP port of the SCL server for telescope control.",
    )
    client_group.add_argument(
        "--propid",
        type=str,
        default=defaults.get("propid", "2019A-0305"), # XXX we should change default from alex
        help="Proposal ID to include in observation submissions.",
    )

    # ==================================================================================
    # UI Settings
    # ==================================================================================
    ui_group = parser.add_argument_group("UI Settings")
    ui_group.add_argument(
        "--show-plots",
        action="store_true",
        default=defaults.get("show_plots", False),
        help="Whether to display proposed chunks interactively or just save to disk.",
    )
    ui_group.add_argument(
        "--ui-mode",
        choices=("cli", "web"),
        default=defaults.get("ui_mode", "cli"),
        help=(
            'User interface to use. Choose "cli" for command-line or "web" for local '
            "web interface."
        ),
    )

    # ==================================================================================
    # Model Settings
    # ==================================================================================
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model-path-or-alias",
        type=str,
        default=defaults.get("model_path_or_alias", "mock"),
        help='Alias name or path to the trained model. Choose "mock" for debugging.',
    )
    model_group.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", "cpu"),
        help="Torch device for model inference, e.g. cpu, cuda, or cuda:0.",
    )
    model_group.add_argument(
        "--field-choice-method",
        choices=("interp",),
        default=defaults.get("field_choice_method", "interp"),
        help="Method used to map model outputs to physical fields.",
    )

    # ==================================================================================
    # Pathing
    # ==================================================================================
    path_group = parser.add_argument_group("Pathing")
    path_group.add_argument(
        "--field-lookup-dir",
        type=Path,
        default=defaults.get("field_lookup_dir", "./field_lookups"),
        help="Directory containing field lookup tables for AI inference.",
    )
    path_group.add_argument(
        "--fields-path",
        type=Path,
        default=defaults.get("fields_path", None),
        help=(
            "Optional path to a fields file used to construct lookups. If omitted, "
            "lookups are loaded from --field-lookup-dir."
        ),
    )
    path_group.add_argument(
        "--obs-history-filename",
        type=Path,
        default=defaults.get("obs_history_filename", None),
        help=(
            "Optional prior observing-history CSV (columns: field_id, filter, "
            "timestamp) used to seed the survey backlog before this session's "
            "own logged visits are replayed on top."
        ),
    )
    path_group.add_argument(
        "--output-directory",
        type=Path,
        default=defaults.get("output_directory", "./observing_logs"),
        help="Directory where observing logs are written.",
    )
    path_group.add_argument(
        "--session-id",
        type=str,
        default=defaults.get("session_id", None),
        help="Optional observing-session identifier.",
    )

    # ==================================================================================
    # Operational Settings
    # ==================================================================================
    operational_group = parser.add_argument_group("Operational Settings")
    operational_group.add_argument(
        "--sim-time-now",
        type=str,
        default=defaults.get("sim_time_now", None),
        help=(
            "Optional UTC time to simulate running this script. When set, the scheduler"
            "time base is shifted by a fixed offset from the real clock."
        ),
    )
    operational_group.add_argument(
        "--sim-time-offset",
        type=str,
        default=defaults.get("sim_time_offset", None),
        help=(
            "Optional fixed offset to shift the scheduler time base from the real "
            "clock. This is an alternative to --sim-time-now. Provide offset in format "
            "expected by time_utils.standardize_timedelta()."
        ),
    )
    operational_group.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.get("chunk_size", 3),
        help="Number of observations to generate per proposal chunk.",
    )
    operational_group.add_argument(
        "--min-chunk-size",
        type=int,
        default=defaults.get("min_chunk_size", None),
        help=(
            "Minimum number of observations allowed in a chunk before the scheduler "
            "automatically generates a new proposed chunk. Default None generates a "
            "new chunk immediately after every observation submission."
        ),
    )
    operational_group.add_argument(
        "--observing-poll-rate-sec",
        type=float,
        default=defaults.get("observing_poll_rate_sec", 1),
        help="How often to poll the telescope while waiting for an exposure.",
    )
    operational_group.add_argument(
        "--telemetry-poll-rate-sec",
        type=float,
        default=defaults.get("telemetry_poll_rate_sec", 20),
        help="How often to re-check telemetry before deciding whether to replan.",
    )
    operational_group.add_argument(
        "--seeing-window",
        type=str,
        default=defaults.get("seeing_window", "15m"),
        help=(
            "Time window for recent seeing measurements to include as model features. "
            'Format is a number followed by a time unit, e.g. "15m" for 15 minutes.'
        ),
    )
    operational_group.add_argument(
        "--auto-approve",
        action="store_true",
        default=defaults.get("auto_approve", False),
        help=(
            "Whether to skip user approval of proposed chunks. When set, the scheduler "
            "automatically accepts all proposed chunks."
        ),
    )
    operational_group.add_argument(
        "--ra-dec-recovery",
        action="store_true",
        default=defaults.get("ra_dec_recovery", False),
        help=(
            "When recovering from completed observing history logs, whether to use the "
            "last logged RA/Dec as the initial pointing. Default uses zenith."
        ),
    )

    # ==================================================================================
    # Observing Limits
    # ==================================================================================
    limits_group = parser.add_argument_group("Observing Limits")
    limits_group.add_argument(
        "--start-time",
        type=str,
        default=defaults.get("start_time", None),
        help=(
            "Optional start time. Both this and sun elevation conditions must be met "
            "to start the observing session."
        ),
    )
    limits_group.add_argument(
        "--start-sun-elevation-deg",
        type=float,
        default=defaults.get("start_sun_elevation_deg", -14),
        help=(
            "Optional sun elevation threshold. Both this and start time must be met "
            "to start the observing session."
        ),
    )
    limits_group.add_argument(
        "--stop-time",
        type=str,
        default=defaults.get("stop_time", None),
        help=(
            "Optional stop time. Either this or sun elevation condition being met "
            "will end the observing session."
        ),
    )
    limits_group.add_argument(
        "--stop-sun-elevation-deg",
        type=float,
        default=defaults.get("stop_sun_elevation_deg", -14),
        help=(
            "Optional sun elevation threshold. Either this or stop time being met "
            "will end the observing session."
        ),
    )

    return parser.parse_args()


def main():
    """Run the live scheduler with YAML defaults and optional CLI overrides."""

    # parse and validate CLI arguments, with defaults loaded from YAML config
    args = parse_args()
    if args.min_chunk_size is not None and not (0 <= args.min_chunk_size < args.chunk_size):
        raise ValueError(
            f"min_chunk_size ({args.min_chunk_size}) must be in [0, chunk_size) "
            f"(chunk_size={args.chunk_size}) so that at least one observation is "
            "submitted per chunk."
        )
    if args.start_time is not None:
        args.start_time = time_utils.standardize_time(args.start_time)
    if args.stop_time is not None:
        args.stop_time = time_utils.standardize_time(args.stop_time)
    if args.start_sun_elevation_deg is not None:
        args.start_sun_elevation_deg = args.start_sun_elevation_deg * units.deg
    if args.stop_sun_elevation_deg is not None:
        args.stop_sun_elevation_deg = args.stop_sun_elevation_deg * units.deg

    # set up simulated clock for testing
    clock = time_utils.Clock()
    if args.sim_time_now is not None and args.sim_time_offset is not None:
        raise ValueError(
            "You can specify --sim-time-now or --sim-time-offset, but not both."
        )
    elif args.sim_time_now is not None:
        fake_start_ts = time_utils.standardize_time(args.sim_time_now)
        clock = time_utils.Clock(offset=fake_start_ts - clock.now(real=True))
    elif args.sim_time_offset is not None:
        clock = time_utils.Clock(
            offset=time_utils.standardize_timedelta(args.sim_time_offset)
        )

    # Setup logger. The format arg determines how message is formatted. For example, with the format below, a message will be printed like:
    # 2026-05-01 13:40:05 - INFO - Initializing blancops Live Scheduler...".
    now_real = time_utils.unix_to_datetime(clock.now(real=True))
    now_real = now_real.strftime("%Y%m%d_%H%M%S")
    logger = configure_logger(
        level="info",
        log_to_stdout=True,
        log_to_file=True,
        outdir=args.output_directory / "logs",
        filename=f"run_live_scheduler_{now_real}.log",
        use_tqdm=True
    )

    # log the clock settings
    if args.sim_time_now is not None or args.sim_time_offset is not None:
        now_sim = time_utils.unix_to_datetime(clock.now())
        now_sim = now_sim.strftime("%Y-%m-%d %H:%M:%S UTC")
        now_real = time_utils.unix_to_datetime(clock.now(real=True))
        now_real = now_real.strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.info(
            "[Scheduler] Fake clock enabled: real UTC shifted by %.3f seconds.",
            clock.offset,
        )
        logger.info(f"[Scheduler] Simulated current UTC time: {now_sim}")
        logger.info(f"[Scheduler] Real current UTC time: {now_real}")

    # log the parsed arguments
    logger.info(f"User-provided command line call: {' '.join(sys.argv)}")
    logger.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # initialize requested telescope client
    logger.info("Initializing blancops Live Scheduler...")
    if args.client_mode.lower() == "mock":
        client = MockTelescopeClient(
            exposure_duration=args.mock_exposure_duration,
            clock=clock,
            seeing_window=args.seeing_window,
        )
    else:
        client = BlancoSCLTelescopeClient(
            propid=args.propid,
            server_ip=args.scl_server_ip,
            server_port=args.scl_server_port,
            clock=clock,
            seeing_window=args.seeing_window,
            daytime_testing="test" in args.client_mode.lower(),
        )

    # initialize model runner
    if args.model_path_or_alias.lower() == "mock":
        model = MockModelRunner(clock=clock)
    else:
        model = AIModelRunner(
            model_path_or_alias=args.model_path_or_alias,
            field_lookup_dir=args.field_lookup_dir,
            fields_path=args.fields_path,
            obs_history_path=args.obs_history_filename,
            device=args.device,
            field_choice_method=args.field_choice_method,
            clock=clock,
            mode='test',  # XXX check this
            seeing_window=args.seeing_window,
        )

    # initialize ui
    if args.ui_mode.lower() == "cli":
        ui = CLIInterface(
            output_dir=args.output_directory,
            show_plots=args.show_plots,
            clock=clock,
        )
    else:
        raise NotImplementedError(f"UI mode '{args.ui_mode}' is not implemented yet.")

    # initialize progress manager with session metadata and persisted history
    progress = ProgressManager(
        output_dir=args.output_directory,
        clock=clock,
        session_id=args.session_id,
        start_time=args.start_time,
        start_sun_elevation=args.start_sun_elevation_deg,
        stop_time=args.stop_time,
        stop_sun_elevation=args.stop_sun_elevation_deg,
    )

    # initialize orchestrator with all components for running the scheduler loop
    orchestrator = SchedulerOrchestrator(
        client,
        model,
        ui,
        progress,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        observing_poll_rate_sec=args.observing_poll_rate_sec,
        telemetry_poll_rate_sec=args.telemetry_poll_rate_sec,
        clock=clock,
        auto_approve=args.auto_approve,
        ra_dec_recovery=args.ra_dec_recovery,
    )

    # run the scheduler loop
    try:
        orchestrator.run()  # XXX make sure there are stops built in there

    # Big Red Stop Button: handle graceful shutdown on Ctrl+C keyboard interrupt
    except KeyboardInterrupt:
        logger.warning("!" * 88)
        logger.warning("EMERGENCY STOP TRIGGERED (Ctrl+C)")
        logger.warning("Halting all scheduler loops.")
        logger.warning("!" * 88)
        client.close()  # ensure any open connections are closed on exit
        sys.exit(0)


if __name__ == "__main__":
    main()
