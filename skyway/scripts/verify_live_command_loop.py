"""
Diagnostic script to verify raw TCP/IP hardware loops and exposure sequences. This is a
stripped-down version of the full live scheduler loop, designed to test the core command
submission and status polling logic without the complexities of the full scheduler
or AI model generation. It can be run in "mock" mode for offline testing or with the
real Blanco SCLN client ("blanco") for on-site diagnostics. The schedule to be run is
a random walk around zenith, with "dark" exposures of a specified duration.
"""

import argparse
import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# ensure skyway is in the path if running directly from the scripts folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skyway.live_scheduler.client import MockTelescopeClient, BlancoSCLTelescopeClient
from skyway.live_scheduler.interface import CLIInterface
from skyway.live_scheduler.model_runner import MockModelRunner
from skyway.ephemerides import ephemerides


def setup_logging(log_dir="diagnostic_logs"):
    """Configure global logging to output to the terminal and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)

    # generate a timestamped filename so logs are never overwritten
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"diagnostic_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),  # prints to terminal simultaneously
        ],
    )
    return log_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a stripped-down hardware loop test."
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="Number of exposures to submit in the sequence.",
    )
    parser.add_argument(
        "--exp-time", type=int, default=30, help="Exposure time in seconds."
    )
    parser.add_argument(
        "--poll-rate",
        type=float,
        default=1.0,
        help="Seconds to sleep between checking observing status.",
    )
    parser.add_argument(
        "--client-mode",
        choices=["mock", "blanco", "blanco_test"],
        default="mock",
        help=(
            'Which client to use for the test. Any other mode containing "test" (e.g. '
            '"blanco_test") enables day-time testing on the real Blanco client, '
            "submitting harmless dark exposures."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="diagnostic_logs",
        help="Directory to save logs and outputs.",
    )
    parser.add_argument(
        "--scl-server-ip",
        type=str,
        default="observer4.ctio.noao.edu",
        help="IP address of the SCL server.",
    )
    parser.add_argument(
        "--scl-server-port", type=int, default=20000, help="TCP port of the SCL server."
    )
    parser.add_argument(
        "--propid",
        type=str,
        default="2013A-9999",
        help="Proposal ID to include in submissions.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Whether to display plots in the CLI interface or just save to output dir",
    )
    parser.add_argument(
        "--override-error",
        action="store_true",
        help=(
            "Whether to override the sun elevation check for real science exposures. "
            "THIS IS DANGEROUS AND SHOULD ONLY BE USED FOR TESTING PURPOSES UNDER "
            "DIRECT SUPERVISION OF THE STAFF."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    log_file = setup_logging(log_dir=args.output_dir)
    ui = CLIInterface(output_dir=args.output_dir, show_plots=args.show_plots)

    # log the exact terminal command used to launch the script
    command_used = " ".join(sys.argv)
    logging.info("=" * 60)
    logging.info(f"STARTING HARDWARE DIAGNOSTIC LOOP")
    logging.info(f"Command run: python {command_used}")
    logging.info(f"Log file: {log_file}")
    logging.info("=" * 60)

    # 1. Initialize Client
    if args.client_mode == "mock":
        logging.info("[Test] Initializing Mock Client for offline testing...")
        client = MockTelescopeClient(exposure_duration=args.exp_time)
    else:
        logging.info(
            f"[Test] Initializing SCLN Client at {args.scl_server_ip}:{args.scl_server_port}..."
        )
        client = BlancoSCLTelescopeClient(
            propid=args.propid,
            server_ip=args.scl_server_ip,
            server_port=args.scl_server_port,
            daytime_testing="test" in args.client_mode.lower(),
            override_error=args.override_error,
        )

    # 2. Generate the Observing Chunk
    logging.info(f"[Test] Generating observation chunk of size {args.cycles}...")
    model = MockModelRunner()

    # get telemetry and default starting position at zenith
    start_ra, start_dec = ephemerides.get_source_ra_dec("zenith")
    telemetry = client.get_telemetry(print_data=True)
    if telemetry.get("pointing_ra") is None:
        logging.warning(f"[Test] No current RA received from telemetry; defaulting to zenith {start_ra}")
        telemetry["pointing_ra"] = start_ra
    if telemetry.get("pointing_dec") is None:
        logging.warning(f"[Test] No current Dec received from telemetry; defaulting to zenith {start_dec}")
        telemetry["pointing_dec"] = start_dec

    # generate the random walk sequence created by MockModelRunner
    chunk_attempt = 1
    while True:
        logging.info(f"[Test] Generating observation chunk attempt {chunk_attempt}...")
        chunk_df = model.generate_chunk(
            telemetry=telemetry,
            available_fields=[],
            masked_field_ids=[],
            chunk_size=args.cycles,
        )

        if chunk_df is None or chunk_df.empty:
            logging.warning("[Test] Model returned an empty chunk. Regenerating...")
            chunk_attempt += 1
            continue

        ui.display_chunk(chunk_df)
        approved = ui.get_user_decision()
        if approved:
            break

        logging.info("[Test] Chunk rejected by user. Regenerating a new proposal...")
        chunk_attempt += 1

    logging.info("[Test] Approved observation chunk:")
    logging.info("\n%s", chunk_df.to_string(index=False))

    # 3. The Execution Loop
    logging.info("[Test] Entering physical submission loop...")

    for i, obs_row in chunk_df.iterrows():
        obs_number = i + 1
        obs_row_dict = obs_row.to_dict()
        obs_row_dict["expTime"] = args.exp_time

        logging.info(
            f"[Test] --- Processing Observation {obs_number}/{args.cycles} ---"
        )
        logging.info("[Test] Observation row to submit:")
        logging.info("\n%s", pd.DataFrame([obs_row_dict]).to_string(index=False))

        # poll until the telescope is ready to accept a command
        # (this safely waits for the previous exposure to finish)
        logging.info("[Test] Polling for readiness...")
        while not client.check_exposure_status():
            time.sleep(args.poll_rate)

        logging.info(f"[Test] Telescope ready. Submitting observation {obs_number}...")
        response = client.submit_observation(obs_row_dict, exp_time=args.exp_time)

        # check for hardware aborts
        if response and response.get("status") == "FAILED":
            logging.error(
                f"[Test] HARDWARE ABORT: Server returned FAILED: {response.get('message')}"
            )
            logging.error("[Test] Safely closing connection and exiting script.")
            client.close()
            sys.exit(1)

    # 4. Wait for the final exposure to finish before closing
    logging.info("=" * 60)
    logging.info(
        "[Test] All commands submitted. Waiting for the final exposure to complete..."
    )

    # reset loop one last time
    while not client.check_exposure_status():
        time.sleep(args.poll_rate)

    logging.info("[Test] Final exposure finished successfully!")
    logging.info("[Test] Closing connection.")
    logging.info("=" * 60)

    client.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
