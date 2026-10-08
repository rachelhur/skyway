"""User-interface adapters for approving scheduler observation chunks.

This module defines the scheduler-facing UI contract and a CLI implementation used for
local, human-in-the-loop operation.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from skyway.math import units
from skyway.plotting import live_scheduling_viz
from skyway.ephemerides import time_utils
import sys, select
import pathlib
import logging
logger = logging.getLogger(__name__)

class BaseInterface(ABC):
    """Abstract interface for user interaction with proposed observation chunks."""

    @abstractmethod
    def __init__(self, output_dir, show_plots):
        """
        Initialize the interface.

        Arguments
        ---------
        output_dir: str or pathlib.Path, optional
            Directory to save any generated outputs (e.g. plots).
        show_plots: bool
            Whether to display plots interactively or just save to disk.
        """

        pass

    @abstractmethod
    def display_chunk(self, chunk_df):
        """
        Render the proposed observation chunk for operator review.

        Arguments
        ---------
        chunk_df: pandas.DataFrame
            Proposed observations and associated metadata.
        """

        pass

    @abstractmethod
    def get_user_decision(self):
        """
        Get user approval/rejection for the current proposed chunk.

        Returns
        -------
        approved: bool
            Whether the user approves the proposed chunk for execution.
        gw_trigger: bool
            Whether the user triggered a GW follow-up.
        quit_requested: bool
            Whether the user requested a clean shutdown.
        """

        pass

    @abstractmethod
    def check_for_replan_signal(self):
        """
        Check whether operator requested an asynchronous chunk replan.

        Returns
        -------
        bool
            True if a replan should be triggered.
        """

        pass


class CLIInterface(BaseInterface):
    """Command-line interface for chunk preview and approval."""

    def __init__(self, output_dir=None, show_plots=True, clock=None):
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self.show_plots = show_plots
        self.clock = clock or time_utils.Clock()
        if self.output_dir is None and not self.show_plots:
            logger.warning("[Interface] Warning: No plots will be saved or displayed.")

        # set up a single, persistent matplotlib interactive mode
        if self.show_plots:
            plt.ion()
            self.fig = plt.figure(figsize=(10.5, 8.5)) # live_scheduling_viz default
            self.fig.canvas.manager.set_window_title("Live Scheduler - Current Chunk")

    def display_chunk(self, chunk_df, completed_df=None, candidate_df=None, current=None):
        """Print the proposed chunk and save a simple RA/Dec plot."""

        # print the proposed chunk as a table in the terminal
        logger.info(
                    "\n" + "=" * 88
                    + "\n Proposed Observing Chunk \n"
                    + "=" * 88
                    + "\n" + chunk_df.to_string(index=False)
                    + "\n" + "=" * 88
                    )

        # skip plotting when upstream returns an empty/malformed proposal
        required_cols = {"ra", "dec"}
        if chunk_df.empty:
            logger.info("[Interface] Chunk is empty; skipping plot generation.")
            return
        if not required_cols.issubset(chunk_df.columns):
            logger.info(
                "[Interface] Chunk missing required ra/dec columns; skipping plot generation."
            )
            return

        # clear the existing window or ensure background cleanup
        if self.show_plots:
            plt.figure(self.fig.number) # make our persistent window the active one
            plt.close('all') # clear the old plot elements
        else:
            plt.close('all') # if only saving to disk, prevent memory leaks

        # generate the plot
        # NB: this will draw onto the currently active figure we just cleared
        # XXX update to include completed, future, and current fields in the plot
        # center and time-stamp the plot on the scheduler clock so the simulated
        # time is respected during testing and the true UTC during live runs
        live_scheduling_viz.plot_live_schedule_snapshot(
            proposed_df=chunk_df,
            completed_df=completed_df,
            candidate_df=candidate_df,
            current=current,
            time=self.clock.now()
        )

        # 2ave uniquely by timestamp (this acts on the currently active figure)
        if self.output_dir is not None:
            outname = self.output_dir / f"chunk_proposal_{self.clock.now()}.png"
            plt.savefig(outname)
            logger.info(f"[Interface] Plot saved to {outname}")

        # update the persistent window
        if self.show_plots:
            self.fig.canvas.draw_idle()  # request a redraw
            self.fig.canvas.flush_events() # process GUI events (prevent window freeze)
            plt.pause(0.01) # give the ssh -X loop a tiny breather to render

    def get_user_decision(self):
        """Prompt for Y/N/Q approval and return scheduler decision payload."""

        while True:
            resp = (
                input("Approve this chunk? [Y]es, [N]o (mask fields), [NT] No and trigger GW ToO, [Q]uit: ").strip().upper()
            )
            if resp == "Y":
                logger.info("[Interface] Chunk accepted.")
                return True, False, False
            elif resp == "N":
                logger.info("[Interface] Chunk rejected.")
                return False, False, False
            elif resp == "NT":
                logger.info("[Interface] Chunk rejected with temporary GW follow-up trigger.")
                return False, True, False
            elif resp == "Q":
                logger.info("[Interface] Clean shutdown requested at prompt.")
                return False, False, True
            else:
                logger.warning("[Interface] Invalid input, please enter Y, N, NT, or Q.")

    def check_for_replan_signal(self):
        """
        Non-blocking soft-interrupt check using standard select. Performs the following
        user-input checks:
        - r + Enter: signal to replan the current chunk
        - t + Enter: (temporary) signal gravitational-wave follow-up observations

        Returns
        -------
        str
            Returns the type of signal detected, or None if no signal detected. Options:
            - r = 'replan': requested a replan of the current chunk
            - t = 'gw-trigger': requested a gravitational-wave follow-up (temporary)
            - q = 'shutdown': requested a graceful shutdown of the scheduler
            - None: no signal detected
        """
        # select.select checks if a file descriptor is ready to be read
        # [sys.stdin] is the list of objects we want to check
        # 0.0 timeout: tell select to return instantly (non-blocking)
        if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:

            # entering this block means user hit Enter; read the line and clean buffer
            user_input = sys.stdin.readline().strip().lower()
            if user_input == 'r':
                logger.info("[Interface] Soft interrupt detected! 'r' entered.")
                return 'replan'
            elif user_input == 't':
                logger.info("[Interface] Gravitational-wave trigger signal detected! 't' entered.")
                return 'gw-trigger'
            elif user_input == 'q':
                logger.info("[Interface] Shutdown signal detected! 'q' entered.")
                return 'shutdown'
            else: # ignore other input; note that reading the input cleared the buffer
                pass

        return None
