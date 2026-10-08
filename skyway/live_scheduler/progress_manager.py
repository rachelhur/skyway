"""Persistent session progress and observing-history management for the live scheduler."""

import json
import os
from skyway.ephemerides import time_utils, ephemerides
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class ProgressManager:
    """Track the current observing session and log completed fields to disk."""

    def __init__(
        self,
        output_dir,
        clock=None,
        session_id=None,
        start_time=None,
        start_sun_elevation=None,
        stop_time=None,
        stop_sun_elevation=None,
    ):
        """
        Initialize session metadata, output paths, and persisted history.

        Arguments
        ---------
        output_dir: str
            Directory where observing history should be written.
        clock: Clock, optional
            Clock instance to use for simulated time management. Default assumes the
            real-time clock.
        session_id: str, optional
            Explicit observing-session identifier. If omitted, one is generated.
        start_time: float, optional
            Unix timestamp for the start of the observing session.
        start_sun_elevation: float, optional
            Sun elevation threshold for the start of the observing session.
        stop_time: float, optional
            Unix timestamp for the end of the observing session.
        stop_sun_elevation: float, optional
            Sun elevation threshold for the end of the observing session.
        """

        self.output_dir = output_dir
        self.clock = clock or time_utils.Clock()
        self.session_id = session_id or self._generate_session_id()
        self.history_file = os.path.join(
            self.output_dir, f"observing_log_{self.session_id}.jsonl"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.completed_fields = self._load_history()

        # XXX sun-el-based conditions are tricky, since we can't reliably distinguish
        # between rising and setting without additional logic and more assumptions,
        # the user needs to be made super aware of for safety reasons. We should add
        # some extra checks to make sure that the user isn't doing anything really
        # stupid. Maybe an additional function that has hardcoded safety thresholds to
        # make sure nothing runs in the unsafe time, unless we're explicitly using the
        # mock API for testing purposes where no communication is happening with the
        # real telescope control system.
        # For instance, with starting/stopping on only sun el (and they use the same el)
        # then the scheduler script won't enter the run loop until after the sun is set,
        # which wastes valuable early setup time.
        self.start_time = start_time
        self.start_sun_elevation = start_sun_elevation
        self.stop_time = stop_time
        self.stop_sun_elevation = stop_sun_elevation
        if self.start_sun_elevation is not None:
            raise NotImplementedError(
                "sun-elevation-based start conditions are not implemented yet."
            )
        if self.stop_sun_elevation is not None:
            raise NotImplementedError(
                "sun-elevation-based stop conditions are not implemented yet."
            )
        if self.start_time is None or self.stop_time is None:
            raise ValueError(
                "start_time and stop_time must be specified for now."
            )
            

    # XXX add a log sink so important scheduler messages are also written to disk

    def _generate_session_id(self):
        """Map a continuous observing block across midnight to a single session date."""
        from datetime import timedelta

        # map the current time to a start date; if before noon use the previous day
        local_now = time_utils.unix_to_local_datetime(self.clock.now())
        if local_now.hour < 12:
            session_date = local_now - timedelta(days=1)
        else:
            session_date = local_now
        session_date = session_date.strftime("%Y-%m-%d")
        if abs(self.clock.offset) > 1e-9:
            session_date = f"sim_{session_date}"
        logger.info(f"[Progress] Generated session ID: {session_date}")
        return session_date

    def _load_history(self):
        """Load completed field IDs from the existing JSONL history file."""

        # resume by replaying the existing JSONL history file line by line
        fields = []
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                for line in f:
                    line = line.strip()

                    # ignore blank lines so a partial write does not break recovery
                    if not line:
                        continue

                    # skip malformed lines instead of failing the whole restart
                    try:
                        fields.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.info(
                            f"[Progress] Skipping malformed history line in {self.history_file}."
                        )
                        continue

            logger.info(
                f"[Progress] Resumed with {len(fields)} completed fields from {self.history_file}."
            )
        return pd.DataFrame(fields).convert_dtypes()

    def record_completion(self, obs_row):
        """
        Append a completed observation to the session history.

        Arguments
        ---------
        obs_row: dict or pandas.Series
            Completed observation containing at least a field_id
        """
        # store each JSON object sequentially on a new line as they are completed
        obs_dict = obs_row.to_dict() if hasattr(obs_row, "to_dict") else dict(obs_row)
        with open(self.history_file, "a") as f:
            f.write(json.dumps(obs_dict) + "\n")
        self.completed_fields = pd.concat(
            [self.completed_fields, pd.DataFrame([obs_dict])],
            ignore_index=True
        ).convert_dtypes()

    def check_start_condition(self):
        """
        Return whether conditions have been met to start the observing session.
        """
        # get current conditions
        now = self.clock.now()
        ra, dec = ephemerides.get_source_ra_dec("sun", time=now)
        az, el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=now)

        # require both time and sun elevation start conditions to be met if specified
        if self.start_time is not None and now < self.start_time:
            return False
        if self.start_sun_elevation is not None and el > self.start_sun_elevation:
            return False
        return True

    def check_end_condition(self):
        """
        Return whether conditions have been met to end the observing session.
        """
        # get current conditions
        now = self.clock.now()
        ra, dec = ephemerides.get_source_ra_dec("sun", time=now)
        az, el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=now)

        # end if either time or sun elevation end conditions are met
        if self.stop_time is not None and now >= self.stop_time:
            return True
        if self.stop_sun_elevation is not None and el > self.stop_sun_elevation:
            return True
        return False
