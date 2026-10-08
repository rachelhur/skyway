"""Telescope Client adapters for the live scheduler.

This module defines the scheduler-facing telescope interface and provides:

- MockTelescopeClient: local simulation of pointing, slew, and exposure timing.
- BlancoTelescopeClient: placeholder wrapper for the real observatory control path.
"""

from abc import ABC, abstractmethod
from skyway.math import units, geometry
from skyway.ephemerides import ephemerides, time_utils
from skyway.live_scheduler.scl import SCL
from skyway.data_quality.seeing import Seeing, DatabaseSeeing
import json
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class TelescopeClient(ABC):
    """Abstract interface for telescope-control interactions used by the scheduler."""

    @abstractmethod
    def get_telemetry(self):
        """
        Return current telemetry needed by scheduling logic.

        Returns
        -------
        dict
            Current telescope state, including current pointing coordinates.
        """

        pass

    @abstractmethod
    def check_exposure_status(self):
        """
        Report whether the current exposure is complete.

        Returns
        -------
        bool
            True when the system is ready for the next submission.
        """

        pass

    @abstractmethod
    def submit_observation(self, obs_row):
        """
        Submit a single observation request to the control system.

        Arguments
        ---------
        obs_row: dict or pandas.Series
            Observation request containing at least RA, Dec, and filter fields.
        """

        pass

    @abstractmethod
    def check_telemetry_change(self):
        """
        Report whether the telemetry has changed meaningfully since the last check.

        Returns
        -------
        bool
            True if the pointing or other state has changed enough to warrant replanning.
        """

        pass

    @abstractmethod
    def close(self):
        """
        Clean up any open connections or resources when the client is no longer needed.
        """

        pass


class MockTelescopeClient(TelescopeClient):
    """In-memory telescope simulator for development and integration testing."""

    def __init__(self, exposure_duration=90, clock=None, seeing_window="15m"):
        """Initialize mock timing and initial pointing state.

        Arguments
        ---------
        exposure_duration: float [90]
            Simulated exposure time in seconds.
        clock: time_utils.Clock [None]
            Optional custom clock for testing. By default, uses the real current time.
        seeing_window: str ["15m"]
            Time window for recent seeing measurements.
        """

        # model internal state to simulate exposure timing
        self.clock = clock or time_utils.Clock()
        self.last_exposure_submit_time = -float("inf")
        self.exposure_duration = exposure_duration
        self.slew_time = 0

        # track current pointing to model stepping through observations
        self.current_ra, self.current_dec = ephemerides.get_source_ra_dec(
            "zenith", time=self.clock.now()
        )
        self.ra_changed_since_last_check = False
        self.dec_changed_since_last_check = False

        # track the last submitted observation, initialized to dummy values
        self.last_submitted_obs_row = pd.Series({
            "ra": self.current_ra,
            "dec": self.current_dec,
            "filter": None,
        })

        # initialize seeing data, which remains empty for the mock client
        self.seeing = Seeing(window=seeing_window)

        logger.info("[Client] Initialized mock telescope client.")

    def get_telemetry(self, print_data=False):
        """Return the currently simulated telescope telemetry."""
        telemetry_data = {
            "last_exposure": self.last_submitted_obs_row,
            "last_exposure_submit_time": self.last_exposure_submit_time,
            "last_exposure_estimated_start_time": self.last_exposure_submit_time + self.slew_time,
            "last_exposure_estimated_end_time": self.last_exposure_submit_time + self.slew_time + self.exposure_duration,
            "pointing_ra": self.current_ra,
            "pointing_dec": self.current_dec,
            "seeing": self.seeing.raw,
        }
        if print_data:
            logger.info(f"[Client] Telemetry data:\n{telemetry_data}")
        return telemetry_data

    def check_telemetry_change(self):
        """
        Returns True if pointing has changed meaningfully. Currently, this check is
        turned off and always returns False for testing.
        """
        changed = self.ra_changed_since_last_check or self.dec_changed_since_last_check
        self.ra_changed_since_last_check = False
        self.dec_changed_since_last_check = False
        return changed

    def check_exposure_status(self):
        """Return True when simulated slew+exposure time has elapsed."""

        # compare elapsed wall-clock time against modeled slew + exposure duration
        delta = self.clock.now() - self.last_exposure_submit_time
        return delta > self.slew_time + self.exposure_duration

    def submit_observation(self, obs_row, exp_time=None):
        """Submit an observation into the mock queue and update simulator state."""

        self.last_submitted_obs_row = obs_row

        # approximate slew time from angular separation between old/new pointings
        angsep = geometry.angular_separation(
            (self.current_ra, self.current_dec), (obs_row["ra"], obs_row["dec"])
        )
        self.slew_time = geometry.blanco_slew_time(angsep) / units.second

        # update internal state to reflect the new observation
        self.last_exposure_submit_time = self.clock.now()
        self.current_ra = obs_row["ra"]
        self.current_dec = obs_row["dec"]
        if exp_time is not None:
            self.exposure_duration = exp_time

        logger.info(
            f"[Client] SUBMITTED: RA={obs_row['ra']}, DEC={obs_row['dec']}, FILTER={obs_row['filter']}"
        )
        logger.info(
            f"[Client] Estimated time until ready for next submission: {self.slew_time + self.exposure_duration:.1f}s."
        )

    def close(self):
        """No resources to clean up for the mock client."""
        logger.info("[Client] Closing mock telescope client (no resources to clean up).")


class BlancoSCLTelescopeClient(TelescopeClient):
    """
    Blanco telescope control-system integration using SCL network for commands and a
    postgres database for seeing monitoring.
    """

    def __init__(self, propid=None, server_ip="observer4.ctio.noao.edu", server_port=20000, clock=None, seeing_window="15m", daytime_testing=False, override_error=False):
        """
        Initialize and confirm the connection to the control system.

        Arguments
        ---------
        propid: str
            Proposal ID to include with each observation submission.
        server_ip: str ["observer4.ctio.noao.edu"]
            IP address of the SCLN server.
        server_port: int [20000]
            Port number of the SCLN server.
        clock: time_utils.Clock [None]
            Optional custom clock for testing. By default, uses the real current time.
        seeing_window: str ["15m"]
            Time window for recent seeing measurements.
        daytime_testing: bool [False]
            When True, submit harmless day-time test exposures (dark exposures with the
            "block" filter) instead of real science exposures. When the sun is above
            -10.5 degrees elevation, this must be True or initialization fails.
        override_error: bool [False]
            When True, allow submission of real science exposures even when the sun is above
            -10.5 degrees elevation. THIS IS DANGEROUS AND SHOULD ONLY BE USED FOR 
            TESTING PURPOSES UNDER DIRECT SUPERVISION OF THE STAFF.
        Raises
        ------
        RuntimeError
            If the sun is above -10.5 degrees elevation and daytime_testing is False.
        """

        # track time management
        self.clock = clock or time_utils.Clock()
        self.daytime_testing = daytime_testing
        self.last_exposure_submit_time = -float("inf")
        self.last_exposure_duration = 0
        self.last_exposure_estimated_slew_time = 0

        # failsafe: refuse to submit real science exposures while the sun is up
        sun_ra, sun_dec = ephemerides.get_source_ra_dec(
            "sun", time=self.clock.now(real=True)
        )
        _, sun_el = ephemerides.equatorial_to_topographic(
            sun_ra, sun_dec, time=self.clock.now(real=True)
        )
        sun_el_deg = sun_el / units.deg
        if sun_el_deg > -10.5 and not daytime_testing and not override_error:
            raise RuntimeError(
                f"[Client] Sun elevation is {sun_el_deg:.1f} deg (> -10.5 deg) but "
                "daytime_testing is False. Only day-time test exposures are allowed."
            )

        # Initialize the TCP/IP communication client
        logger.info(f"[Client] Attempting to connect to SCLN server at {server_ip}:{server_port}...")
        self.scl_client = SCL(server_ip, server_port)
        self.transaction_id = 0

        # check if connection was successful
        if self.scl_client.is_connected():
            logger.info(f"[Client] Initialized connection to SCLN server at {server_ip}:{server_port}.")
        else:
            logger.warning(f"[Client] WARNING: Could not connect to SCLN server at {server_ip}:{server_port}.")

        # initialize connection to the seeing database
        self.seeing = DatabaseSeeing(window=seeing_window)
        self.seeing_changed_since_last_check = False

        # track current pointing based on submissions
        self.current_time = self.clock.now()
        self.current_ra, self.current_dec = ephemerides.get_source_ra_dec("zenith", time=self.current_time)

        # track the last submitted observation for telemetry reporting
        self.last_submitted_obs_row = pd.Series()

        self.propid = propid

    def _build_base_message(self, msg_type):
        """Helper to construct the standard JSON envelope for SCLN messages."""
        if msg_type == "COMMAND":
            self.current_time = self.clock.now()
            cmd = {
                "type": "COMMAND",
                "source": "DECamAISched",
                "target": "SISPI",
                "timestamp": time_utils.unix_to_datetime(self.current_time).isoformat(timespec='milliseconds'),
                "transaction_id": str(self.transaction_id),
                "command": "EXPOSE",
            }
            self.transaction_id += 1
        elif msg_type == "TELEMETRY":
            cmd = {
                "type": "TELEMETRY",
                "command": "TELEMETRY",
            }
        else:
            raise ValueError(f"[Client] Unsupported message type: {msg_type}")
        return cmd

    def get_telemetry(self, print_data=False):
        """
        Fetch live telemetry.
        """
        cmd = self._build_base_message("TELEMETRY")

        # send a request for telemetry and parse the response
        telemetry_data = {}
        ra, dec = self.current_ra, self.current_dec
        try:
            response_str = self.scl_client.send_command(json.dumps(cmd))
            if not response_str:
                raise RuntimeError("No telemetry response received from SCLN server.")

            response = json.loads(response_str)
            telemetry_data = response.get("telemetry", {})

        except Exception as e:
            logger.exception(f"[Client] Error fetching telemetry: {e}")

        # tcs info has some pointing info, but not all fields guaranteed to be present
        # NB: the tcs report is a few minutes delayed, so it's better to track submits
        tcs = telemetry_data.get("tcs_infot", {})
        tcs_time = time_utils.standardize_time(val) if (val := tcs.get("time_recorded")) is not None else None
        tcs_ra = float(val) * units.deg if (val := tcs.get("tel_ra")) is not None else None
        tcs_dec = float(val) * units.deg if (val := tcs.get("tel_dec")) is not None else None
        tcs_az = float(val) * units.deg if (val := tcs.get("tel_az")) is not None else None
        tcs_el = float(val) * units.deg if (val := tcs.get("tel_el")) is not None else None
        tcs_zd = float(val) * units.deg if (val := tcs.get("tel_zd")) is not None else None
        tcs_airmass = float(val) if (val := tcs.get("airmass")) is not None else None
        tcs_ha = float(val) * units.deg if (val := tcs.get("tel_ha")) is not None else None

        # print data upon request
        if print_data:
            logger.info(f"[Client] Telemetry data:\n{telemetry_data}")

        # fetch seeing data
        changed = self.seeing.update()
        if not self.seeing.raw.empty:
            pred = self.seeing.predict(band='i', el=90 * units.deg, now=self.clock.now()) / units.arcsec
            logger.info(f"[Client] Current seeing prediction (i, zenith): {pred:.3f} arcsec")
        else:
            logger.info("[Client] No seeing data logged; dataframe empty.")
        self.seeing_changed_since_last_check = changed or self.seeing_changed_since_last_check

        return {
            "last_exposure": self.last_submitted_obs_row,
            "last_exposure_submit_time": self.last_exposure_submit_time,
            "last_exposure_estimated_start_time": self.last_exposure_submit_time + self.last_exposure_estimated_slew_time,
            "last_exposure_estimated_end_time": self.last_exposure_submit_time + self.last_exposure_estimated_slew_time + self.last_exposure_duration,
            "pointing_ra": ra,
            "pointing_dec": dec,
            "seeing": self.seeing.raw,
            "tel_time": tcs_time,
            "tel_ra": tcs_ra,
            "tel_dec": tcs_dec,
            "tel_az": tcs_az,
            "tel_el": tcs_el,
            "tel_zd": tcs_zd,
            "tel_airmass": tcs_airmass,
            "tel_ha": tcs_ha,
        }

    def check_telemetry_change(self):
        """Returns True if the seeing data has changed since the last telemetry check."""
        changed = self.seeing_changed_since_last_check
        self.seeing_changed_since_last_check = False
        return changed

    def check_exposure_status(self):
        """Return exposure readiness state from control system."""

        # send a request for telemetry
        cmd = self._build_base_message("TELEMETRY")
        try:
            response_str = self.scl_client.send_command(json.dumps(cmd))
            if not response_str:
                return False

            # server provides a bool indicating if it can accept the next EXPOSE command
            response = json.loads(response_str)
            return response.get("readyToExpose", False)

        except Exception as e:
            logger.exception(f"[Client] Error checking exposure status: {e}")
            return False

    def submit_observation(self, obs_row, exp_time=None):
        """Submit one observation request to the control system."""
        cmd = self._build_base_message("COMMAND")

        # map the desired observation to the command parameters expected by SCLN
        angsep = geometry.angular_separation(
            (self.current_ra, self.current_dec), (obs_row["ra"], obs_row["dec"])
        ) if self.current_ra is not None and self.current_dec is not None else 0
        self.last_exposure_estimated_slew_time = geometry.blanco_slew_time
        self.current_ra = float(obs_row.get("ra", self.current_ra))
        self.current_dec = float(obs_row.get("dec", self.current_dec))

        # day-time testing uses harmless dark exposures; else try reading from obs_row
        if self.daytime_testing:
            exp_type = "dark"
            filt = "block"
            comment = "DO NOT USE"
            exp_time = 10 if exp_time is None else exp_time
        else:
            exp_type = str(obs_row.get("expType", "object"))
            filt = str(obs_row.get("filter", "None"))
            comment = str(obs_row.get(
                "comment",
                f"Submitted by AI scheduler at {time_utils.unix_to_datetime(self.clock.now()).isoformat(timespec='seconds')}"
            ))
            exp_time = obs_row.get("expTime", 90) if exp_time is None else exp_time

        # build command parameters
        cmd["parameters"] = {
            "expTime": str(exp_time), # XXX examples had this as str, but directions say int
            "expType": exp_type,
            "propid": str(obs_row.get("propid", "UNKNOWN")) if self.propid is None else str(self.propid),
            "count": int(obs_row.get("count", 1)),
            "filter": filt,
            "ra": self.current_ra / units.degree,
            "dec": self.current_dec / units.degree,
            "object": str(obs_row.get("field_name", f"pointing_{self.current_time}")),
            "comment": comment,
        }

        # store the submitted observation for telemetry reporting
        self.last_submitted_obs_row = obs_row
        self.last_exposure_submit_time = self.clock.now()
        self.last_exposure_duration = float(cmd["parameters"]["expTime"])
        self.last_exposure_estimated_slew_time = geometry.blanco_slew_time(angsep) / units.second

        # send the command and wait for the synchronous response
        logger.info(f"[Client] SUBMIT: RA={cmd['parameters']['ra']}, DEC={cmd['parameters']['dec']}, FILTER={cmd['parameters']['filter']}")
        try:
            response_str = self.scl_client.send_command(json.dumps(cmd))
            response = json.loads(response_str) if response_str else {}

            if response.get("status") == "FAILED":
                logger.warning(f"[Client] EXPOSURE FAILED: {response.get('message')}")

            return response

        except Exception as e:
            logger.exception(f"[Client] Error submitting observation: {e}")
            return None

    def close(self):
        """Clean up the SCL client and seeing database connection."""
        self.scl_client.close()
        logger.info("[Client] Closed connection to SCLN server.")
        self.seeing.database.disconnect()
        logger.info("[Client] Closed connection to seeing database.")
