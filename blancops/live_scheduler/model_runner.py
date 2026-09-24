"""Model-runner interfaces for live scheduler chunk generation.

This module defines the contract used by the scheduler to request a candidate chunk of
observations and provides two implementations:

- MockModelRunner: stochastic sky-walk generator for demos and integration tests.
- AIModelRunner: placeholder for the production ML-backed model.
"""

import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from blancops.configs.constants import IDX2FILTER
from blancops.configs.experiment_schema import ActionConstraints
from blancops.data.features.glob_features import get_night_boundaries
from blancops.environment.live_env import LiveBlancoEnv
from blancops.ephemerides.time_utils import Clock
from blancops.math import geometry, units
from blancops.ephemerides import ephemerides
from blancops.data.lookup_tables import LookupTables
from blancops.ephemerides.ephemerides import HealpixGrid
from blancops.rl.agent_factory import AgentFactory
from blancops.survey.profiles import DES

logger = logging.getLogger(__name__)


class ModelRunner(ABC):
    """
    Abstract interface for observation-chunk generation.

    Implementations are expected to return a pandas DataFrame with one row per proposed
    observation. Downstream components currently expect columns similar to "time", "ra",
    "dec", and "filter".
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the model runner with any necessary setup, such as loading model
        weights or setting up internal state.
        """

        pass

    @abstractmethod
    def generate_chunk(self, telemetry, available_fields, masked_field_ids, priority_trigger, chunk_size):
        """
        Generate a chunk of proposed observations.

        Arguments
        ---------
        telemetry: dict
            Current telescope/sky state dictionary.
        available_fields: list
            Candidate field set.
        masked_field_ids: Iterable[int] or None
            Field ids to drop from the action space for this chunk.
        priority_trigger: bool
            When True, the env masks all non-priority-1 fields until priority-1
            work is complete (see LiveBlancoEnv.set_priority_trigger).
        chunk_size: int
            Number of sequential observations to propose.

        Returns
        -------
            pandas.DataFrame: Proposed observation chunk.
        """
        pass

    def record_visit(self, obs_row) -> None:
        """Notify the runner that an observation was submitted to the telescope.

        Called by the orchestrator immediately after each hardware submission so
        that subsequent rollouts reflect the running visit history. Default is a
        no-op; AI-backed runners override to update their internal env.
        """
        pass

    def resume_interrupted_session(self, completed_obs) -> None:
        """Seed the runner with this night's persisted visit history on restart.

        Called once by the orchestrator before the scheduling loop so an
        AI-backed runner can rebuild its env's cumulative survey state after a
        session was terminated mid-night. Default is a no-op; runners with no
        persistent state ignore it.
        """
        pass


class MockModelRunner(ModelRunner):
    """Randomized mock implementation used for development and dry runs."""

    def __init__(self, clock=None):
        self.clock = clock or Clock()
        self.current_field_id = 0

    def generate_next_observation(self, telemetry, masked_field_ids):
        """
        Sample a valid next pointing near the current one.

        The sample is accepted only if it:
        - is at least 1.5 deg from the current pointing,
        - is at least 5 deg from any masked field,
        - is above 30 deg elevation.
        """

        current_ra, current_dec = telemetry["pointing_ra"], telemetry["pointing_dec"]
        ra, dec = current_ra, current_dec

        # rejection-sample nearby pointings until one passes all validity checks
        valid = False
        while not valid:
            drad = 10 * units.deg
            delta_ra = random.uniform(-drad, drad) / np.cos(current_dec)
            delta_dec = random.uniform(-drad, drad)
            ra = (current_ra + delta_ra) % (360 * units.deg)
            dec = np.clip(current_dec + delta_dec, -90 * units.deg, 90 * units.deg)

            # Ensure minimum step size from current pointing.
            angsep = geometry.angular_separation((ra, dec), (current_ra, current_dec))
            valid_currentangsep = angsep > 1.5 * units.deg  # 1.5deg minimum step
            if not valid_currentangsep:
                continue

            # keep away from user/system-masked fields
            #if len(masked_field_ids) > 0:
            if False:  # XXX temporarily disable this check until we have a catalog
                angsep = geometry.angular_separation(
                    (ra, dec), masked_field_ids[["ra", "dec"]].values.T
                )
                valid_angsep = np.all(angsep > 5 * units.deg)  # 5deg threshold
            else:
                valid_angsep = True
            if not valid_angsep:
                continue

            # enforce telescope visibility via elevation floor
            az, el = ephemerides.equatorial_to_topographic(ra, dec, time=self.clock.now())
            valid_el = el > 30 * units.deg  # 30deg elevation limit

            valid = valid_currentangsep and valid_angsep and valid_el

        return ra, dec

    def generate_chunk(self, telemetry, available_fields, masked_field_ids, priority_trigger=False, chunk_size=10):
        """
        Generate a mock chunk as a short random walk in sky coordinates.

        Note:
            available_fields is currently unused but included to preserve interface
            compatibility with other production model runners. priority_trigger is
            accepted for parity but ignored: the mock has no catalog to gate.
        """

        logger.info(
            "[Model] Generating mock observing chunk based on telemetry and field masks..."
        )
        out = []

        # start from the current telescope pointing and walk forward
        ra, dec = telemetry["pointing_ra"], telemetry["pointing_dec"]
        for i in range(chunk_size):
            ra, dec = self.generate_next_observation(
                telemetry={"pointing_ra": ra, "pointing_dec": dec},
                masked_field_ids=masked_field_ids,
            )

            # keep output schema aligned with scheduler expectations
            out.append(
                {
                    "field_id": self.current_field_id,
                    "time": i,
                    "ra": ra,
                    "dec": dec,
                    "filter": random.choice(["g", "r", "i", "z", "Y"]),
                }
            )
            self.current_field_id += 1

        return pd.DataFrame(out)


class AIModelRunner(ModelRunner):
    def __init__(self, model_path_or_alias: str, field_lookup_dir: Path, fields_path: Path = None,
                 obs_history_path: Path = None,
                 device: str = "cpu", field_choice_method: str = "interp",
                 mode='test', clock=None, sun_elevation_deg=DES.sun_el_limit, seeing_window="15m"):
        self.device = device
        # self.TESTING_MODE = mode == 'test' # XXX remove before production
        self.clock = clock or Clock()

        # Fields and Lookups
        self.fields_dir = Path(field_lookup_dir)
        self.lookups = self._get_lookups(fields_path, self.fields_dir)

        # Prior-survey backlog (visited x out of target before this session).
        # Replayed ahead of the session's own logged visits in
        # resume_interrupted_session so the env's counts reflect both.
        self._backlog_history = self._load_backlog(obs_history_path)

        # Model
        self._build_agent(model_path_or_alias=model_path_or_alias, field_choice_method=field_choice_method)
        logger.info(f"Loaded model weights from {model_path_or_alias} into memory.")

        # if self.TESTING_MODE:
        now_ts = self.clock.now()
        zenith_ra, zenith_dec = ephemerides.get_source_ra_dec("zenith", time=now_ts)
        telemetry = {'ra': zenith_ra, 'dec': zenith_dec, 'filter_idx': 0, 'timestamp': now_ts}
        # else:
        #     pass

        self.env = self._build_env(
            telemetry_now=telemetry,
            sun_elevation_deg=sun_elevation_deg,
            seeing_window=seeing_window
        )
        self.hpGrid = HealpixGrid(nside=self.cfg.data.nside, is_azel="azel" in self.cfg.data.action_space)

    def _build_agent(self, model_path_or_alias, field_choice_method):
        # Agent and Model
        factory = AgentFactory() # Defaults to workspace().deployable_models
        self.agent, self.cfg, self.norm_stats = factory.build_agent(
            model_path_or_alias=model_path_or_alias,
            lookups=self.lookups,
            field_choice_method=field_choice_method,
            device=self.device
        )

    def _build_env(self, telemetry_now, sun_elevation_deg, seeing_window):
        constraints_cfg = ActionConstraints(sun_el_limit=sun_elevation_deg) # Uses default constraints
        env = LiveBlancoEnv(
            cfg=self.cfg,
            constraints_cfg=constraints_cfg,
            lookups=self.lookups,
            norm_stats=self.norm_stats,
            telemetry_init=telemetry_now,
            seeing_window=seeing_window
        )
        return env

    @staticmethod
    def _load_backlog(obs_history_path):
        """Load a prior observing-history CSV into a completed-visit frame.

        Args
        ----
        obs_history_path : Path or None
            CSV with columns field_id, filter, timestamp. None disables seeding.

        Returns
        -------
        pandas.DataFrame or None
            The loaded history, or None when no path was given.
        """
        if obs_history_path is None:
            return None
        history = pd.read_csv(obs_history_path)
        logger.info(
            "[Model] Loaded %d backlog visits from %s.",
            len(history), obs_history_path,
        )
        return history

    @staticmethod
    def _get_lookups(fields_path, fields_dir):
        if fields_path:
            lookups = LookupTables.build_lookups_from_fields(fields_path=fields_path, outdir=fields_dir, write_to_disk=True)
        else:
            lookups = LookupTables.load_from_dir(fields_dir, include_historic=False)
        return lookups

    def generate_chunk(self, telemetry=None, available_fields=[], masked_field_ids=[],
                       priority_trigger=False, chunk_size=10,
                       new_fields=None, new_lookup_dir=None) -> pd.DataFrame:
        """Schedule a chunk of `chunk_size` observations given current state.

        available_fields is accepted but unused. masked_field_ids is a list of field
        ids (or None) that the env drops from the action space for this chunk, on
        top of the priority gate driven by priority_trigger. Syncs the env to
        telemetry, applies the masks, runs a snapshotted rollout, then restores
        the env so live state is not mutated.
        """
        telemetry = self.resolve_rollout_telemetry(telemetry)
        self.update_lookups(new_fields, new_dir=new_lookup_dir)

        self.env.set_priority_trigger(priority_trigger)
        self.env.set_field_mask(masked_field_ids)

        self.env.sync_telemetry(telemetry)
        rollout_snapshot = self.env.save_snapshot()
        obs, info = self.env.get_obs(), self.env.get_info()

        logger.info("[Model] Generating state features and running inference...")
        proposed_schedule = self._rollout(init_obs=obs, init_info=info, chunk_size=chunk_size)

        self.env.restore_snapshot(rollout_snapshot)
        return proposed_schedule

    def record_visit(self, obs_row) -> None:
        """Update the live env's visit history after a hardware submission."""
        self.env.record_visit(obs_row)

    def resume_interrupted_session(self, completed_obs) -> None:
        """Rebuild the live env's survey state from the backlog plus this session's visits.

        The static backlog (``_backlog_history``) is replayed every restart and
        the session's own logged visits are accumulated on top, since
        ``LiveBlancoEnv.resume_interrupted_session`` rebuilds counts from a
        single frame rather than incrementally.
        """
        frames = [
            f for f in (self._backlog_history, completed_obs)
            if f is not None and len(f) > 0
        ]
        combined = pd.concat(frames, ignore_index=True) if frames else completed_obs
        self.env.resume_interrupted_session(combined)

    def resolve_rollout_telemetry(self, telemetry: pd.Series | dict) -> dict:
        """Normalize raw client telemetry to env-canonical form.

        Renames key aliases (``pointing_ra`` → ``ra``, etc.), sets ``timestamp`` as
        ``last_exposure_estimated_start_time``, and gets filter from
        ``last_exposure``. Works on a copy — does not mutate the input dict.
        """
        tel = dict(telemetry)
        if 'pointing_ra' in tel:
            tel.setdefault('ra', tel.pop('pointing_ra'))
        if 'pointing_dec' in tel:
            tel.setdefault('dec', tel.pop('pointing_dec'))
        ts = tel.pop('last_exposure_estimated_start_time', None)
        if ts is None or not np.isfinite(ts):  # no exposure submitted yet -> sentinel anchor
            ts = self.clock.now()
        tel.setdefault('timestamp', ts)
        if tel.get('last_exposure', None) is not None:
            filt = tel['last_exposure'].get('filter')
            tel['filter'] = filt if filt in IDX2FILTER.values() else 'g'

        return tel

    def update_lookups(self, new_fields_path, new_dir=None):
        if new_fields_path:
            new_lookups = LookupTables.build_lookups_from_fields(
                fields_path=new_fields_path, write_to_disk=False)
            self.lookups = self.lookups.merge(new_lookups, new_dir=new_dir)
            self.env.refresh_lookups(self.lookups)
            self.agent.lookups = self.lookups

    def _rollout(self, init_obs: dict, init_info: dict, chunk_size: int) -> pd.DataFrame:
        proposed_schedule = {'bin_idx': [],
                    'field_id': [],
                    'field_name': [],
                    'filter': [],
                    'expTime': [],
                    'timestamp': [],
                    'ra': [],
                    'dec': [],
                    }

        info = self.env.set_constraints(airmass_limit=2.5, sun_el_limit=-10.5)

        obs, info = init_obs, init_info
        for i in range(chunk_size):
            bin_idx, filter_idx, field_id = self.agent.choose_bin_filter_field(obs, info, self.hpGrid)
            filter = IDX2FILTER[filter_idx]
            actions = {'bin': np.int32(bin_idx), 'field_id': np.int32(field_id), 'filter_idx': np.int32(filter_idx)}

            proposed_schedule['bin_idx'].append(bin_idx)
            proposed_schedule['field_id'].append(field_id)
            proposed_schedule['filter'].append(filter)
            proposed_schedule['timestamp'].append(info.get('timestamp'))

            ra, dec = self.lookups.fields[["ra", "dec"]].loc[field_id]
            field_name = f"{field_id}_{ra:.5f}_{dec:.5f}_{filter}"
            exp_time = float(self.lookups.fidfilt_exptime[int(field_id), int(filter_idx)])

            proposed_schedule['field_name'].append(field_name)
            proposed_schedule['expTime'].append(exp_time)
            proposed_schedule['ra'].append(ra)
            proposed_schedule['dec'].append(dec)

            obs, reward, terminated, truncated, info = self.env.step(actions)
            if terminated or truncated: #ie, end of night - orchestrator default stops this, but doesn't hurt to have extra check here
                break

        for key, val in proposed_schedule.items():
            if key in ['bin_idx', 'field_id', 'timestamp']:
                dtype = np.int64
            elif key in ['ra', 'dec', 'expTime']:
                dtype = np.float64
            elif key in ['filter', 'field_name']:
                dtype = str
            proposed_schedule[key] = np.array(val, dtype=dtype)

        return pd.DataFrame(proposed_schedule)
