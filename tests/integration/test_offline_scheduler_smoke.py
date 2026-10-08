"""End to end: fields CSV (degrees) -> run-offline-scheduler -> SISPI observing scripts, one half night."""
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skyway.configs.paths import OfflineRunPaths
from skyway.data.lookup_tables import LookupTables
from skyway.environment.offline_env import resolve_observing_windows
from skyway.ephemerides import ephemerides
from skyway.io.file_io import read_schedule_csv
from skyway.rl.agent_factory import AgentFactory
from skyway.scripts import run_offline_scheduler

MODEL = "bc_v1"
NIGHT = "2026-06-24-half2"
AIRMASS_LIMIT = 2.0
SUN_EL_LIMIT = -12.0
REFERENCE = json.loads((Path(__file__).parents[1] / "unit" / "test_sispi" / "reference_sispi.json").read_text())


def _model_available() -> bool:
    try:
        AgentFactory().resolve_model_dir(MODEL)
        return True
    except FileNotFoundError:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _model_available(), reason=f"deployable model {MODEL!r} not installed"),
]


@pytest.fixture(scope="module")
def fields_deg() -> pd.DataFrame:
    """Twelve fields in degrees, g and r, one exposure each; a few never rise above the airmass limit."""
    rng = np.random.default_rng(0)
    ra = rng.uniform(220, 320, 12)
    dec = np.concatenate([rng.uniform(-60, -15, 10), [70.0, 75.0]])
    return pd.DataFrame([
        dict(RA=ra[i], DEC=dec[i], filter=f, count=1, exptime=90, field_name=f"F{i}")
        for i in range(12) for f in "gr"
    ])


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, fields_deg) -> Path:
    tmp = tmp_path_factory.mktemp("smoke")
    fields_csv = tmp / "fields.csv"
    fields_deg.to_csv(fields_csv, index=False)
    outdir = tmp / "run"
    argv = ["run-offline-scheduler", "-m", MODEL, "--fields", str(fields_csv), "-d", NIGHT, "-o", str(outdir),
            "--airmass_limit", str(AIRMASS_LIMIT), "--sun_el_limit", str(SUN_EL_LIMIT),
            "--save_observing_script", "--propid", "2026B-SMOKE", "--program", "smoke-test"]
    mp = pytest.MonkeyPatch()
    mp.setattr(sys, "argv", argv)
    try:
        run_offline_scheduler.main()
    finally:
        mp.undo()
        logger = logging.getLogger("skyway")
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
        logger.propagate = True
    return outdir


def test_outputs_written(run_dir):
    paths = OfflineRunPaths(run_dir)
    assert (paths.lookups / "fields_table.json").exists()
    assert (paths.nights / f"{NIGHT}.csv").exists()
    assert (paths.observing_scripts / f"{NIGHT}_sispi.json").exists()
    assert (paths.observing_scripts / "all_nights_sispi.json").exists()


def test_sispi_entries_match_reference_layout(run_dir):
    entries = json.loads((OfflineRunPaths(run_dir).observing_scripts / f"{NIGHT}_sispi.json").read_text())
    assert entries
    for e in entries:
        assert list(e) == list(REFERENCE[0])
        assert e["propid"] == "2026B-SMOKE"


def test_scheduled_fields_come_from_input_and_respect_counts(run_dir, fields_deg):
    entries = json.loads((OfflineRunPaths(run_dir).observing_scripts / f"{NIGHT}_sispi.json").read_text())
    by_name = fields_deg.drop_duplicates("field_name").set_index("field_name")
    targets = Counter({(r["field_name"], r["filter"]): r["count"] for _, r in fields_deg.iterrows()})
    observed = Counter((e["object"], e["filter"]) for e in entries)
    for (name, filt), n in observed.items():
        assert n <= targets[(name, filt)], (name, filt)
    for e in entries:
        assert e["RA"] == pytest.approx(by_name.loc[e["object"], "RA"] % 360, abs=1e-4)
        assert e["dec"] == pytest.approx(by_name.loc[e["object"], "DEC"], abs=1e-4)


def test_exposures_inside_night_and_airmass_limit(run_dir):
    window, = resolve_observing_windows(SUN_EL_LIMIT, observing_nights=[NIGHT])
    lookups = LookupTables.load_from_dir(OfflineRunPaths(run_dir).lookups)
    df = read_schedule_csv(OfflineRunPaths(run_dir).nights / f"{NIGHT}.csv")
    assert ((df["timestamp"] >= window.start_ts) & (df["timestamp"] <= window.end_ts)).all()
    ra = lookups.fields["ra"].to_numpy()[df["field_id"]]
    dec = lookups.fields["dec"].to_numpy()[df["field_id"]]
    el = np.array([ephemerides.equatorial_to_topographic(r, d, time=t)[1]
                   for r, d, t in zip(ra, dec, df["timestamp"])])
    assert (1 / np.sin(el) < AIRMASS_LIMIT).all()
