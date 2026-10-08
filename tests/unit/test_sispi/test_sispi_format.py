"""SISPI JSON output of `telescope.blanco.write_sispi`, checked against a reference script.

`reference_sispi.json` holds the first entries of the magic-spring script
(experiments/bc/old/magic-spring-tests/magic-spring_v1/magic-spring.json), whose
layout follows obztak's SISPI_DICT (kadrlica/obztak, obztak/field.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blancops.data.lookup_tables import LookupTables
from blancops.math import units
from blancops.survey.profiles import DES
from blancops.telescope.blanco import BLANCO, write_sispi

REFERENCE = json.loads((Path(__file__).parent / "reference_sispi.json").read_text())

FIELD_NAMES = ["fieldA", "fieldB", "fieldC"]
FIELD_RA_DEG = np.array([10.0, 359.99, 180.0])
FIELD_DEC_DEG = np.array([-30.0, -89.5, 5.0])

# 2026-06-23 night at CTIO: starts before UTC midnight, ends after
TIMESTAMPS = pd.to_datetime(
    ["2026-06-23T23:30:00", "2026-06-24T03:00:00", "2026-06-24T09:45:00"], utc=True
).astype("int64").to_numpy() / 1e9
SCHEDULED_FIELD_IDS = [0, 1, 2]
SCHEDULED_FILTERS = ["g", "Y", "r"]
SISPI_IDS = {"propid": "2026A-0001", "proposer": "tester", "program": "test-program"}


@pytest.fixture
def lookups(tmp_path: Path) -> LookupTables:
    """Three fields; exposure time set only for the scheduled (field, filter) pairs."""
    fields = pd.DataFrame({
        "field": FIELD_NAMES,
        "ra": FIELD_RA_DEG * units.deg,
        "dec": FIELD_DEC_DEG * units.deg,
    })
    fields.index.name = "field_id"
    counts = np.zeros((len(fields), len(DES.filter2idx)), dtype=int)
    exptime = np.zeros_like(counts, dtype=float)
    for fid, filt, t in zip(SCHEDULED_FIELD_IDS, SCHEDULED_FILTERS, [90.0, 45.0, 120.0]):
        counts[fid, DES.filter2idx[filt]] = 1
        exptime[fid, DES.filter2idx[filt]] = t
    return LookupTables(fields=fields, target_fidfilt_counts=counts, fidfilt_exptime=exptime, dir=tmp_path)


@pytest.fixture
def schedule_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": TIMESTAMPS,
        "field_id": SCHEDULED_FIELD_IDS,
        "filter": SCHEDULED_FILTERS,
    })


def _write(schedule_df, lookups, tmp_path, **kwargs) -> tuple[Path, list[dict]]:
    path = write_sispi(schedule_df, "2026-06-23-full", tmp_path, lookups, **{**SISPI_IDS, **kwargs})
    return path, json.loads(path.read_text())


def test_keys_and_order_match_reference(schedule_df, lookups, tmp_path):
    _, entries = _write(schedule_df, lookups, tmp_path)
    for entry in entries:
        assert list(entry) == list(REFERENCE[0])


def test_value_types_match_reference(schedule_df, lookups, tmp_path):
    _, entries = _write(schedule_df, lookups, tmp_path)
    for entry in entries:
        for key, ref_val in REFERENCE[0].items():
            if key == "propid":
                assert isinstance(entry[key], str)
            else:
                assert type(entry[key]) is type(ref_val), key


def test_constant_fields_match_reference(schedule_df, lookups, tmp_path):
    _, entries = _write(schedule_df, lookups, tmp_path)
    for entry in entries:
        for key in ("seqnum", "seqtot", "count", "expType", "wait", "comment"):
            assert entry[key] == REFERENCE[0][key], key


def test_one_entry_per_exposure_in_schedule_order(schedule_df, lookups, tmp_path):
    _, entries = _write(schedule_df, lookups, tmp_path)
    assert [e["object"] for e in entries] == [FIELD_NAMES[f] for f in SCHEDULED_FIELD_IDS]
    assert [e["filter"] for e in entries] == SCHEDULED_FILTERS
    expected_seqids = [f"datetime: {pd.Timestamp(t, unit='s', tz='UTC').isoformat(timespec='seconds')}" for t in TIMESTAMPS]
    assert [e["seqid"] for e in entries] == expected_seqids


def test_coordinates_in_degrees_and_in_range(schedule_df, lookups, tmp_path):
    _, entries = _write(schedule_df, lookups, tmp_path)
    ras = np.array([e["RA"] for e in entries])
    decs = np.array([e["dec"] for e in entries])
    assert np.all((ras >= 0) & (ras < 360))
    assert np.all((decs >= -90) & (decs <= 90))
    np.testing.assert_allclose(ras, FIELD_RA_DEG[SCHEDULED_FIELD_IDS], atol=1e-5)
    np.testing.assert_allclose(decs, FIELD_DEC_DEG[SCHEDULED_FIELD_IDS], atol=1e-5)


def test_exptime_from_lookups_and_positive(schedule_df, lookups, tmp_path):
    _, entries = _write(schedule_df, lookups, tmp_path)
    assert [e["expTime"] for e in entries] == [90, 45, 120]


def test_ids_written_to_every_entry(schedule_df, lookups, tmp_path):
    _, entries = _write(schedule_df, lookups, tmp_path)
    for entry in entries:
        for key, val in SISPI_IDS.items():
            assert entry[key] == val


def test_file_named_after_night(schedule_df, lookups, tmp_path):
    path, _ = _write(schedule_df, lookups, tmp_path)
    assert path.name == "2026-06-23-full_sispi.json"


def test_blanco_observing_script_writer_is_sispi():
    assert BLANCO.observing_script_writer is write_sispi
    assert DES.telescope.observing_script_writer is write_sispi


def test_filter_override(schedule_df, lookups, tmp_path):
    lookups.fidfilt_exptime[:, DES.filter2idx["i"]] = 60.0
    _, entries = _write(schedule_df, lookups, tmp_path, filter_override_val="i")
    assert [e["filter"] for e in entries] == ["i"] * len(entries)
    assert [e["expTime"] for e in entries] == [60] * len(entries)


def test_zero_exptime_raises(schedule_df, lookups, tmp_path):
    schedule_df["filter"] = ["z", "Y", "r"]
    with pytest.raises(ValueError, match="no positive exposure time"):
        _write(schedule_df, lookups, tmp_path)


def test_unknown_filter_raises(schedule_df, lookups, tmp_path):
    schedule_df["filter"] = ["u", "Y", "r"]
    with pytest.raises(KeyError):
        _write(schedule_df, lookups, tmp_path)


def test_out_of_order_timestamps_raise(schedule_df, lookups, tmp_path):
    with pytest.raises(ValueError, match="time order"):
        _write(schedule_df.iloc[::-1], lookups, tmp_path)


def test_missing_propid_raises(schedule_df, lookups, tmp_path):
    with pytest.raises(ValueError, match="propid"):
        _write(schedule_df, lookups, tmp_path, propid=None)


def test_missing_program_raises(schedule_df, lookups, tmp_path):
    with pytest.raises(ValueError, match="program"):
        _write(schedule_df, lookups, tmp_path, program=None)
