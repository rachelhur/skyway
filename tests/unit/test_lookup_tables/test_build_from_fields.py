"""`LookupTables.build_lookups_from_fields` on small synthetic fields tables."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skyway.data.lookup_tables import LookupTables
from skyway.survey.profiles import DES

G, R, Z = (DES.filter2idx[f] for f in "grz")


def _fields_deg() -> pd.DataFrame:
    """Three fields in degrees; fieldB has two filters, fieldC has a negative RA."""
    return pd.DataFrame({
        "RA": [10.0, 200.0, 200.0, -10.0],
        "Dec": [-30.0, 5.0, 5.0, -60.0],
        "filter": ["g", "r", "z", "g"],
        "count": [2, 1, 3, 1],
        "exptime": [90, 60, 120, 90],
        "field_name": ["fieldA", "fieldB", "fieldB", "fieldC"],
    })


def _build(df: pd.DataFrame, **kwargs) -> LookupTables:
    return LookupTables.build_lookups_from_fields(fields_df=df, outdir=Path("."), **{"radec_units": "deg", **kwargs})


def test_matrices_in_survey_filter_order():
    lk = _build(_fields_deg())
    assert lk.target_fidfilt_counts.shape == (3, DES.num_filters)
    expected_counts = np.zeros((3, DES.num_filters), dtype=int)
    expected_counts[0, G], expected_counts[1, R], expected_counts[1, Z], expected_counts[2, G] = 2, 1, 3, 1
    np.testing.assert_array_equal(lk.target_fidfilt_counts, expected_counts)
    assert lk.fidfilt_exptime[1, Z] == 120 and lk.fidfilt_exptime[1, G] == 0


def test_field_ids_follow_first_appearance():
    lk = _build(_fields_deg())
    assert list(lk.fields["field"]) == ["fieldA", "fieldB", "fieldC"]
    assert list(lk.fields.index) == [0, 1, 2]


def test_degrees_converted_and_ra_wrapped():
    lk = _build(_fields_deg())
    np.testing.assert_allclose(np.degrees(lk.fields["ra"]), [10.0, 200.0, 350.0])
    np.testing.assert_allclose(np.degrees(lk.fields["dec"]), [-30.0, 5.0, -60.0])


def test_csv_round_trip(tmp_path):
    fields_csv = tmp_path / "fields.csv"
    _fields_deg().to_csv(fields_csv, index=False)
    built = LookupTables.build_lookups_from_fields(
        fields_path=fields_csv, outdir=tmp_path / "lookups", write_to_disk=True, radec_units="deg",
    )
    loaded = LookupTables.load_from_dir(tmp_path / "lookups")
    pd.testing.assert_frame_equal(loaded.fields, built.fields, check_dtype=False)
    np.testing.assert_array_equal(loaded.target_fidfilt_counts, built.target_fidfilt_counts)
    np.testing.assert_array_equal(loaded.fidfilt_exptime, built.fidfilt_exptime)


def test_json_radians_matches_csv_degrees(tmp_path):
    df = _fields_deg()
    df["RA"], df["Dec"] = np.radians(df["RA"]), np.radians(df["Dec"])
    fields_json = tmp_path / "fields.json"
    df.to_json(fields_json, orient="records")
    from_json = LookupTables.build_lookups_from_fields(fields_path=fields_json, outdir=tmp_path, radec_units="rad")
    from_deg = _build(_fields_deg())
    np.testing.assert_allclose(from_json.fields[["ra", "dec"]].to_numpy(), from_deg.fields[["ra", "dec"]].to_numpy())
    np.testing.assert_array_equal(from_json.target_fidfilt_counts, from_deg.target_fidfilt_counts)


def test_default_field_names():
    lk = _build(_fields_deg().drop(columns=["field_name"]))
    assert list(lk.fields["field"]) == ["field_0", "field_1", "field_2"]


@pytest.mark.parametrize("change, match", [
    (lambda df: df.drop(columns=["exptime"]), "Missing columns"),
    (lambda df: df.assign(filter=["u", "r", "z", "g"]), "Unknown filter"),
    (lambda df: df.assign(Dec=[-30.0, 5.0, 5.0, -95.0]), r"Dec outside \[-90, 90\]"),
    (lambda df: df.assign(count=[2, 0, 3, 1]), "'count' must be a positive number"),
    (lambda df: df.assign(count=[2, -1, 3, 1]), "'count' must be a positive number"),
    (lambda df: df.assign(count=[2, np.nan, 3, 1]), "'count' must be a positive number"),
    (lambda df: df.assign(exptime=[90, 60, 0, 90]), "'exptime' must be a positive number"),
    (lambda df: df.assign(filter=["g", "r", "r", "g"]), "listed more than once"),
    (lambda df: df.assign(field_name=["fieldA", "fieldB", "fieldB2", "fieldC"]), "'field' varies within a field_id"),
    (lambda df: df.assign(field_name=["dup", "fieldB", "fieldB", "dup"]), "Field name 'dup'"),
    (lambda df: df.assign(field_id=[0, 1, 2, 2]), "multiple different field_ids"),
])
def test_bad_input_raises(change, match):
    with pytest.raises(ValueError, match=match):
        _build(change(_fields_deg()))


def test_degrees_passed_as_radians_raises():
    with pytest.raises(ValueError, match="may be in degrees"):
        _build(_fields_deg(), radec_units="rad")


def test_unknown_units_raise():
    with pytest.raises(ValueError, match="radec_units"):
        _build(_fields_deg(), radec_units="arcmin")


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "fields.txt"
    path.write_text("ra dec\n")
    with pytest.raises(ValueError, match="Unsupported fields file type"):
        LookupTables.build_lookups_from_fields(fields_path=path, radec_units="deg")


def test_radians_read_as_degrees_warns(caplog):
    df = _fields_deg()
    df["RA"], df["Dec"] = np.radians(df["RA"] % 360), np.radians(df["Dec"])
    with caplog.at_level("WARNING"):
        _build(df)
    assert "fits the radian range" in caplog.text
