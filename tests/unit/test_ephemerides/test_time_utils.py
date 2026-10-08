"""`time_utils.standardize_time`: default (lenient) parsing and `strict=True` time-zone checks."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from dateutil.parser import UnknownTimezoneWarning

from blancops.ephemerides.time_utils import standardize_time

TS = 1793755800.0  # 2026-11-04 01:30 UTC = 2026-11-03 22:30 at CTIO (UTC-3)

SAME_MOMENT = [
    "2026-11-04T01:30",
    "2026-11-04 01:30",
    "2026-11-04T01:30:00Z",
    "2026-11-04 01:30 UTC",
    "2026-11-03T22:30-03:00",
    "2026-11-03 22:30 -0300",
    "2026-11-03T22:30-03",
    "Nov 4 2026 01:30",
    "1793755800",
]


@pytest.mark.parametrize("value", [TS, int(TS), np.float32(1.0), np.int64(TS)])
def test_numbers_pass_through(value):
    assert standardize_time(value) == float(value)


def test_naive_datetime_is_utc():
    assert standardize_time(datetime(2026, 11, 4, 1, 30)) == TS


def test_aware_datetime_keeps_its_offset():
    assert standardize_time(datetime(2026, 11, 3, 22, 30, tzinfo=timezone(timedelta(hours=-3)))) == TS


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("value", SAME_MOMENT)
def test_accepted_strings(value, strict):
    assert standardize_time(value, strict=strict) == TS


def test_date_without_time_is_midnight_utc():
    assert standardize_time("2026-11-04", strict=True) == TS - 1.5 * 3600


def test_default_keeps_dateutil_sign_for_utc_offset_names():
    # dateutil follows POSIX: 'UTC-3' is three hours east of UTC, so 22:30 'UTC-3' is 19:30 UTC
    assert standardize_time("2026-11-03 22:30 UTC-3") == TS - 6 * 3600


def test_default_assumes_utc_for_unknown_abbreviation():
    with pytest.warns(UnknownTimezoneWarning):
        assert standardize_time("2026-11-03 22:30 CLST") == TS - 3 * 3600


@pytest.mark.parametrize("value", ["2026-11-03 22:30 UTC-3", "2026-11-03 22:30 GMT-3", "2026-11-03 22:30 utc+3"])
def test_strict_rejects_utc_offset_names(value):
    with pytest.raises(ValueError, match="write the UTC offset as a number"):
        standardize_time(value, strict=True)


@pytest.mark.parametrize("value", ["2026-11-03 22:30 CLST", "2026-11-03 22:30 America/Santiago",
                                   "tonight", "2026-13-04T01:30"])
def test_strict_rejects_unknown_zones_and_bad_strings(value):
    with pytest.raises(ValueError, match="is not a valid time"):
        standardize_time(value, strict=True)


def test_strict_does_not_leak_warning_filter():
    with pytest.raises(ValueError):
        standardize_time("2026-11-03 22:30 CLST", strict=True)
    with pytest.warns(UnknownTimezoneWarning):
        standardize_time("2026-11-03 22:30 CLST")


def test_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported time format"):
        standardize_time(object())
