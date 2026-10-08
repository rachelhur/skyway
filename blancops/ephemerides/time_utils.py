import re
import warnings
from datetime import datetime, timezone, timedelta
from dateutil.parser import UnknownTimezoneWarning, parse
from pandas import Timedelta


class Clock:
    """
    UTC clock with an optional fixed offset to simulate different testing times.

    Methods
    -------
    now: Return the current UTC timestamp, optionally without the stored offset.
    convert: Apply the stored offset to a timestamp expressed in UTC seconds.
    """

    def __init__(self, offset=0.0):
        """
        Initialize the clock with an optional offset.

        Arguments
        ---------
        offset: float [0.0]
            Number of seconds to offset the clock by. Default is 0 (real current time).
            Use this to simulate running code at a different time for testing purposes.
        """
        self.offset = float(offset)

    def now(self, real=False):
        """
        Return the current UTC timestamp as simulated by the clock.

        Arguments
        ---------
        real: bool [False]
            If True, return the real current UTC time regardless of simulated offset.

        Returns
        -------
        float
            Current simulated UTC timestamp in seconds.
        """
        current = utc_now()
        return current if real else current + self.offset

    def convert(self, timestamp):
        """
        Convert a real-world UTC timestamp to the simulated clock's time by applying the
        stored offset.

        Arguments
        ---------
        timestamp: float, str, datetime
            Time variable to convert

        Returns
        -------
        float
            Converted UTC timestamp according to the simulated clock.
        """
        return standardize_time(timestamp) + self.offset_seconds


def utc_now():
    """
    Return the current time in UTC as UNIX timestamp.
    """
    return datetime.now(tz=timezone.utc).timestamp()


def datetime_to_unix(dt, tz=timezone.utc):
    """
    Convert a datetime object to a UNIX timestamp.

    Arguments
    ---------
    dt: datetime
        Datetime object to convert. Should be timezone-aware.
    tz: timezone
        Timezone to use for conversion if dt is timezone-naive. Default is UTC.

    Returns
    -------
    float
        Corresponding UNIX timestamp.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.timestamp()


def unix_to_datetime(ts):
    """
    Convert a UNIX timestamp to a timezone-aware datetime object in UTC.

    Arguments
    ---------
    ts: float
        UNIX timestamp in UTC to convert.

    Returns
    -------
    datetime
        Corresponding timezone-aware datetime object in UTC.
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def unix_to_local_datetime(ts):
    """
    Convert a UNIX timestamp in UTC to a datetime object in the local timezone.

    Arguments
    ---------
    ts: float
        UNIX timestamp in UTC to convert.

    Returns
    -------
    datetime
        Corresponding timezone-aware datetime object in the local timezone.
    """
    return unix_to_datetime(ts).astimezone()


def standardize_time(t, strict=False):
    """
    Ensures a time variable is in standard format: UNIX timestamp in UTC. Assumes that
    numerical inputs are already in correct format. Parses string inputs using dateutil
    for flexibility. Objects without a clear timezone are assumed to be in UTC.

    Arguments
    ---------
    t: float, str, datetime
        Time variable to standardize.
    strict: bool [False]
        For strings, raise ValueError instead of guessing a time zone: reject 'UTC-3' /
        'GMT-3' (dateutil reverses their sign) and abbreviations dateutil does not know
        (it would assume UTC). Numeric offsets such as '-03:00' are accepted.

    Returns
    -------
    float
        UNIX timestamp in UTC.
    """

    # helper function to check if a string can be parsed as a number
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # parse strings into numbers if possible else datetime; coerce everything
    # else numerically.
    if isinstance(t, str):
        if is_number(t):
            return float(t)
        if not strict:
            dt = parse(t)
        else:
            if re.search(r"(UTC|GMT)\s*[+-]", t, flags=re.IGNORECASE):
                raise ValueError(
                    f"{t!r}: write the UTC offset as a number, e.g. 2026-11-03T22:30-03:00 "
                    f"('UTC-3' and 'GMT-3' are read with the opposite sign)"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error", UnknownTimezoneWarning)
                try:
                    dt = parse(t)
                except (ValueError, OverflowError, UnknownTimezoneWarning):
                    raise ValueError(
                        f"{t!r} is not a valid time; use a UTC date and time such as 2026-11-04T01:30, a time "
                        f"with a numeric UTC offset such as 2026-11-03T22:30-03:00, or unix seconds"
                    )
    elif isinstance(t, datetime):
        dt = t
    else:
        # Numeric input, including numpy scalars (np.float32 / np.int*) that are
        # not subclasses of the builtin float. float() coerces them to a native
        # timestamp and raises TypeError for genuinely unsupported types.
        try:
            return float(t)
        except (TypeError, ValueError):
            raise ValueError("Unsupported time format")

    # assume datetimes without timezone info are in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def standardize_timedelta(t):
    """
    Ensures a time variable is in standard format: float seconds. Assumes that
    numerical inputs are already in correct format. Attempts to parse string inputs
    as numerical values first, then as pandas.Timedelta inputs.

    Arguments
    ---------
    t: float, str, timedelta
        Time variable to standardize.

    Returns
    -------
    float
        Time delta in seconds
    """

    # helper function to check if a string can be parsed as a number
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # keep numerical inputs as is, parse strings into numbers if possible else timedelta
    if isinstance(t, (int, float)):
        return float(t)
    elif isinstance(t, str):
        if is_number(t):
            return float(t)
        td = Timedelta(t)
    elif isinstance(t, (timedelta, Timedelta)):
        td = Timedelta(t)
    else:
        raise ValueError("Unsupported time format")

    return td.total_seconds()
