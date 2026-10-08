from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class ConstraintSet:
    """Observability constraints for a telescope.

    A pointing is schedulable only when every constraint returns True.
    """

    # ------------------------------------------------------------------ #
    # Sky / atmosphere                                                     #
    # ------------------------------------------------------------------ #
    max_airmass: float
    """Upper bound on airmass."""

    min_moon_sep_deg: float
    """Minimum angular separation from the Moon's centre, degrees."""

    max_wind_speed_ms: float
    """Wind speed at dome height, m/s.  Dome closes or tracking degrades above this."""

    max_sun_alt_deg: float
    """
    Upper bound on the Sun's altitude, degrees.
    """

    # ------------------------------------------------------------------ #
    # Horizon / pointing limits                                            #
    # ------------------------------------------------------------------ #
    horizon_alt_deg: float
    """
    Scalar lower altitude limit in degrees.  Used when horizon_mask is None.
    Represents the worst-case physical or regulatory floor across all azimuths.
    """

    # ------------------------------------------------------------------ #
    # Mount geometry                                                       #
    # ------------------------------------------------------------------ #
    equatorial_limit: EquatorialLimit | None = field(default=None, compare=False, hash=False)
    """
    Optional operational pointing-limit envelope for an equatorial mount,
    evaluated in the (Hour Angle, Declination) plane.  When set, is_observable()
    additionally requires the pointing to sit inside this envelope.
    """

    # ------------------------------------------------------------------ #
    # Core evaluation                                                      #
    # ------------------------------------------------------------------ #

    def is_observable(
        self,
        az: float,
        alt: float,
        airmass: float,
        moon_sep: float,
        wind_speed: float,
        sun_alt: float,
        ha: float | None = None,
        dec: float | None = None,
    ) -> bool:
        """
        Return True iff all constraints pass for the given pointing and
        ambient conditions.

        Parameters
        ----------
        az, alt     : pointing in degrees (topocentric)
        airmass     : pre-computed sec(z) for this pointing and time
        moon_sep    : angular separation from Moon centre, degrees
        wind_speed  : dome wind speed, m/s
        sun_alt     : Sun altitude, degrees
        ha          : Hour Angle in radians.  Required for the equatorial_limit
                      check; when omitted that check is skipped.
        dec         : Declination in degrees.  Required for the equatorial_limit
                      check; when omitted that check is skipped.
        """
        if airmass > self.max_airmass:
            return False
        if moon_sep < self.min_moon_sep_deg:
            return False
        if wind_speed > self.max_wind_speed_ms:
            return False
        if sun_alt > self.max_sun_alt_deg:
            return False
        if self.horizon_mask is not None:
            if not self.horizon_mask(az, alt):
                return False
        elif alt < self.horizon_alt_deg:
            return False
        if self.equatorial_limit is not None and ha is not None and dec is not None:
            if not bool(self.equatorial_limit.satisfies(ha, dec)):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Per-filter overrides                                                 #
    # ------------------------------------------------------------------ #

    def filter_overrides(self) -> dict[str, ConstraintSet]:
        """
        Return a dict of filter_name -> ConstraintSet for filters that require
        constraints *tighter* than the base set (e.g. u-band needs larger moon
        separation, narrowband filters need dark sky).

        Override in subclasses or monkeypatch per-profile.  The scheduler
        calls constraints_for_filter(filt) rather than this method directly.

        Default: no per-filter overrides.
        """
        return {}

    def constraints_for_filter(self, filter_name: str) -> ConstraintSet:
        """
        Return the effective ConstraintSet for a specific filter.
        Falls back to self if no override is registered.
        """
        return self.filter_overrides().get(filter_name, self)

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"ConstraintSet("
            f"max_X={self.max_airmass}, "
            f"moon≥{self.min_moon_sep_deg}°, "
            f"wind≤{self.max_wind_speed_ms}m/s, "
            f"sun≤{self.max_sun_alt_deg}°, "
            f"alt≥{self.horizon_alt_deg}°)"
        )



@dataclass(frozen=True)
class EquatorialLimit:
    """
    Operational pointing-limit envelope for an equatorial mount, expressed in
    the (Hour Angle, Declination) plane.
    """

    bound_points: np.ndarray = field(compare=False, hash=False)
    """Symmetric (2N-1, 2) array of (HA_hours, max_Dec_deg), sorted by HA."""

    max_ha_hours: float = 5.25
    """Absolute Hour Angle limit in decimal hours.  Pointings past +/- this fail."""

    dec_floor: float = -89.0
    """Lowest schedulable Declination in degrees (observatory tracking floor)."""

    def __post_init__(self) -> None:
        interp = interp1d(
            self.bound_points[:, 0],
            self.bound_points[:, 1],
            kind="linear",
            bounds_error=False,
            fill_value=-999.0,  # out-of-range HA -> impossible Dec ceiling
        )
        object.__setattr__(self, "_interp", interp)

    @classmethod
    def from_ha_dec_table(
        cls,
        table: np.ndarray,
        max_ha_hours: float = 5.25,
        dec_floor: float = -89.0,
    ) -> EquatorialLimit:
        """
        Build from a one-sided table of (HA_hours, max_Dec_deg) points with
        HA >= 0.  The first row must be the HA=0 point; remaining rows are
        mirrored to negative Hour Angles to form the full symmetric envelope.
        """
        table = np.asarray(table, dtype=float)
        neg = np.copy(table[1:])
        neg[:, 0] = -neg[:, 0]
        bound_points = np.vstack((neg[::-1], table))
        return cls(bound_points=bound_points, max_ha_hours=max_ha_hours, dec_floor=dec_floor)

    def satisfies(self, ha: float, dec: float):
        """
        Return True iff the pointing sits inside the operational envelope.

        Args
        ----
        ha : Hour Angle in radians (scalar or array-like)
        dec: Declination in degrees (scalar or array-like)
        """
        ha = np.asarray(ha)
        dec = np.asarray(dec)

        ha_hours = ha * (12.0 / np.pi)

        within_absolute_limits = (
            (ha_hours >= -self.max_ha_hours)
            & (ha_hours <= self.max_ha_hours)
            & (dec >= self.dec_floor)
        )
        under_curve = dec <= self._interp(ha_hours)
        return within_absolute_limits & under_curve

    def __repr__(self) -> str:
        return (
            f"EquatorialLimit("
            f"|HA|<={self.max_ha_hours}h, "
            f"dec>={self.dec_floor}deg)"
        )

