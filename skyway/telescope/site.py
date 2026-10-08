from __future__ import annotations

import numpy as np
from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time


@dataclass(frozen=True)
class ObservingSite:
    """
    Geographic and temporal identity of a telescope site.

    All coordinates follow the standard astronomical convention:
      lat  — degrees, positive north of equator
      lon  — degrees, positive east of Greenwich
      alt  — metres above the WGS-84 ellipsoid
    """

    name: str
    lat: float    # degrees
    lon: float    # degrees
    alt: float    # metres
    timezone: str  # IANA timezone string, e.g. "America/Santiago"

    # ------------------------------------------------------------------ #
    # Astropy interop                                                       #
    # ------------------------------------------------------------------ #

    def earth_location(self) -> EarthLocation:
        return EarthLocation(
            lat=self.lat * u.deg,
            lon=self.lon * u.deg,
            height=self.alt * u.m,
        )

    # ------------------------------------------------------------------ #
    # Night boundary computation                                           #
    # ------------------------------------------------------------------ #

    def night_window(
        self,
        date: str,
        twilight_deg: float = -18.0,
        n_samples: int = 1440,
    ) -> tuple[Time, Time]:
        """
        Return (evening_twilight, morning_twilight) for the night that begins
        on `date` (UTC date string, e.g. '2025-03-15').

        Uses astronomical twilight by default (sun altitude = -18°).
        Pass twilight_deg=-12 for nautical or -6 for civil twilight.

        Returns astropy Time objects in UTC.
        """
        loc = self.earth_location()
        noon = Time(f"{date} 12:00:00", scale="utc")

        times = noon + np.linspace(0.0, 24.0, n_samples) * u.hour
        frame = AltAz(obstime=times, location=loc)
        sun_alt = get_body("sun", times).transform_to(frame).alt.deg

        # Find sign changes: negative → eve crossing, positive → morn crossing
        signs = np.sign(sun_alt - twilight_deg)
        crossings = np.where(np.diff(signs))[0]

        eve_idx  = crossings[signs[crossings] > 0][0]   # descending through limit
        morn_idx = crossings[signs[crossings] < 0][0]   # ascending through limit

        return times[eve_idx], times[morn_idx]

    def lst(self, time: Time) -> float:
        """Local sidereal time in hours at the given UTC Time."""
        return time.sidereal_time("apparent", longitude=self.lon * u.deg).hour

    def __repr__(self) -> str:
        return (
            f"ObservingSite({self.name!r}, "
            f"lat={self.lat:.4f}°, lon={self.lon:.4f}°, alt={self.alt:.0f}m)"
        )
