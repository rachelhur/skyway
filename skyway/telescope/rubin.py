"""
!--------------------------------------------------------------------------!
! CLAUDE-GENERATED AND NOT VERIFIED. DO NOT TRUST!
!--------------------------------------------------------------------------!

Vera C. Rubin Observatory / LSST
---------------------------------
Site      : Cerro Pachón, Chile
Elevation : 2 647 m
Instrument: LSST Camera (LSSTCam) — 9.6 sq deg FOV, 3.2 Gpx

References
----------
- Ivezić et al. 2019, ApJ 873 111  (LSST system overview)
- LSST System Requirements Document, LPM-17
- Rubin Observatory System Specifications, LSE-29
- OpSim / rubin_sim documentation (https://rubin-sim.lsst.io)
"""
from __future__ import annotations

from dataclasses import replace

from skyway.telescope.base import TelescopeProfile
from skyway.telescope.constraints import ConstraintSet
from skyway.telescope.parameters import TelescopeParameters
from skyway.telescope.site import ObservingSite

# ------------------------------------------------------------------ #
# Site                                                                 #
# ------------------------------------------------------------------ #

_SITE = ObservingSite(
    name="Cerro Pachón",
    lat=-30.2446,   # degrees north  (30°14'40.8" S)
    lon=-70.7494,   # degrees east   (70°44'57.8" W)
    alt=2647.0,     # metres
    timezone="America/Santiago",
)

# ------------------------------------------------------------------ #
# Slew model                                                           #
# ------------------------------------------------------------------ #
# Source: LSST System Requirements Document LPM-17 §3.2
#   - Az  : 7 deg/s peak,  7 deg/s² accel  (cable-wrap limited to ±270°)
#   - Alt : 3.5 deg/s peak, 3.5 deg/s² accel
# TODO: fit SlewModel(rate, intercept) to Rubin visit data.

_SLEW = None

# ------------------------------------------------------------------ #
# Instrument parameters                                                #
# ------------------------------------------------------------------ #

_PARAMS = TelescopeParameters(
    slew=_SLEW,

    # LSSTCam readout: ~2.3 s for 15-second snaps (pipelined)
    readout_time=2.3,

    # Filter change: camera has a 6-slot carousel; mechanical exchange ≈120 s
    filter_change_time=120.0,

    # Shutter open+close: ≈1 s
    overhead_time=1.0,

    # Effective focal-plane diameter.
    # LSSTCam: 641 mm diameter FP on a 8.36 m (effective) primary → 3.5° FOV
    fov_deg=3.5,

    # LSST baseline cadence: two 15-second snaps per standard visit (30 s total).
    # min is one snap; max is a single long exposure in specialist modes.
    min_visit_duration=15.0,
    max_visit_duration=120.0,

    # LSST filter complement (ugrizy — Sloan-like + y-band NIR)
    filters=("u", "g", "r", "i", "z", "y"),
)

# ------------------------------------------------------------------ #
# Observability constraints                                            #
# ------------------------------------------------------------------ #

class _RubinConstraints(ConstraintSet):
    """
    Rubin-specific constraint set with per-filter moon-separation overrides.

    u-band is most sensitive to scattered moonlight, so we enforce a tighter
    separation limit.  Other filters use the base 30°.
    """
    def filter_overrides(self) -> dict[str, ConstraintSet]:
        from dataclasses import replace as dc_replace
        return {
            "u": dc_replace(self, min_moon_sep_deg=40.0),
        }


_CONSTRAINTS = _RubinConstraints(
    # LSST SRD requires observations at X ≤ 1.5 for the main survey;
    # deep-drilling fields relax this to 2.0 in later-survey years.
    max_airmass=1.5,

    # Moon separation: 30° baseline; u-band override → 40° (see above)
    min_moon_sep_deg=30.0,

    # Dome wind limit per LSST System Specs; image quality degrades above ~10 m/s
    max_wind_speed_ms=15.0,

    # Sun must be below -12 (nautical) before LSST main-survey visits begin.
    max_sun_alt_deg=-12.0,

    # Physical horizon mask is complex (mountain silhouette + telescope structure);
    # use a scalar floor of 20° as a conservative proxy.
    # To plug in the real mask: _CONSTRAINTS = replace(_CONSTRAINTS, horizon_mask=my_fn)
    horizon_alt_deg=20.0,
)

# ------------------------------------------------------------------ #
# Primary profile                                                      #
# ------------------------------------------------------------------ #

RUBIN = TelescopeProfile(
    key="rubin",
    display_name="Vera C. Rubin Observatory",
    site=_SITE,
    parameters=_PARAMS,
    constraints=_CONSTRAINTS,
)

# ------------------------------------------------------------------ #
# Simulation variant                                                   #
# ------------------------------------------------------------------ #
# Used by SimulationEnv and rubin_sim / opsim-based training pipelines.
# Wind is not modelled in opsim, horizon is simplified, and airmass /
# moon cuts are relaxed slightly to match the OpSim scheduler's behaviour.

RUBIN_SIM = replace(
    RUBIN,
    key="rubin_sim",
    display_name="Rubin Observatory (rubin_sim / opsim)",
    constraints=replace(
        _CONSTRAINTS,
        max_airmass=2.0,
        min_moon_sep_deg=20.0,
        max_wind_speed_ms=99.0,   # wind is not simulated; effectively disabled
        horizon_alt_deg=15.0,
    ),
)
