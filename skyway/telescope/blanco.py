"""
Víctor M. Blanco Telescope / CTIO
----------------------------------
Site      : Cerro Tololo Inter-American Observatory, Chile
Elevation : 2207 m
Instrument: Dark Energy Camera (DECam) — 3.0 sq deg (2.2 deg) FOV, 570 Mpx (62 CCDs)

This module defines two profiles:

  BLANCO       — standard DECam broadband survey mode (DES-era cadence)

References
----------
1. Flaugher et al. 2015, AJ 150 150  (DECam instrument paper)
2. DES Collaboration 2005, astro-ph/0510346  (DES science requirements)
3. CTIO / NOIRLab instrument pages: https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/ #XXX LAST UPDATE 2020/2/6
4. DECam Exposure Time Calculator: https://www.ctio.noirlab.edu/~decam/etc/
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from skyway.math import units
from skyway.telescope.base import TelescopeProfile
from skyway.telescope.constraints import ConstraintSet, EquatorialLimit
from skyway.telescope.parameters import SlewModel, TelescopeParameters
from skyway.telescope.site import ObservingSite

# ------------------------------------------------------------------ #
# Site                                                                 #
# ------------------------------------------------------------------ #

_SITE = ObservingSite(
    name="Cerro Tololo Inter-American Observatory",
    lat=-30.169661,
    lon=-70.806525,
    alt=2206.8,
    timezone="America/Santiago",
)

# ------------------------------------------------------------------ #
# Slew model                                                           #
# ------------------------------------------------------------------ #

# Fit parameters to DECam exposure data (decam-exposures-20251211.fits)
_SLEW = SlewModel(rate=2.19, intercept=23.51)

# ------------------------------------------------------------------ #
# Instrument parameters — DECam broadband                             #
# ------------------------------------------------------------------ #

_PARAMS = TelescopeParameters(
    slew=_SLEW,
    readout_time=20.6, # Ref. 3
    overhead_time=8.0, # Ref. 3: "hexapod movement, filter change, and others"; occurs every exposure
    # filter_change_time defaults to 0 (filter changes overlap with readout)
    # shutter_overhead=1.0, # Ref. 3: "approximately 1 sec"
    fov_deg=2.2,
    # inter_ccd_gap=(3.0, 2.3), # (long, short) gap between CCDs in mm

    # DES used 90-second exposures as the standard visit; shorter visits
    # are used in some transient / ToO programs.  Max is a scheduler ceiling.
    min_visit_duration=2.0,
    max_visit_duration=30.0 * 60, # 30 min?

    # DECam broadband filter complement as installed for DES + community programs.
    # Effective wavelength centres (nm): g≈475, r≈638, i≈775, z≈919, Y≈988
    # VR is a wide Vr filter used by programmes like DESGW (gravitational waves).
    filters=("g", "r", "i", "z", "Y", "VR", "N964"),
    # Wavelengths (nm) used by features and seeing, from obztak seeing.py:
    # https://github.com/kadrlica/obztak/blob/c28fab23b09bcff1cf46746eae4ec7e40aeb7f7a/obztak/seeing.py#L22
    filter_wavelengths={"g": 480, "r": 640, "i": 780, "z": 920, "Y": 990},
)

# ------------------------------------------------------------------ #
# Observability constraints — DECam broadband                         #
# ------------------------------------------------------------------ #

class _BlancoConstraints(ConstraintSet):
    """
    Blanco / DECam constraint set.

    Per-filter overrides:
      - Y-band : slightly relaxed moon sep (NIR less affected by scatter)
      - N964   : narrowband, but sky background still matters → base constraint
      - VR     : wide filter used in time-domain programs, often at grey time
                 → relaxed moon sep to 20°
    """
    def filter_overrides(self) -> dict[str, ConstraintSet]:
        from dataclasses import replace as dc_replace
        return {
            "Y":  dc_replace(self, min_moon_sep_deg=20.0),
            "VR": dc_replace(self, min_moon_sep_deg=20.0),
        }


# Official NOIRLab Horizon Limits for the Blanco 4m telescope.
# One-sided table of (Hour Angle in decimal hours, max Declination in degrees);
# EquatorialLimit mirrors it about HA=0 to form the full envelope.
# https://noirlab.edu/science/images/horizonlimits
_BLANCO_OPERATION_RANGE = np.array([
    [0.00,  37.0],
    [1.10,  35.0],  # 01:06:00
    [2.06,  30.0],  # 02:03:36
    [2.64,  25.0],  # 02:38:24
    [3.08,  20.0],  # 03:04:48
    [3.43,  15.0],  # 03:25:48
    [3.72,  10.0],  # 03:43:12
    [3.98,   5.0],  # 03:58:48
    [4.21,   0.0],  # 04:12:36
    [4.42,  -5.0],  # 04:25:12
    [4.61, -10.0],  # 04:36:36
    [4.79, -15.0],  # 04:47:24
    [4.96, -20.0],  # 04:57:36
    [5.12, -25.0],  # 05:07:12
    [5.25, -30.0]   # 05:15:00
])

# Dec floor of -89 deg reflects observatory tracking warnings near the pole.
_CONSTRAINTS = _BlancoConstraints(
    max_airmass=3.0,
    min_moon_sep_deg=30.0,
    max_wind_speed_ms=12.0,
    max_sun_alt_deg=-10.0,
    horizon_alt_deg=15.0,
    equatorial_limit=EquatorialLimit.from_ha_dec_table(
        _BLANCO_OPERATION_RANGE, max_ha_hours=5.25, dec_floor=-89.0
    ),
)

# ------------------------------------------------------------------ #
# SISPI observing script                                             #
# ------------------------------------------------------------------ #

# SISPI, the Blanco control system, loads observing scripts as a JSON list with one exposure
# entry per element. Layout from obztak's SISPI_DICT (kadrlica/obztak, obztak/field.py),
# plus the optional `proposer` key.
_EMPTY_SISPI_DICT = OrderedDict([
    ("object",  None),
    ("seqnum",  None), # 1-indexed
    ("seqtot",  1),
    ("seqid",   ""),
    ("expTime", 90),
    ("RA",      None),
    ("dec",     None),
    ("filter",  None),
    ("count",   1),
    ("expType", "object"),
    ("program", None),
    ("wait",    "False"),
    ("propid",  None),
    ("comment", ""),
])


def write_sispi(schedule_df: pd.DataFrame, name: str, save_dir: Path, lookups, *,
                propid: str, proposer: str, program: str,
                filter_override_val: str | None = None) -> Path:
    """Write a schedule as a SISPI JSON script `<name>_sispi.json`, one exposure entry per row.

    Parameters
    ----------
    schedule_df : pd.DataFrame
        Time-ordered schedule with columns `timestamp` (unix s), `field_id` and `filter`.
    name : str
        Output filename stem.
    save_dir : Path
        Output directory.
    lookups : LookupTables
        Field coordinates (radians), names and per-(field, filter) exposure times.
    propid : str
        Proposal id written to every entry.
    proposer : str
        Proposer written to every entry.
    program : str
        Program name written to every entry.
    filter_override_val : str or None, optional
        Filter written to every entry instead of the scheduled filter.

    Returns
    -------
    Path
        Path of the written file.
    """
    if not propid:
        raise ValueError("A propid is required to write a SISPI file.")
    if not program:
        raise ValueError("A program is required to write a SISPI file.")
    timestamps = schedule_df['timestamp'].to_numpy(dtype=float)
    if (np.diff(timestamps) < 0).any():
        raise ValueError("SISPI schedule timestamps must be in time order.")
    dts = pd.to_datetime(timestamps, utc=True, unit='s')
    outpath = Path(save_dir) / f"{name}_sispi.json"

    field_ids = schedule_df['field_id'].to_numpy(dtype=int)
    if filter_override_val is not None:
        filters = [filter_override_val] * len(field_ids)
    else:
        filters = schedule_df['filter'].to_list()
    filter_idxs = np.array([lookups.survey.filter2idx[f] for f in filters], dtype=int)

    exptimes = lookups.fidfilt_exptime[field_ids, filter_idxs].astype(int)
    if (exptimes <= 0).any():
        bad = sorted({(int(f), filt) for f, filt, t in zip(field_ids, filters, exptimes) if t <= 0})
        raise ValueError(f"Scheduled (field_id, filter) pairs have no positive exposure time: {bad}")

    names = lookups.fields['field'].to_numpy()[field_ids]
    ras = lookups.fields['ra'].to_numpy()[field_ids] / units.deg
    decs = lookups.fields['dec'].to_numpy()[field_ids] / units.deg

    sispi_list = []
    for i in range(len(field_ids)):
        obs = _EMPTY_SISPI_DICT.copy()
        obs.update({
            "object": str(names[i]),
            "seqnum": 1,
            "seqtot": 1,
            "seqid": f"datetime: {dts[i].isoformat(timespec='seconds')}",
            "expTime": int(exptimes[i]),
            "RA": round(float(ras[i]), 5),
            "dec": round(float(decs[i]), 5),
            "filter": filters[i],
            "program": program,
            "propid": propid,
            "proposer": proposer,
        })
        sispi_list.append(obs)

    with open(outpath, 'w') as f:
        json.dump(sispi_list, f, indent=4)
    return outpath


# ------------------------------------------------------------------ #
# Primary profile — DECam broadband                                   #
# ------------------------------------------------------------------ #

BLANCO = TelescopeProfile(
    key="blanco",
    display_name="Víctor M. Blanco Telescope (DECam)",
    site=_SITE,
    parameters=_PARAMS,
    constraints=_CONSTRAINTS,
    observing_script_writer=write_sispi,
)
