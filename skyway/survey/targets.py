from __future__ import annotations

from typing import Literal

import numpy as np
import healpy as hp
import pandas as pd
from astropy import units as au
from astropy.coordinates import SkyCoord

from skyway.survey.footprint import galactic_plane_mask
from skyway.survey.visibility import visible_fields


def build_candidate_fields(
    tile_radec_rad: np.ndarray,                  # (n, 2) [ra, dec] radians
    nside: int,
    g_lat_lim: float,
    window_utc: tuple[str, str],
    *,
    airmass_limit: float = None,
    require: Literal["any", "all"] = "any",
) -> pd.DataFrame:
    """
    Return candidate fields surviving galactic-plane exclusion AND window
    visibility. Output schema matches LookupTables.fields: columns
    name, ra, dec (radians), plus frac_observable. Index is contiguous 0..N-1.
    """
    ra = tile_radec_rad[:, 0]
    dec = tile_radec_rad[:, 1]

    exclusion = galactic_plane_mask(nside, g_lat_lim)     # (npix,)
    pix = hp.ang2pix(nside, np.pi / 2 - dec, ra)
    in_exclusion = exclusion[pix]

    vis_mask, frac = visible_fields(
        tile_radec_rad, window_utc, airmass_limit=airmass_limit, require=require
    )

    keep = (~in_exclusion) & vis_mask
    n_keep = int(keep.sum())
    return pd.DataFrame(
        {
            "name": [f"field_{i}" for i in range(n_keep)],
            "ra": ra[keep],
            "dec": dec[keep],
            "frac_observable": frac[keep],
        }
    ).reset_index(drop=True)


def parse_count_arg(s: str) -> int | dict[str, int]:
    """Parse a --count CLI value: a plain int, or per-filter "f=n,..." pairs.

    "3" -> 3; "i=2,z=3,Y=1" -> {"i": 2, "z": 3, "Y": 1}. The dict form is
    consumed by to_lookup_fields_df to set a different count per filter.
    """
    if "=" not in s:
        return int(s)
    out = {}
    for pair in s.split(","):
        filt, _, n = pair.partition("=")
        out[filt.strip()] = int(n)
    return out


def to_lookup_fields_df(
    fields: pd.DataFrame,
    propid: str,
    filt: str,
    count: int | dict[str, int] = 1,
    exptime: float = 90.0,
    priority: int = 0,
    *,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> pd.DataFrame:
    """Assemble a per-field DataFrame for LookupTables.build_lookups_from_fields.

    RA/Dec are read in degrees, wrapped to [0, 360), and converted to radians
    (the units the lookup builder requires). filt is one filter per character
    ("z" -> z only, "zg" -> z and g); each field is emitted once per filter,
    sharing the same exptime and tagged with propid. count is shared across
    filters when an int, or set per filter when a dict keyed by filter
    character (e.g. {"z": 3, "g": 1}).
    """
    ra_deg = fields[ra_col].to_numpy(dtype=float)
    ra_deg = np.where(ra_deg < 0, ra_deg + 360.0, ra_deg)
    dec_deg = fields[dec_col].to_numpy(dtype=float)
    filters = list(filt)
    n = len(ra_deg)
    if isinstance(count, dict):
        count = np.repeat([count[f] for f in filters], n)
    return pd.DataFrame(
        {
            "RA": np.tile(np.radians(ra_deg), len(filters)),
            "DEC": np.tile(np.radians(dec_deg), len(filters)),
            "filter": np.repeat(filters, n),
            "count": count,
            "exptime": exptime,
            "propid": propid,
            "priority": priority
        }
    )

def select_covering_tiles(
    target_ra: np.ndarray,
    target_dec: np.ndarray,
    tile_ra: np.ndarray,
    tile_dec: np.ndarray,
    fov: float = 1.1,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the tiles whose FOV covers a set of target positions. Wrapper around astropy's
    `SkyCoord.match_to_catalog_sky`.

    Each target is matched to its nearest tile center on the sphere. A target
    is considered covered when that nearest center lies within the tile field
    of view, i.e. when the angular separation satisfies

        sep <= fov

    The circular (radius) test is conservative: with fov = 1.1 deg it stays
    inside DECam's real focal-plane footprint, so a "covered" result implies
    on-sky coverage. The returned tile set is the union of nearest tiles taken
    over all covered targets.

    Parameters
    ----------
    target_ra, target_dec : np.ndarray
        Target right ascension and declination in degrees, shape (n_targets,).
    tile_ra, tile_dec : np.ndarray
        Tiling-center right ascension and declination in degrees, shape
        (n_tiles,).
    fov : float, optional
        Tile field-of-view radius in degrees (default 1.1). A target is covered
        when within this angular distance of a tile center.
    return_indices : bool, optional
        Default False, returns mask array over tiles. If True, returns tile indices (default True).

    Returns
    -------
    selected_tile_idx : np.ndarray
        Sorted unique integer indices into the tile arrays for tiles that cover
        at least one target.
    uncovered_target_mask : np.ndarray
        Boolean array, shape (n_targets,), True where the target's nearest tile
        is farther than fov (no tile covers it).
    """
    target_ra = np.asarray(target_ra, dtype=float)
    target_dec = np.asarray(target_dec, dtype=float)
    tile_ra = np.asarray(tile_ra, dtype=float)
    tile_dec = np.asarray(tile_dec, dtype=float)

    if target_ra.size == 0 or tile_ra.size == 0:
        return (
            np.empty(0, dtype=int),
            np.ones(target_ra.size, dtype=bool),
        )

    targets = SkyCoord(ra=target_ra * au.deg, dec=target_dec * au.deg)
    tiles = SkyCoord(ra=tile_ra * au.deg, dec=tile_dec * au.deg)

    # Nearest tile center per target (great-circle separation).
    nearest_tile_idx, sep, _ = targets.match_to_catalog_sky(tiles)

    covered = sep <= fov * au.deg
    selected_tile_idx = np.unique(nearest_tile_idx[covered])
    if not return_indices:
        out = np.zeros(tile_ra.size, dtype=bool)
        out[selected_tile_idx] = True
    uncovered_target_mask = ~covered

    return selected_tile_idx, uncovered_target_mask


def select_in_path(vertices, ra, dec, wrap=180., radius=0.0):
    """
    Checks if a set of (ra, dec) points (in degrees) fall within a given polygon.

    A positive radius (degrees) grows the polygon outward by that margin. matplotlib
    only grows a counter-clockwise polygon, so the vertices are reoriented to CCW
    first, making the radius sign consistent regardless of input winding.
    """
    import matplotlib
    # Copy coordinates to avoid modifying the original arrays
    ra, dec = np.copy(ra), np.copy(dec)

    # Wrap Right Ascension coordinates
    ra -= 360 * (ra > wrap)

    # Reorient to CCW (positive signed area) so a positive radius grows outward
    verts = np.asarray(vertices, dtype=float)
    vx, vy = verts[:, 0], verts[:, 1]
    signed_area = 0.5 * np.sum(vx * np.roll(vy, -1) - np.roll(vx, -1) * vy)
    if signed_area < 0:
        verts = verts[::-1]

    # Create the matplotlib path from the provided vertices
    path = matplotlib.path.Path(verts)

    # Format the test points into an (M, 2) array
    points = np.vstack([ra, dec]).T

    # Check which points are inside the path
    sel = path.contains_points(points, radius=radius)

    return sel
