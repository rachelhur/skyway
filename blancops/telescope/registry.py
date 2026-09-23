"""
Telescope registry.

Telescopes can be resolved by calling get_telescope(key).

Adding a new telescope
----------------------
1. Create telescope/<name>.py (see blanco.py as template).
2. Import the profile(s) below and add them to _ALL_PROFILES.
3. Add the key string to configs/enums.py :: TelescopeKey.

"""
from __future__ import annotations

from blancops.telescope.base import TelescopeProfile
from blancops.telescope.blanco import BLANCO
from blancops.telescope.rubin import RUBIN, RUBIN_SIM

# ------------------------------------------------------------------ #
# Registry construction                                                #
# ------------------------------------------------------------------ #

_ALL_PROFILES: list[TelescopeProfile] = [
    RUBIN,
    RUBIN_SIM,
    BLANCO,
]

REGISTRY: dict[str, TelescopeProfile] = {p.key: p for p in _ALL_PROFILES}

# Sanity-check: duplicate keys would silently shadow each other
assert len(REGISTRY) == len(_ALL_PROFILES), (
    "Duplicate telescope keys detected in _ALL_PROFILES. "
    "Each profile must have a unique .key attribute."
)

# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def get_telescope(key: str) -> TelescopeProfile:
    """
    Resolve a telescope key string to its TelescopeProfile.

    Parameters
    ----------
    key : str
        A TelescopeKey enum value or its string equivalent,
        e.g. "rubin", "rubin_sim", "blanco".

    Returns
    -------
    TelescopeProfile

    Raises
    ------
    KeyError
        If the key is not registered.  The error message lists all
        valid keys so callers get an actionable failure.
    """
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown telescope key {key!r}. "
            f"Registered keys: {list_telescopes()}"
        )
    return REGISTRY[key]


def list_telescopes() -> list[str]:
    """Return all registered telescope keys in sorted order."""
    return sorted(REGISTRY.keys())


def get_telescopes_by_site(site_name: str) -> list[TelescopeProfile]:
    """
    Return all profiles whose site.name contains `site_name` (case-insensitive).
    Useful when multiple instruments share a mountaintop (e.g. Cerro Tololo).
    """
    needle = site_name.lower()
    return [p for p in REGISTRY.values() if needle in p.site.name.lower()]
