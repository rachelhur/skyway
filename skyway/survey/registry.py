"""
Survey registry.

Surveys can be resolved by calling ``get_survey(key)``.

Adding a new survey
----------------------
1. Create an entry for a telescope at the bottom of profiles.py
2. Import the profile(s) below and add them to _ALL_PROFILES.
3. Add the key string to configs/enums.py :: SurveyKey. # XXX not implemented

"""
from __future__ import annotations

from skyway.survey.profiles import SurveyProfile, DES

# ------------------------------------------------------------------ #
# Registry construction                                                #
# ------------------------------------------------------------------ #

_ALL_PROFILES: list[SurveyProfile] = [
    DES
]

REGISTRY: dict[str, SurveyProfile] = {p.key: p for p in _ALL_PROFILES}

# Check for duplicates -- XXX should be done in pytest
assert len(REGISTRY) == len(_ALL_PROFILES), (
    "Duplicate survey keys detected in _ALL_PROFILES. "
    "Each profile must have a unique .key attribute."
)

# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def get_survey(key: str) -> SurveyProfile:
    """
    Resolve a telescope key string to its SurveyProfile.

    Parameters
    ----------
    key : str
        A SurveyKey enum value or its string equivalent,
        e.g. "rubin", "rubin_sim", "blanco".

    Returns
    -------
    SurveyProfile

    Raises
    ------
    KeyError
        If the key is not registered.  The error message lists all
        valid keys so callers get an actionable failure.
    """
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown survey key {key!r}. "
            f"Registered keys: {list_surveys()}"
        )
    return REGISTRY[key]


def list_surveys() -> list[str]:
    """Return all registered survey keys in sorted order."""
    return sorted(REGISTRY.keys())

