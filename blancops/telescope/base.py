from __future__ import annotations

from dataclasses import dataclass, replace

from blancops.telescope.constraints import ConstraintSet
from blancops.telescope.parameters import TelescopeParameters
from blancops.telescope.site import ObservingSite


@dataclass(frozen=True)
class TelescopeProfile:
    """
    Immutable bundle of everything site- and hardware-specific for one
    telescope + instrument combination.

    This is the single object the rest of the codebase interacts with.
    Nothing outside `telescope/` should import ObservingSite, TelescopeParameters,
    or ConstraintSet directly — they access them through this profile.

    Usage
    -----
    from blancops.telescope import get_telescope

    t_profile = get_telescope("blanco")
    t_slew  = t_profile.parameters.slew.slew_time(distance=15.0)
    ok      = t_profile.constraints.is_observable(az, alt, X, moon_sep, wind, sun_alt)
    loc     = t_profile.site.earth_location()
    """

    key: str
    """
    Machine-readable identifier.  Must match a TelescopeKey enum value and
    a key in telescope.registry.REGISTRY.  Lowercase, underscore-separated.
    """

    display_name: str
    """Human-readable name shown in logs and reports."""

    site: ObservingSite
    parameters: TelescopeParameters
    constraints: ConstraintSet

    # ------------------------------------------------------------------ #
    # Convenience constructors                                             #
    # ------------------------------------------------------------------ #

    def with_relaxed_constraints(self, key_suffix: str = "relaxed", **overrides) -> TelescopeProfile:
        """
        Return a copy with selected ConstraintSet fields overridden.

        Designed for building simulation variants without duplicating the full
        t_profile definition.  The new key is ``{self.key}_{key_suffix}``.

        Example
        -------
        rubin_sim = RUBIN.with_relaxed_constraints(
            key_suffix="sim",
            max_airmass=2.0,
            min_moon_sep_deg=20.0,
            max_wind_speed_ms=99.0,
        )
        """
        return replace(
            self,
            key=f"{self.key}_{key_suffix}",
            constraints=replace(self.constraints, **overrides),
        )

    def with_parameters(self, key_suffix: str = "custom", **overrides) -> TelescopeProfile:
        """
        Return a copy with selected TelescopeParameters fields overridden.

        Useful for modelling instrument upgrades (e.g. faster readout after a
        CCD swap) without forking the whole profile.
        """
        return replace(
            self,
            key=f"{self.key}_{key_suffix}",
            parameters=replace(self.parameters, **overrides),
        )

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"TelescopeProfile(key={self.key!r}, name={self.display_name!r})"
