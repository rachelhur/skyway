from dataclasses import dataclass, field, replace

import numpy as np

from blancops.configs.enums import AcceptanceRule
from blancops.telescope.base import TelescopeProfile
from blancops.telescope.registry import get_telescope


@dataclass(frozen=True)
class SurveyProfile:
    key: str
    filters: tuple[str, ...]        # survey filters in action-space order
    telescope_key: str              # telescope the survey runs on; supplies filter wavelengths
    seeing_ref_filter: str          # band seeing is projected to when a pointing has no filter
    sun_el_limit: float
    valid_teff_threshold: float
    min_teff_per_band: dict = field(default_factory=dict)

    def __post_init__(self):
        """Refuse filters the survey's telescope does not install or gives no wavelength for.

        i.e., narrowband filters for DECam doesn't have a wavelength in the Blanco telescope profile

        Raises
        ------
        ValueError
            Naming the survey, the telescope and the filter.
        """
        params = self.telescope.parameters
        for f in self.filters:
            if f not in params.filters or f not in params.filter_wavelengths:
                raise ValueError(f"Survey '{self.key}' uses filter '{f}', which telescope '{self.telescope_key}' "
                                 f"does not install with a wavelength.")
        if self.seeing_ref_filter not in self.filters:
            raise ValueError(f"Survey '{self.key}' seeing reference band '{self.seeing_ref_filter}' is not one of "
                             f"its filters {self.filters}.")

    def check_telescope(self, telescope: TelescopeProfile) -> None:
        """Refuse a telescope at a different site, or without the survey's filters at the same wavelengths.

        Variants of the survey's own telescope (relaxed constraints, changed timing parameters) pass.

        Parameters
        ----------
        telescope : TelescopeProfile
            Telescope an environment or dataset was given.

        Raises
        ------
        ValueError
            Naming the survey, the survey's telescope and the given telescope.
        """
        own = self.telescope
        same_site = (telescope.site.lat, telescope.site.lon) == (own.site.lat, own.site.lon)
        given = telescope.parameters.filter_wavelengths
        same_filters = all(given.get(f) == own.parameters.filter_wavelengths[f] for f in self.filters)
        same_norm = telescope.parameters.filter_wave_norm == own.parameters.filter_wave_norm
        if not (same_site and same_filters and same_norm):
            raise ValueError(f"Survey '{self.key}' runs on telescope '{own.key}', but was given telescope "
                             f"'{telescope.key}' with a different site, filter wavelengths or wavelength normalizer.")

    def check_lookups(self, lookups) -> None:
        """Refuse lookups built for a different survey.

        Parameters
        ----------
        lookups : LookupTables
            Lookups an environment or dataset was given.

        Raises
        ------
        ValueError
            Naming both surveys.
        """
        if lookups.survey != self:
            raise ValueError(f"Lookups were built for survey '{lookups.survey.key}', but survey '{self.key}' was given.")

    def acceptance_thresholds(self, acceptance: AcceptanceRule | str) -> np.ndarray:
        """Minimum accepted teff per filter index under an acceptance rule.

        Parameters
        ----------
        acceptance : AcceptanceRule or str
            UNIFORM (the single `valid_teff_threshold` for every band) or
            DES_PER_BAND (`min_teff_per_band`).

        Returns
        -------
        np.ndarray
            [n_filters] thresholds, ordered by `filter2idx`.
        """
        rule = AcceptanceRule(acceptance)
        if rule is AcceptanceRule.UNIFORM:
            return np.full(self.num_filters, self.valid_teff_threshold, dtype=float)
        return np.array([self.min_teff_per_band[self.idx2filter[i]] for i in range(self.num_filters)], dtype=float)

    @property
    def telescope(self) -> TelescopeProfile:
        return get_telescope(self.telescope_key)

    @property
    def num_filters(self):
        return len(self.filters)

    @property
    def filter2wave(self):
        wavelengths = self.telescope.parameters.filter_wavelengths
        return {f: wavelengths[f] for f in self.filters}

    @property
    def filter2idx(self):
        return {f: i for i, f in enumerate(self.filters)}

    @property
    def idx2filter(self):
        return {i: f for i, f in enumerate(self.filters)}

    @property
    def idx2wave(self):
        wavelengths = self.filter2wave
        return {i: wavelengths[f] for i, f in enumerate(self.filters)}

    @property
    def seeing_ref_wave(self):
        return self.filter2wave[self.seeing_ref_filter]

# ----------------------------------------------------------- #
# -------------------- Survey Profiles ---------------------- #
# ----------------------------------------------------------- #

DES = SurveyProfile(
    key="des",
    filters=("g", "r", "i", "z", "Y"),
    telescope_key="blanco",
    seeing_ref_filter="r",  # obztak's seeing reference band
    sun_el_limit=-10.5,
    valid_teff_threshold=0.3,
    min_teff_per_band={"g": 0.2, "r": 0.3, "i": 0.3, "z": 0.3, "Y": 0.2},   # Per-band minimum accepted teff (Morganson et al. 2018, Table 4).
)
