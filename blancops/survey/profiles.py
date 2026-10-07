from dataclasses import dataclass, field, replace

import numpy as np

from blancops.configs.enums import AcceptanceRule


@dataclass(frozen=True)
class SurveyProfile:
    key: str
    filter2wave: dict
    sun_el_limit: float
    valid_teff_threshold: float
    min_teff_per_band: dict = field(default_factory=dict)

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
            [n_filters] thresholds, ordered by FILTER2IDX.
        """
        rule = AcceptanceRule(acceptance)
        if rule is AcceptanceRule.UNIFORM:
            return np.full(self.num_filters, self.valid_teff_threshold, dtype=float)
        return np.array([self.min_teff_per_band[self.idx2filter[i]] for i in range(self.num_filters)], dtype=float)

    @property
    def filters(self):
        return list(self.filter2wave.keys())

    @property
    def num_filters(self):
        return len(self.filters)

    @property
    def filter2idx(self):
        return {f: i for i, f in enumerate(self.filters)}

    @property
    def idx2filter(self):
        return {i: f for i, f in enumerate(self.filters)}

    @property
    def idx2wave(self):
        return {i: self.filter2wave[f] for i, f in enumerate(self.filters)}

# ----------------------------------------------------------- #
# -------------------- Survey Profiles ---------------------- #
# ----------------------------------------------------------- #

DES = SurveyProfile(
    key="des",
    filter2wave={'g': 480,'r': 640,'i': 780,'z': 920,'Y': 990},
    sun_el_limit=-10.5,
    valid_teff_threshold=0.3,
    min_teff_per_band={"g": 0.2, "r": 0.3, "i": 0.3, "z": 0.3, "Y": 0.2},   # Per-band minimum accepted teff (Morganson et al. 2018, Table 4).
)
