from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlewModel:
    """Linear fit to slew. All values in degrees and seconds.

    Parameters
    ----------
    rate : float
        Slew time per degree of on-sky distance (sec/deg)
    intercept : float
        Intercept of the linear fit (sec).
    """

    rate: float
    intercept: float

    def slew_time(self, distance: float | np.ndarray) -> float | np.ndarray:
        """Slew time between exposures: intercept + rate * d.

        Parameters
        ----------
        distance : float or np.ndarray
            On-sky slew distance in degrees.

        Returns
        -------
        float or np.ndarray
            Slew time in seconds, same shape as distance.
        """
        return self.intercept + self.rate * distance


@dataclass(frozen=True)
class TelescopeParameters:
    """
    Hardware capabilities and fixed timing constants for one telescope +
    instrument combination.

    Timing convention (per visit):
        total_time = exposure + max(readout + overhead [+ filter_change], slew)

    All times in seconds, all angles in degrees.
    """

    # ------------------------------------------------------------------ #
    # Slew                                                                 #
    # ------------------------------------------------------------------ #
    slew: SlewModel | None

    # ------------------------------------------------------------------ #
    # Instrument timing                                                    #
    # ------------------------------------------------------------------ #
    readout_time: float         # seconds — detector readout after each exposure
    overhead_time: float        # seconds — other fixed per-exposure overhead
    # shutter_overhead: float     # seconds — open + close per exposure

    # ------------------------------------------------------------------ #
    # Field of view                                                        #
    # ------------------------------------------------------------------ #
    fov_deg: float              # effective diameter of the focal plane, degrees

    # ------------------------------------------------------------------ #
    # Visit duration bounds                                                #
    # ------------------------------------------------------------------ #
    min_visit_duration: float   # seconds — shortest scientifically useful exposure
    max_visit_duration: float   # seconds — scheduler ceiling (not a hardware limit)

    # ------------------------------------------------------------------ #
    # Filter complement                                                    #
    # ------------------------------------------------------------------ #
    filters: tuple[str, ...]    # ordered tuple of available filter names

    # ------------------------------------------------------------------ #
    # Filter change                                                        #
    # ------------------------------------------------------------------ #
    filter_change_time: float = 0.0  # seconds

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def fov_sq_deg(self) -> float:
        """Solid angle of the focal plane in square degrees (circular aperture)."""
        return math.pi * (self.fov_deg / 2.0) ** 2

    # ------------------------------------------------------------------ #
    # Per-visit overhead                                                   #
    # ------------------------------------------------------------------ #

    def visit_overhead(self, filter_change: bool | np.ndarray = False) -> float | np.ndarray:
        """Fixed per-visit overhead: readout + overhead [+ filter_change]. Excludes slew time.

        Parameters
        ----------
        filter_change : bool or np.ndarray
            Whether the filter changes between the two exposures.

        Returns
        -------
        float or np.ndarray
            Overhead in seconds, same shape as filter_change.
        """
        return self.readout_time + self.overhead_time + self.filter_change_time * filter_change

    def dead_time(
        self, distance: float | np.ndarray, filter_change: bool | np.ndarray = False
    ) -> float | np.ndarray:
        """Time between finished consecutive exposure finish to start time: max(visit_overhead, slew_time(d)).

        Parameters
        ----------
        distance : float or np.ndarray
            On-sky slew distance in degrees.
        filter_change : bool or np.ndarray
            Whether the filter changes between the two exposures.

        Returns
        -------
        float or np.ndarray
            Dead time in seconds, broadcast over distance and filter_change.
        """
        if self.slew is None:
            raise ValueError("No SlewModel fitted for this telescope; dead_time is undefined.")
        return np.maximum(self.visit_overhead(filter_change), self.slew.slew_time(distance))

    def __repr__(self) -> str:
        return (
            f"TelescopeParameters("
            f"fov={self.fov_deg}°, "
            f"filters={self.filters}, "
            f"readout={self.readout_time}s)"
        )
