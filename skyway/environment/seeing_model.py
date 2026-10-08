"""Seeing FWHM strategy models used by the environments.

`base._get_fwhm` delegates to one of these. The constant model is the
forward-sim default and the extension point for a future forecast model;
the predictive model wraps the rolling-history Seeing predictor for the
live and historic envs. `fwhm()` returns plain arcsec (the env feature
scale) and defaults to the i-band zenith value, so the feature is
pointing-independent unless an explicit band/el is requested.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from skyway.math import units
from skyway.data_quality.seeing import Seeing, convert_seeing


class SeeingModel(ABC):
    """Strategy returning the delivered FWHM (arcsec) at a pointing."""

    @abstractmethod
    def fwhm(self, timestamp: float, band: str, el: float) -> float:
        """Delivered FWHM in arcsec for band (letter) and el (radians)."""

    def add(self, *, date, seeing, band, el) -> None:
        """Ingest a measurement. No-op for models without history."""
        return None


class ConstantSeeingModel(SeeingModel):
    """Constant zenith seeing, returned at i-band zenith by default.

    Holds a fixed delivered zenith seeing in `ref_band`. `fwhm()` converts
    it to the requested band/el via convert_seeing, defaulting to i-band at
    zenith.
    """

    def __init__(self, zenith_seeing: float, ref_band: str = "r",
                 ref_el: float = None):
        self._seeing = float(zenith_seeing)
        self._ref_band = ref_band
        self._ref_el = np.pi / 2 if ref_el is None else float(ref_el)

    def fwhm(self, timestamp: float, band: str = "i", el: float = np.pi / 2) -> float:
        projected = convert_seeing(
            self._seeing * units.arcsec,
            to_band=band, to_el=el,
            from_band=self._ref_band, from_el=self._ref_el,
        )
        return float(projected) / units.arcsec


class PredictiveSeeingModel(SeeingModel):
    """Rolling-history predictor wrapping Seeing.

    `add()` ingests real measurements (plain arcsec); `fwhm()` returns the
    causal prediction (plain arcsec) including the instrument component,
    at i-band zenith by default.
    """

    def __init__(self, seeing_cfg):
        self._seeing = Seeing(
            window=seeing_cfg.window,
            retention_window=seeing_cfg.retention_window,
            from_instrument=seeing_cfg.from_instrument * units.arcsec,
            to_instrument=seeing_cfg.to_instrument * units.arcsec,
        )

    def add(self, *, date, seeing, band, el) -> None:
        self._seeing.add(
            date=date,
            seeing=np.asarray(seeing, dtype=float) * units.arcsec,
            band=band,
            el=el,
        )

    def prune(self, now=None) -> None:
        self._seeing.prune(now=now)

    def fwhm(self, timestamp: float, band: str = "i", el: float = np.pi / 2) -> float:
        predicted = self._seeing.predict(band=band, el=el, now=timestamp)
        return float(predicted) / units.arcsec

    def replace(self, raw):
        self._seeing.replace(raw)
