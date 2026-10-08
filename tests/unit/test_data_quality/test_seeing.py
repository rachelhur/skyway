import unittest
from importlib import util
from pathlib import Path
import pandas as pd
from skyway.math import units
from skyway.data_quality.seeing import Seeing, convert_seeing
from skyway.ephemerides.time_utils import standardize_time


class TestSeeing(unittest.TestCase):
    def test_add_populates_raw_and_converted_history(self):
        seeing = Seeing(window="1h")
        seeing.add(
            date="2026-06-05T00:00:00",
            seeing=1.3 * units.arcsec,
            band="g",
            el=60 * units.deg,
        )

        self.assertEqual(list(seeing.raw.columns), ["date", "seeing", "band", "el"])
        self.assertEqual(list(seeing.data.columns), ["date", "seeing", "band", "el"])
        self.assertEqual(len(seeing.raw), 1)
        self.assertEqual(len(seeing.data), 1)
        self.assertEqual(seeing.raw.iloc[0]["seeing"], 1.3 * units.arcsec)
        self.assertEqual(seeing.raw.iloc[0]["band"], "g")
        self.assertEqual(seeing.data.iloc[0]["band"], "i")
        self.assertEqual(seeing.data.iloc[0]["el"], 90 * units.deg)
        self.assertAlmostEqual(
            seeing.data.iloc[0]["seeing"],
            convert_seeing(
                1.3 * units.arcsec,
                to_band="i",
                to_el=90 * units.deg,
                from_band="g",
                from_el=60 * units.deg,
                from_instrument=0.5 * units.arcsec,
                to_instrument=0.0,
            ),
        )

    def test_add_handles_multiple_array_inputs(self):
        seeing = Seeing(window="1h")
        seeing.add(
            date=[
                pd.Timestamp("2026-06-05T00:00:00"),
                pd.Timestamp("2026-06-05T00:05:00"),
            ],
            seeing=[1.1 * units.arcsec, 1.2 * units.arcsec],
            band=["r", "i"],
            el=[55 * units.deg, 65 * units.deg],
        )

        self.assertEqual(len(seeing.raw), 2)
        self.assertEqual(len(seeing.data), 2)
        self.assertEqual(list(seeing.raw["band"]), ["r", "i"])

    def test_prune_drops_old_history(self):
        seeing = Seeing(window="1h")
        seeing.add(
            date=["2026-06-05T00:00:00", "2026-06-05T01:30:00"],
            seeing=[1.0 * units.arcsec, 1.1 * units.arcsec],
            band="i",
            el=90 * units.deg,
        )

        seeing.prune(now="2026-06-05T02:00:00", retention_window="30m")

        self.assertEqual(len(seeing.raw), 1)
        self.assertEqual(len(seeing.data), 1)
        self.assertEqual(
            seeing.raw.iloc[0]["date"], standardize_time("2026-06-05T01:30:00")
        )

    def test_predict_uses_nominal_fallback_without_history(self):
        seeing = Seeing()

        predicted = seeing.predict(
            band="i", el=90 * units.deg, now="2026-06-05T00:00:00"
        )

        self.assertAlmostEqual(predicted, 0.9 * units.arcsec, places=6)


if __name__ == "__main__":
    unittest.main()
