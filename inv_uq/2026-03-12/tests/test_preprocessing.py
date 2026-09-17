"""Tests for reference-scale nondimensionalization."""

import unittest

import pandas as pd

from src.constants import INPUT_NAMES, INPUT_REFERENCE_SCALES
from src.data import load_dataset
from src.preprocessing import (
    ReferenceNondimensionalizer,
    validate_nondimensionalization,
)


class PreprocessingTests(unittest.TestCase):
    def test_real_inputs_remain_varying_and_order_one(self) -> None:
        for dataset in ("RERTR5", "RERTR12"):
            with self.subTest(dataset=dataset):
                raw = load_dataset(dataset).loc[:, INPUT_NAMES]
                scaler = ReferenceNondimensionalizer()
                transformed = scaler.transform(raw)
                audit = validate_nondimensionalization(raw, transformed)

                self.assertFalse(audit["transformed_is_constant"].any())
                self.assertTrue(audit["order_one_range"].all())

    def test_round_trip_recovers_physical_inputs(self) -> None:
        raw = load_dataset("RERTR12").loc[:9, INPUT_NAMES]
        scaler = ReferenceNondimensionalizer()
        restored = scaler.inverse_transform(scaler.transform(raw))

        pd.testing.assert_frame_equal(
            restored, raw.astype(float), rtol=1e-14, atol=0.0
        )

    def test_scaling_is_parameter_specific(self) -> None:
        raw = pd.DataFrame(
            {
                name: [scale, 2.0 * scale]
                for name, scale in INPUT_REFERENCE_SCALES.items()
            }
        )
        transformed = ReferenceNondimensionalizer().transform(raw)

        self.assertTrue((transformed.iloc[0] == 1.0).all())
        self.assertTrue((transformed.iloc[1] == 2.0).all())

    def test_zero_reference_scale_is_rejected(self) -> None:
        scales = dict(INPUT_REFERENCE_SCALES)
        scales["vResol"] = 0.0

        with self.assertRaisesRegex(ValueError, "finite and nonzero"):
            ReferenceNondimensionalizer(scales)


if __name__ == "__main__":
    unittest.main()
