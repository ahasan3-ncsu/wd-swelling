"""Tests for strict dataset loading and validation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.constants import INPUT_NAMES
from src.data import expected_columns, load_dataset, validate_dataset


class DataTests(unittest.TestCase):
    def test_real_datasets_pass_validation(self) -> None:
        for dataset in ("RERTR5", "RERTR12"):
            with self.subTest(dataset=dataset):
                frame = load_dataset(dataset)
                audit = validate_dataset(dataset, frame)

                self.assertEqual(tuple(frame.columns), expected_columns(dataset))
                self.assertEqual(audit.rows, 3200)
                self.assertEqual(audit.missing_values, 0)
                self.assertEqual(audit.nonfinite_values, 0)
                self.assertEqual(audit.duplicate_input_rows, 0)

    def test_rertr12_byte_order_mark_is_removed(self) -> None:
        frame = load_dataset("RERTR12")
        self.assertEqual(frame.columns[0], "dGrainHBS")

    def test_schema_validation_rejects_missing_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            pd.DataFrame({name: [1.0] for name in INPUT_NAMES}).to_csv(
                source, index=False
            )
            with self.assertRaisesRegex(ValueError, "Invalid RERTR5 schema"):
                load_dataset("RERTR5", source)

    def test_validation_rejects_duplicate_inputs(self) -> None:
        frame = load_dataset("RERTR5").iloc[:2].copy()
        frame.loc[1, list(INPUT_NAMES)] = frame.loc[
            0, list(INPUT_NAMES)
        ].to_numpy()

        with self.assertRaisesRegex(ValueError, "duplicate input rows"):
            validate_dataset("RERTR5", frame)


if __name__ == "__main__":
    unittest.main()
