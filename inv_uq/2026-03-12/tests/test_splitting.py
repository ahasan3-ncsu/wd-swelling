"""Tests for deterministic, exhaustive Slice 2 partitions."""

import unittest

import pandas as pd

from src.data import load_dataset
from src.splitting import (
    SPLIT_SEEDS,
    make_regression_splits,
    validate_splits,
)


class SplittingTests(unittest.TestCase):
    def test_split_is_deterministic_and_exactly_sized(self) -> None:
        for dataset in ("RERTR5", "RERTR12"):
            with self.subTest(dataset=dataset):
                frame = load_dataset(dataset)
                first = make_regression_splits(
                    frame["Fuel_Swelling"], seed=SPLIT_SEEDS[dataset]
                )
                second = make_regression_splits(
                    frame["Fuel_Swelling"], seed=SPLIT_SEEDS[dataset]
                )
                pd.testing.assert_frame_equal(first, second)
                self.assertEqual(
                    first["split"].value_counts().to_dict(),
                    {"train": 2240, "validation": 480, "test": 480},
                )

    def test_split_is_exhaustive_and_disjoint(self) -> None:
        frame = load_dataset("RERTR5")
        splits = make_regression_splits(
            frame["Fuel_Swelling"], seed=SPLIT_SEEDS["RERTR5"]
        )
        audit = validate_splits(splits, expected_indices=frame.index)
        self.assertEqual(audit.rows, len(frame))
        self.assertEqual(audit.duplicate_indices, 0)
        self.assertEqual(audit.missing_indices, 0)

    def test_each_rank_stratum_is_represented_in_every_split(self) -> None:
        frame = load_dataset("RERTR12")
        splits = make_regression_splits(
            frame["Fuel_Swelling"], seed=SPLIT_SEEDS["RERTR12"]
        )
        counts = splits.groupby(["stratum", "split"]).size().unstack()
        self.assertTrue((counts > 0).all().all())

    def test_different_seeds_change_membership(self) -> None:
        values = load_dataset("RERTR5")["Fuel_Swelling"]
        first = make_regression_splits(values, seed=1)
        second = make_regression_splits(values, seed=2)
        self.assertFalse(first["split"].equals(second["split"]))


if __name__ == "__main__":
    unittest.main()
