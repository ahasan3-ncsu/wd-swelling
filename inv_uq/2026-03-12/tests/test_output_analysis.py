"""Tests for Slice 2 descriptive output diagnostics."""

import unittest

import numpy as np

from src.constants import INPUT_NAMES, MODELING_OUTPUT_NAMES
from src.data import load_dataset
from src.output_analysis import (
    audit_output_transformations,
    high_correlation_pairs,
    input_output_sensitivity,
    output_correlations,
    summarize_outputs,
)


class OutputAnalysisTests(unittest.TestCase):
    def test_summaries_cover_all_modeling_outputs(self) -> None:
        for dataset in ("RERTR5", "RERTR12"):
            with self.subTest(dataset=dataset):
                summary = summarize_outputs(dataset, load_dataset(dataset))
                self.assertEqual(tuple(summary.index), MODELING_OUTPUT_NAMES[dataset])
                self.assertTrue((summary["count"] == 3200).all())
                self.assertTrue(np.isfinite(summary.to_numpy(dtype=float)).all())

    def test_correlations_cover_all_outputs_and_match_dataframe(self) -> None:
        for dataset in ("RERTR5", "RERTR12"):
            frame = load_dataset(dataset)
            outputs = frame.loc[:, list(MODELING_OUTPUT_NAMES[dataset])]
            for method, matrix in output_correlations(dataset, frame).items():
                with self.subTest(dataset=dataset, method=method):
                    self.assertEqual(tuple(matrix.columns), MODELING_OUTPUT_NAMES[dataset])
                    self.assertEqual(tuple(matrix.index), MODELING_OUTPUT_NAMES[dataset])
                    np.testing.assert_allclose(matrix, outputs.corr(method=method))

    def test_high_correlation_pairs_are_unique(self) -> None:
        correlation = output_correlations("RERTR5", load_dataset("RERTR5"))[
            "spearman"
        ]
        pairs = high_correlation_pairs(correlation)
        unordered = [frozenset(row) for row in pairs[["output_1", "output_2"]].values]
        self.assertEqual(len(unordered), len(set(unordered)))
        self.assertTrue((pairs["absolute_correlation"] >= 0.85).all())

    def test_transformation_diagnostics_cover_all_outputs(self) -> None:
        for dataset in ("RERTR5", "RERTR12"):
            with self.subTest(dataset=dataset):
                frame = load_dataset(dataset)
                audit = audit_output_transformations(dataset, frame)
                self.assertEqual(tuple(audit.index), MODELING_OUTPUT_NAMES[dataset])
                self.assertNotIn("recommendation", audit.columns)
                for name, row in audit.iterrows():
                    values = frame[name]
                    scale = values[values > 0].median()
                    self.assertAlmostEqual(row["positive_median_scale"], scale)
                    self.assertAlmostEqual(row["raw_skewness"], values.skew())
                    self.assertAlmostEqual(
                        row["log1p_scaled_skewness"], np.log1p(values / scale).skew()
                    )

    def test_sensitivity_screen_has_expected_shape_and_finite_values(self) -> None:
        for dataset in ("RERTR5", "RERTR12"):
            with self.subTest(dataset=dataset):
                sensitivity = input_output_sensitivity(
                    dataset, load_dataset(dataset)
                )
                self.assertEqual(tuple(sensitivity.index), INPUT_NAMES)
                self.assertEqual(
                    tuple(sensitivity.columns), MODELING_OUTPUT_NAMES[dataset]
                )
                self.assertTrue(np.isfinite(sensitivity.to_numpy()).all())



if __name__ == "__main__":
    unittest.main()
