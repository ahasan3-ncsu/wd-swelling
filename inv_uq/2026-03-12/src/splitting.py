"""Fixed, validated regression splits for Slice 2 and later surrogates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_SEEDS = {"RERTR5": 202603125, "RERTR12": 202603122}


@dataclass(frozen=True)
class SplitAudit:
    rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    duplicate_indices: int
    missing_indices: int


def _allocation(size: int, fractions: tuple[float, ...]) -> np.ndarray:
    raw = np.asarray(fractions, dtype=float) * size
    counts = np.floor(raw).astype(int)
    remainder = size - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def make_regression_splits(
    stratify_values: pd.Series,
    *,
    seed: int,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    bins: int = 10,
) -> pd.DataFrame:
    """Create deterministic target-stratified train/validation/test indices.

    Stable ranks define equal-frequency strata, avoiding fragile numeric bin
    edges. Allocation is performed within each stratum and adjusted globally
    to exactly match the requested split sizes.
    """
    values = pd.Series(stratify_values, copy=True)
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Stratification values must be nonempty and finite")
    if bins < 2 or bins > len(values):
        raise ValueError("bins must be between 2 and the number of rows")
    fraction_array = np.asarray(fractions, dtype=float)
    if (
        len(fraction_array) != len(SPLIT_NAMES)
        or np.any(fraction_array <= 0.0)
        or not np.isclose(fraction_array.sum(), 1.0)
    ):
        raise ValueError("fractions must contain three positive values summing to 1")

    row_indices = values.index.to_numpy()
    order = np.argsort(values.to_numpy(dtype=float), kind="stable")
    strata = np.empty(len(values), dtype=int)
    strata[order] = np.minimum(
        np.arange(len(values), dtype=int) * bins // len(values), bins - 1
    )

    rng = np.random.default_rng(seed)
    assignments = np.empty(len(values), dtype=object)
    for stratum in range(bins):
        positions = np.flatnonzero(strata == stratum)
        positions = rng.permutation(positions)
        counts = _allocation(len(positions), fractions)
        start = 0
        for name, count in zip(SPLIT_NAMES, counts, strict=True):
            assignments[positions[start : start + count]] = name
            start += int(count)

    # Correct small rounding differences to exact global requested counts.
    desired = _allocation(len(values), fractions)
    current = np.array([(assignments == name).sum() for name in SPLIT_NAMES])
    while not np.array_equal(current, desired):
        receiver = int(np.flatnonzero(current < desired)[0])
        donor = int(np.flatnonzero(current > desired)[0])
        donor_positions = np.flatnonzero(assignments == SPLIT_NAMES[donor])
        position = int(rng.choice(donor_positions))
        assignments[position] = SPLIT_NAMES[receiver]
        current[donor] -= 1
        current[receiver] += 1

    result = pd.DataFrame(
        {
            "row_index": row_indices,
            "split": assignments,
            "stratum": strata,
        }
    ).sort_values("row_index", kind="stable", ignore_index=True)
    validate_splits(result, expected_indices=values.index)
    return result


def validate_splits(
    splits: pd.DataFrame, *, expected_indices: pd.Index
) -> SplitAudit:
    """Raise unless split indices form an exhaustive, disjoint partition."""
    required = {"row_index", "split"}
    missing_columns = sorted(required - set(splits.columns))
    if missing_columns:
        raise ValueError(f"Missing split columns: {missing_columns}")
    unknown = sorted(set(splits["split"]) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unknown split labels: {unknown}")

    duplicate_indices = int(splits["row_index"].duplicated().sum())
    expected = set(expected_indices.tolist())
    actual = set(splits["row_index"].tolist())
    missing_indices = len(expected - actual)
    extra_indices = len(actual - expected)
    if duplicate_indices or missing_indices or extra_indices or len(splits) != len(expected):
        raise ValueError(
            "Split indices are not an exhaustive disjoint partition; "
            f"duplicates={duplicate_indices}, missing={missing_indices}, "
            f"extra={extra_indices}"
        )
    counts = splits["split"].value_counts()
    if any(int(counts.get(name, 0)) == 0 for name in SPLIT_NAMES):
        raise ValueError("Every split must contain at least one row")
    return SplitAudit(
        rows=len(splits),
        train_rows=int(counts["train"]),
        validation_rows=int(counts["validation"]),
        test_rows=int(counts["test"]),
        duplicate_indices=duplicate_indices,
        missing_indices=missing_indices,
    )
