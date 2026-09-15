"""Strict loading and validation for the two simulation datasets."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATASET_PATHS, INPUT_NAMES, OUTPUT_NAMES


@dataclass(frozen=True)
class DataAudit:
    dataset: str
    rows: int
    columns: int
    missing_values: int
    nonfinite_values: int
    duplicate_rows: int
    duplicate_input_rows: int
    constant_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a table-friendly representation of the audit."""
        return asdict(self)


def expected_columns(dataset: str) -> tuple[str, ...]:
    """Return the exact schema expected for a named dataset."""
    try:
        return INPUT_NAMES + OUTPUT_NAMES[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}") from exc


def load_dataset(dataset: str, path: str | Path | None = None) -> pd.DataFrame:
    """Load one dataset and enforce its numeric schema.

    ``utf-8-sig`` intentionally strips the byte-order mark present in the
    RERTR12 header.
    """
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset {dataset!r}")

    source = Path(path) if path is not None else DATASET_PATHS[dataset]
    frame = pd.read_csv(source, encoding="utf-8-sig")
    expected = expected_columns(dataset)

    missing = [name for name in expected if name not in frame.columns]
    unexpected = [name for name in frame.columns if name not in expected]
    if missing or unexpected:
        raise ValueError(
            f"Invalid {dataset} schema; missing={missing}, unexpected={unexpected}"
        )

    frame = frame.loc[:, expected].copy()
    for column in expected:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def audit_dataset(dataset: str, frame: pd.DataFrame | None = None) -> DataAudit:
    """Summarize structural problems without silently altering the data."""
    data = load_dataset(dataset) if frame is None else frame
    values = data.to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    constant = tuple(
        column for column in data.columns if data[column].nunique(dropna=False) <= 1
    )
    return DataAudit(
        dataset=dataset,
        rows=len(data),
        columns=len(data.columns),
        missing_values=int(data.isna().sum().sum()),
        nonfinite_values=int((~finite_mask & ~np.isnan(values)).sum()),
        duplicate_rows=int(data.duplicated().sum()),
        duplicate_input_rows=int(data.duplicated(subset=list(INPUT_NAMES)).sum()),
        constant_columns=constant,
    )


def validate_dataset(dataset: str, frame: pd.DataFrame | None = None) -> DataAudit:
    """Raise on invalid values or duplicate simulation inputs."""
    audit = audit_dataset(dataset, frame)
    problems: list[str] = []
    if audit.missing_values:
        problems.append(f"{audit.missing_values} missing values")
    if audit.nonfinite_values:
        problems.append(f"{audit.nonfinite_values} nonfinite values")
    if audit.duplicate_input_rows:
        problems.append(f"{audit.duplicate_input_rows} duplicate input rows")
    if problems:
        raise ValueError(f"{dataset} validation failed: {', '.join(problems)}")
    return audit

