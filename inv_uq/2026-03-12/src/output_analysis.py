"""Descriptive output diagnostics for Slice 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .constants import INPUT_NAMES, MODELING_OUTPUT_NAMES


@dataclass(frozen=True)
class TransformationAudit:
    """Descriptive statistics before and after a scale-aware log1p transform."""

    output: str
    nonnegative: bool
    zero_fraction: float
    raw_skewness: float
    positive_median_scale: float
    log1p_scaled_skewness: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _modeling_outputs(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    try:
        names = MODELING_OUTPUT_NAMES[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}") from exc
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing {dataset} modeling outputs: {missing}")
    return frame.loc[:, names].astype(float)


def summarize_outputs(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return physical-unit distribution and range diagnostics."""
    outputs = _modeling_outputs(dataset, frame)
    records = []
    for name in outputs:
        values = outputs[name]
        positive = values[values > 0.0]
        records.append(
            {
                "output": name,
                "count": int(values.count()),
                "min": float(values.min()),
                "q05": float(values.quantile(0.05)),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "q75": float(values.quantile(0.75)),
                "q95": float(values.quantile(0.95)),
                "max": float(values.max()),
                "std": float(values.std()),
                "skewness": float(values.skew()),
                "zero_fraction": float(values.eq(0.0).mean()),
                "positive_dynamic_range": (
                    float(positive.max() / positive.min())
                    if not positive.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records).set_index("output")


def output_correlations(
    dataset: str, frame: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Return Pearson linear and Spearman rank correlation matrices."""
    outputs = _modeling_outputs(dataset, frame)
    return {
        "pearson": outputs.corr(method="pearson"),
        "spearman": outputs.corr(method="spearman"),
    }


def high_correlation_pairs(
    correlation: pd.DataFrame, threshold: float = 0.85
) -> pd.DataFrame:
    """List unique output pairs whose absolute correlation meets a threshold."""
    records = []
    for first, second in combinations(correlation.columns, 2):
        value = float(correlation.loc[first, second])
        if abs(value) >= threshold:
            records.append(
                {
                    "output_1": first,
                    "output_2": second,
                    "correlation": value,
                    "absolute_correlation": abs(value),
                }
            )
    columns = [
        "output_1",
        "output_2",
        "correlation",
        "absolute_correlation",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        "absolute_correlation", ascending=False, ignore_index=True
    )


def audit_output_transformations(
    dataset: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compare raw and log1p skewness without recommending a transformation.

    The diagnostic transform is ``log1p(y / s)``, where ``s`` is the median
    positive value in the supplied frame. The scale makes it dimensionless.
    These are descriptive statistics only. Any later modeling transform must
    fit its data-derived scale using training data only.
    """
    outputs = _modeling_outputs(dataset, frame)
    audits = []
    for name in outputs:
        values = outputs[name]
        nonnegative = bool(values.ge(0.0).all())
        positive = values[values > 0.0]
        scale = float(positive.median()) if not positive.empty else np.nan
        raw_skew = float(values.skew())
        if nonnegative and np.isfinite(scale) and scale > 0.0:
            transformed_skew = float(np.log1p(values / scale).skew())
        else:
            transformed_skew = np.nan
        audits.append(
            TransformationAudit(
                output=name,
                nonnegative=nonnegative,
                zero_fraction=float(values.eq(0.0).mean()),
                raw_skewness=raw_skew,
                positive_median_scale=scale,
                log1p_scaled_skewness=transformed_skew,
            ).to_dict()
        )
    return pd.DataFrame(audits).set_index("output")


def input_output_sensitivity(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Compute a screening-level Spearman input-output sensitivity matrix.

    Rank correlation is invariant to positive reference scaling, so this is
    numerically safe for raw dimensions near 1e-18 and is explicitly only an
    initial, marginal sensitivity screen.
    """
    outputs = _modeling_outputs(dataset, frame)
    missing = [name for name in INPUT_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    combined = pd.concat(
        [frame.loc[:, INPUT_NAMES].astype(float), outputs], axis=1
    )
    correlation = combined.corr(method="spearman")
    return correlation.loc[list(INPUT_NAMES), list(outputs.columns)]


def ranked_sensitivities(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy absolute ranking for every output."""
    records = []
    for output in sensitivity.columns:
        ordered = sensitivity[output].abs().sort_values(ascending=False)
        for rank, parameter in enumerate(ordered.index, start=1):
            records.append(
                {
                    "output": output,
                    "rank": rank,
                    "parameter": parameter,
                    "spearman": float(sensitivity.loc[parameter, output]),
                    "absolute_spearman": float(ordered.loc[parameter]),
                }
            )
    return pd.DataFrame(records).set_index(["output", "rank"])
