"""Safe input nondimensionalization performed before statistical scaling."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .constants import INPUT_NAMES, INPUT_REFERENCE_SCALES, ORDER_ONE_RANGE


@dataclass(frozen=True)
class RangeAudit:
    parameter: str
    raw_min: float
    raw_max: float
    raw_range: float
    reference_scale: float
    transformed_min: float
    transformed_max: float
    transformed_range: float
    raw_is_constant: bool
    transformed_is_constant: bool
    order_one_range: bool


class ReferenceNondimensionalizer:
    """Divide inputs by fixed parameter-specific, nonzero scales."""

    def __init__(self, scales: Mapping[str, float] = INPUT_REFERENCE_SCALES):
        self.scales = {name: float(scales[name]) for name in INPUT_NAMES}
        invalid = {
            name: scale
            for name, scale in self.scales.items()
            if not np.isfinite(scale) or scale == 0.0
        }
        if invalid:
            raise ValueError(f"Reference scales must be finite and nonzero: {invalid}")

    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Return dimensionless inputs without modifying the caller's frame."""
        missing = [name for name in INPUT_NAMES if name not in inputs.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        transformed = inputs.loc[:, INPUT_NAMES].astype(float).copy()
        transformed /= pd.Series(self.scales)
        return transformed

    def inverse_transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Map dimensionless inputs back to physical units."""
        missing = [name for name in INPUT_NAMES if name not in inputs.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        restored = inputs.loc[:, INPUT_NAMES].astype(float).copy()
        restored *= pd.Series(self.scales)
        return restored


def audit_transformed_ranges(
    raw_inputs: pd.DataFrame,
    transformed_inputs: pd.DataFrame,
    scales: Mapping[str, float] = INPUT_REFERENCE_SCALES,
    acceptable_range: tuple[float, float] = ORDER_ONE_RANGE,
) -> pd.DataFrame:
    """Verify variation is preserved and transformed ranges are order-one.

    Exact equality is used for the constant check. Tolerance-based helpers such
    as ``numpy.isclose`` are unsafe for raw dimensions near 1e-18.
    """
    lower, upper = acceptable_range
    records: list[RangeAudit] = []
    for name in INPUT_NAMES:
        raw = raw_inputs[name].to_numpy(dtype=float)
        transformed = transformed_inputs[name].to_numpy(dtype=float)
        raw_min, raw_max = float(np.min(raw)), float(np.max(raw))
        tx_min, tx_max = float(np.min(transformed)), float(np.max(transformed))
        raw_range = raw_max - raw_min
        tx_range = tx_max - tx_min
        records.append(
            RangeAudit(
                parameter=name,
                raw_min=raw_min,
                raw_max=raw_max,
                raw_range=raw_range,
                reference_scale=float(scales[name]),
                transformed_min=tx_min,
                transformed_max=tx_max,
                transformed_range=tx_range,
                raw_is_constant=raw_range == 0.0,
                transformed_is_constant=tx_range == 0.0,
                order_one_range=(raw_range == 0.0) or (lower <= tx_range <= upper),
            )
        )
    return pd.DataFrame(record.__dict__ for record in records).set_index("parameter")


def validate_nondimensionalization(
    raw_inputs: pd.DataFrame,
    transformed_inputs: pd.DataFrame,
) -> pd.DataFrame:
    """Raise if scaling collapses variation or produces a poor numeric range."""
    audit = audit_transformed_ranges(raw_inputs, transformed_inputs)
    collapsed = audit.index[
        (~audit["raw_is_constant"]) & audit["transformed_is_constant"]
    ].tolist()
    not_order_one = audit.index[~audit["order_one_range"]].tolist()
    if collapsed or not_order_one:
        raise ValueError(
            "Nondimensionalization validation failed; "
            f"collapsed={collapsed}, non_order_one_ranges={not_order_one}"
        )
    return audit

