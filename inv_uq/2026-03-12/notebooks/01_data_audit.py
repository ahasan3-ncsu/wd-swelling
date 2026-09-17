"""Interactive Slice 1 audit for both simulation datasets."""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    # Resolve the local package independently of the launch directory.
    _project_root = str(Path(__file__).resolve().parents[1])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from src.constants import INPUT_NAMES, MODELING_OUTPUT_NAMES
    from src.data import audit_dataset, load_dataset, validate_dataset
    from src.preprocessing import (
        ReferenceNondimensionalizer,
        validate_nondimensionalization,
    )

    return (
        INPUT_NAMES,
        MODELING_OUTPUT_NAMES,
        ReferenceNondimensionalizer,
        audit_dataset,
        load_dataset,
        mo,
        pd,
        plt,
        validate_dataset,
        validate_nondimensionalization,
    )


@app.cell
def _(load_dataset, validate_dataset):
    datasets = {
        name: load_dataset(name) for name in ("RERTR5", "RERTR12")
    }
    audits = {
        name: validate_dataset(name, frame)
        for name, frame in datasets.items()
    }
    return audits, datasets


@app.cell
def _(audits, mo, pd):
    audit_table = pd.DataFrame(
        audit.to_dict() for audit in audits.values()
    ).set_index("dataset")
    mo.vstack(
        [
            mo.md("# Slice 1: data and nondimensionalization audit"),
            mo.md(
                "Both named datasets are checked without modifying their raw values."
            ),
            mo.ui.table(audit_table, selection=None),
        ]
    )
    return (audit_table,)


@app.cell
def _(
    INPUT_NAMES,
    ReferenceNondimensionalizer,
    datasets,
    validate_nondimensionalization,
):
    nondimensionalizer = ReferenceNondimensionalizer()
    transformed_inputs = {}
    range_audits = {}
    for name, frame in datasets.items():
        raw_inputs = frame.loc[:, INPUT_NAMES]
        _transformed = nondimensionalizer.transform(raw_inputs)
        transformed_inputs[name] = _transformed
        range_audits[name] = validate_nondimensionalization(
            raw_inputs, _transformed
        )
    return nondimensionalizer, range_audits, transformed_inputs


@app.cell
def _(mo, range_audits):
    dataset_selector = mo.ui.dropdown(
        options=list(range_audits), value="RERTR5", label="Dataset"
    )
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(dataset_selector, mo, range_audits):
    selected_range_audit = range_audits[dataset_selector.value]
    mo.vstack(
        [
            mo.md("## Input range verification"),
            mo.ui.table(selected_range_audit.reset_index(), selection=None),
        ]
    )
    return (selected_range_audit,)


@app.cell
def _(dataset_selector, mo, plt, transformed_inputs):
    transformed = transformed_inputs[dataset_selector.value]
    figure, axes = plt.subplots(3, 3, figsize=(12, 9))
    for axis, column in zip(axes.flat, transformed.columns, strict=True):
        axis.hist(transformed[column], bins=30, color="#3274a1", alpha=0.85)
        axis.set_title(column)
        axis.set_xlabel("Dimensionless value")
    figure.suptitle(f"{dataset_selector.value}: nondimensionalized inputs")
    figure.tight_layout()
    mo.mpl.interactive(figure)
    return (figure,)


@app.cell
def _(MODELING_OUTPUT_NAMES, datasets, mo):
    output_summary = mo.accordion(
        {
            name: frame.loc[:, MODELING_OUTPUT_NAMES[name]].describe()
            for name, frame in datasets.items()
        }
    )
    mo.vstack([mo.md("## Modeling-output summaries"), output_summary])
    return (output_summary,)


if __name__ == "__main__":
    app.run()
