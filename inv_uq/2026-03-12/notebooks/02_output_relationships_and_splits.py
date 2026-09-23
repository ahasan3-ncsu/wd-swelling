"""Interactive Slice 2 audit of outputs, sensitivities, and fixed splits."""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import sys

    study_dir = Path(__file__).resolve().parents[1]
    if str(study_dir) not in sys.path:
        sys.path.insert(0, str(study_dir))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.data import load_dataset, validate_dataset
    from src.output_analysis import (
        audit_output_transformations,
        input_output_sensitivity,
        output_correlations,
        ranked_sensitivities,
        summarize_outputs,
    )
    from src.splitting import SPLIT_SEEDS, make_regression_splits

    return (
        SPLIT_SEEDS,
        audit_output_transformations,
        input_output_sensitivity,
        load_dataset,
        make_regression_splits,
        mo,
        np,
        output_correlations,
        pd,
        plt,
        ranked_sensitivities,
        summarize_outputs,
        validate_dataset,
    )


@app.cell
def _(load_dataset, validate_dataset):
    datasets = {
        name: load_dataset(name) for name in ("RERTR5", "RERTR12")
    }
    for name, _frame in datasets.items():
        validate_dataset(name, _frame)
    return (datasets,)


@app.cell
def _(mo):
    dataset_selector = mo.ui.dropdown(
        options=["RERTR5", "RERTR12"], value="RERTR5", label="Dataset"
    )
    mo.vstack(
        [
            mo.md("# Slice 2: output relationships and reproducible splits"),
            dataset_selector,
        ]
    )
    return (dataset_selector,)


@app.cell
def _(
    audit_output_transformations,
    dataset_selector,
    datasets,
    input_output_sensitivity,
    output_correlations,
    ranked_sensitivities,
    summarize_outputs,
):
    dataset = dataset_selector.value
    frame = datasets[dataset]
    summary = summarize_outputs(dataset, frame)
    correlations = output_correlations(dataset, frame)
    transformations = audit_output_transformations(dataset, frame)
    sensitivity = input_output_sensitivity(dataset, frame)
    sensitivity_ranking = ranked_sensitivities(sensitivity)
    return (
        correlations,
        dataset,
        frame,
        sensitivity,
        sensitivity_ranking,
        summary,
        transformations,
    )


@app.cell
def _(mo, summary, transformations):
    mo.vstack(
        [
            mo.md("## Physical-unit output distributions"),
            mo.ui.table(summary.reset_index(), selection=None),
            mo.md("## Transformation diagnostics"),
            mo.md(
                "Descriptive statistics use a dimensionless `log1p(y / scale)` diagnostic. "
                "No transformation is selected. Any later transform scale must be "
                "fitted on training data only."
            ),
            mo.ui.table(transformations.reset_index(), selection=None),
        ]
    )
    return


@app.cell
def _(correlations, dataset, mo, np, plt):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, (method, matrix) in zip(
        axes, correlations.items(), strict=True
    ):
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=55)
        axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
        axis.set_title(method.capitalize())
        for row in range(len(matrix.index)):
            for column in range(len(matrix.columns)):
                axis.text(
                    column,
                    row,
                    f"{matrix.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.suptitle(f"{dataset}: output correlations")
    figure.subplots_adjust(wspace=0.45, top=0.82, bottom=0.25)
    mo.mpl.interactive(figure)
    return (figure,)


@app.cell
def _(mo, sensitivity_ranking):
    top_sensitivities = (
        sensitivity_ranking.reset_index()
        .query("rank <= 3")
        .sort_values(["output", "rank"])
    )
    mo.vstack(
        [
            mo.md("## Initial input–output sensitivity screen"),
            mo.md(
                "These are marginal Spearman associations, not causal or "
                "interaction-aware sensitivity indices."
            ),
            mo.ui.table(top_sensitivities, selection=None),
        ]
    )
    return (top_sensitivities,)


@app.cell
def _(
    SPLIT_SEEDS,
    dataset,
    frame,
    make_regression_splits,
    mo,
    pd,
):
    splits = make_regression_splits(
        frame["Fuel_Swelling"], seed=SPLIT_SEEDS[dataset]
    )
    split_counts = (
        splits.groupby(["stratum", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md("Output selection is undecided and awaits review."),
            mo.md("## Fixed 70/15/15 split counts by swelling rank stratum"),
            mo.ui.table(split_counts, selection=None),
            mo.md(
                f"Seed: `{SPLIT_SEEDS[dataset]}`. The saved row indices in "
                "`artifacts/slice2/` are authoritative for later models."
            ),
        ]
    )
    return split_counts, splits


if __name__ == "__main__":
    app.run()
