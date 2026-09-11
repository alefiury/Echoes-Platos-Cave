#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the distribution of SICK relatedness scores to choose band thresholds."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset


HF_DATASET_ID = "RobZamp/sick"
HF_DATASET_REVISION = "refs/convert/parquet"

OUTPUT_DIR = Path("./sick_relatedness_analysis")

PLOT_THRESHOLD_SCHEME = "current"

CURRENT_LOW_MAX = 2.3
CURRENT_MEDIUM_MAX = 3.7

BIN_WIDTH = 0.1

FIGURE_DPI = 300

TERTILE_ROUNDING_DECIMALS = 1


def load_sick() -> pd.DataFrame:
    """Download and combine all SICK splits from Hugging Face."""
    print("=" * 80)
    print("LOADING SICK FROM HUGGING FACE")
    print("=" * 80)
    print(f"Dataset:  {HF_DATASET_ID}")
    print(f"Revision: {HF_DATASET_REVISION}")

    dataset = load_dataset(
        HF_DATASET_ID,
        revision=HF_DATASET_REVISION,
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected a DatasetDict, but received "
            f"{type(dataset).__name__}."
        )

    frames = []

    for split_name, split_dataset in dataset.items():
        frame = split_dataset.to_pandas()
        frame["split"] = split_name

        frames.append(frame)

        print(
            f"Loaded {split_name:12s}: "
            f"{len(frame):,} sentence pairs"
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )

    if "relatedness_score" not in dataframe.columns:
        raise ValueError(
            "The dataset does not contain the expected "
            "'relatedness_score' column.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe["relatedness_score"] = pd.to_numeric(
        dataframe["relatedness_score"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["relatedness_score"]
    ).copy()

    dataframe = dataframe[
        dataframe["relatedness_score"].between(1.0, 5.0)
    ].copy()

    print(f"\nTotal valid pairs: {len(dataframe):,}")

    return dataframe


def calculate_threshold_schemes(
    scores: pd.Series,
) -> Dict[str, Tuple[float, float]]:
    minimum = float(scores.min())
    maximum = float(scores.max())

    score_range = maximum - minimum

    equal_width_low = minimum + score_range / 3
    equal_width_medium = minimum + 2 * score_range / 3

    empirical_low = float(scores.quantile(1 / 3))
    empirical_medium = float(scores.quantile(2 / 3))

    rounded_low = round(
        empirical_low,
        TERTILE_ROUNDING_DECIMALS,
    )
    rounded_medium = round(
        empirical_medium,
        TERTILE_ROUNDING_DECIMALS,
    )

    schemes = {
        "current": (
            CURRENT_LOW_MAX,
            CURRENT_MEDIUM_MAX,
        ),
        "equal_width": (
            equal_width_low,
            equal_width_medium,
        ),
        "empirical_tertiles": (
            empirical_low,
            empirical_medium,
        ),
        "rounded_tertiles": (
            rounded_low,
            rounded_medium,
        ),
    }

    return schemes


def assign_similarity_group(
    scores: pd.Series,
    low_max: float,
    medium_max: float,
) -> pd.Series:
    """Assign every relatedness score to low, medium, or high."""
    conditions = [
        scores <= low_max,
        (scores > low_max) & (scores <= medium_max),
        scores > medium_max,
    ]

    labels = [
        "low",
        "medium",
        "high",
    ]

    groups = np.select(
        conditions,
        labels,
        default="unassigned",
    )

    return pd.Series(
        groups,
        index=scores.index,
        dtype="object",
    )


def save_descriptive_statistics(
    scores: pd.Series,
) -> None:
    """Save general descriptive statistics."""
    statistics = pd.DataFrame(
        {
            "statistic": [
                "count",
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
                "median",
                "skewness",
                "kurtosis",
            ],
            "value": [
                scores.count(),
                scores.mean(),
                scores.std(),
                scores.min(),
                scores.max(),
                scores.median(),
                scores.skew(),
                scores.kurtosis(),
            ],
        }
    )

    output_path = OUTPUT_DIR / "relatedness_statistics.csv"

    statistics.to_csv(
        output_path,
        index=False,
    )

    print("\nDescriptive statistics:")
    print(statistics.to_string(index=False))

    print(f"\nSaved: {output_path}")


def save_quantiles(
    scores: pd.Series,
) -> None:
    """Calculate and save relevant quantiles."""
    requested_quantiles = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.25,
        1 / 3,
        0.40,
        0.50,
        0.60,
        2 / 3,
        0.75,
        0.80,
        0.90,
        0.95,
        0.99,
    ]

    quantile_values = scores.quantile(
        requested_quantiles
    )

    quantiles = pd.DataFrame(
        {
            "quantile": requested_quantiles,
            "percentage": [
                quantile * 100
                for quantile in requested_quantiles
            ],
            "relatedness_score": quantile_values.values,
        }
    )

    output_path = OUTPUT_DIR / "relatedness_quantiles.csv"

    quantiles.to_csv(
        output_path,
        index=False,
    )

    print("\nSelected quantiles:")
    print(
        quantiles.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved: {output_path}")


def save_score_counts(
    scores: pd.Series,
) -> None:
    """Save the number of pairs associated with each exact score."""
    score_counts = (
        scores
        .value_counts()
        .sort_index()
        .rename_axis("relatedness_score")
        .reset_index(name="count")
    )

    score_counts["percentage"] = (
        score_counts["count"]
        / score_counts["count"].sum()
        * 100
    )

    score_counts["cumulative_count"] = (
        score_counts["count"].cumsum()
    )

    score_counts["cumulative_percentage"] = (
        score_counts["percentage"].cumsum()
    )

    output_path = OUTPUT_DIR / "relatedness_score_counts.csv"

    score_counts.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path}")


def build_threshold_comparison(
    scores: pd.Series,
    schemes: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Compare group sizes produced by different threshold schemes."""
    records = []

    for scheme_name, thresholds in schemes.items():
        low_max, medium_max = thresholds

        groups = assign_similarity_group(
            scores=scores,
            low_max=low_max,
            medium_max=medium_max,
        )

        counts = groups.value_counts()

        for group_name in ("low", "medium", "high"):
            count = int(counts.get(group_name, 0))

            records.append(
                {
                    "scheme": scheme_name,
                    "low_max": low_max,
                    "medium_max": medium_max,
                    "group": group_name,
                    "count": count,
                    "percentage": count / len(scores) * 100,
                }
            )

    comparison = pd.DataFrame(records)

    output_path = OUTPUT_DIR / "threshold_comparison.csv"

    comparison.to_csv(
        output_path,
        index=False,
    )

    print("\nThreshold comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved: {output_path}")

    return comparison


def add_threshold_information(
    axes: plt.Axes,
    scores: pd.Series,
    low_max: float,
    medium_max: float,
) -> None:
    """Add threshold lines and group counts to a plot."""
    groups = assign_similarity_group(
        scores=scores,
        low_max=low_max,
        medium_max=medium_max,
    )

    counts = groups.value_counts()

    low_count = int(counts.get("low", 0))
    medium_count = int(counts.get("medium", 0))
    high_count = int(counts.get("high", 0))

    axes.axvline(
        low_max,
        linestyle="--",
        linewidth=2,
        label=f"Low maximum = {low_max:.3f}",
    )

    axes.axvline(
        medium_max,
        linestyle="--",
        linewidth=2,
        label=f"Medium maximum = {medium_max:.3f}",
    )

    annotation = (
        f"Low: {low_count:,} "
        f"({low_count / len(scores) * 100:.1f}%)\n"
        f"Medium: {medium_count:,} "
        f"({medium_count / len(scores) * 100:.1f}%)\n"
        f"High: {high_count:,} "
        f"({high_count / len(scores) * 100:.1f}%)"
    )

    axes.text(
        0.98,
        0.95,
        annotation,
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.85,
        },
    )


def plot_histogram(
    scores: pd.Series,
    scheme_name: str,
    thresholds: Tuple[float, float],
) -> None:
    """Plot the relatedness-score histogram."""
    low_max, medium_max = thresholds

    minimum_bin = np.floor(scores.min() / BIN_WIDTH) * BIN_WIDTH
    maximum_bin = np.ceil(scores.max() / BIN_WIDTH) * BIN_WIDTH

    bins = np.arange(
        minimum_bin,
        maximum_bin + BIN_WIDTH,
        BIN_WIDTH,
    )

    figure, axes = plt.subplots(
        figsize=(12, 7),
    )

    axes.hist(
        scores,
        bins=bins,
        edgecolor="black",
        linewidth=0.4,
    )

    add_threshold_information(
        axes=axes,
        scores=scores,
        low_max=low_max,
        medium_max=medium_max,
    )

    axes.set_title(
        "Distribution of SICK Relatedness Scores\n"
        f"Threshold scheme: {scheme_name}"
    )
    axes.set_xlabel("Relatedness score")
    axes.set_ylabel("Number of sentence pairs")

    axes.set_xlim(
        scores.min() - 0.05,
        scores.max() + 0.05,
    )

    axes.set_xticks(
        np.arange(1.0, 5.01, 0.25)
    )

    axes.grid(
        axis="y",
        alpha=0.25,
    )

    axes.legend()

    figure.tight_layout()

    output_path = OUTPUT_DIR / "relatedness_histogram.png"

    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def plot_ecdf(
    scores: pd.Series,
    scheme_name: str,
    thresholds: Tuple[float, float],
) -> None:
    low_max, medium_max = thresholds

    sorted_scores = np.sort(scores.to_numpy())

    cumulative_probability = (
        np.arange(1, len(sorted_scores) + 1)
        / len(sorted_scores)
    )

    figure, axes = plt.subplots(
        figsize=(12, 7),
    )

    axes.step(
        sorted_scores,
        cumulative_probability,
        where="post",
        linewidth=2,
    )

    axes.axvline(
        low_max,
        linestyle="--",
        linewidth=2,
        label=f"Low maximum = {low_max:.3f}",
    )

    axes.axvline(
        medium_max,
        linestyle="--",
        linewidth=2,
        label=f"Medium maximum = {medium_max:.3f}",
    )

    low_cumulative = (
        scores.le(low_max).mean()
    )

    medium_cumulative = (
        scores.le(medium_max).mean()
    )

    axes.axhline(
        low_cumulative,
        linestyle=":",
        linewidth=1.5,
    )

    axes.axhline(
        medium_cumulative,
        linestyle=":",
        linewidth=1.5,
    )

    axes.scatter(
        [low_max, medium_max],
        [low_cumulative, medium_cumulative],
        zorder=5,
    )

    axes.annotate(
        f"{low_cumulative * 100:.1f}%",
        xy=(low_max, low_cumulative),
        xytext=(8, 8),
        textcoords="offset points",
    )

    axes.annotate(
        f"{medium_cumulative * 100:.1f}%",
        xy=(medium_max, medium_cumulative),
        xytext=(8, 8),
        textcoords="offset points",
    )

    axes.set_title(
        "Empirical Cumulative Distribution of SICK Relatedness\n"
        f"Threshold scheme: {scheme_name}"
    )
    axes.set_xlabel("Relatedness score")
    axes.set_ylabel(
        "Fraction of pairs with score less than or equal to x"
    )

    axes.set_xlim(
        scores.min() - 0.05,
        scores.max() + 0.05,
    )
    axes.set_ylim(0.0, 1.02)

    axes.set_xticks(
        np.arange(1.0, 5.01, 0.25)
    )

    axes.set_yticks(
        np.arange(0.0, 1.01, 0.1)
    )

    axes.grid(alpha=0.25)
    axes.legend()

    figure.tight_layout()

    output_path = OUTPUT_DIR / "relatedness_ecdf.png"

    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def plot_threshold_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Plot group counts produced by each threshold strategy."""
    plot_data = comparison.pivot(
        index="scheme",
        columns="group",
        values="count",
    )

    plot_data = plot_data[
        ["low", "medium", "high"]
    ]

    figure, axes = plt.subplots(
        figsize=(12, 7),
    )

    plot_data.plot(
        kind="bar",
        ax=axes,
        edgecolor="black",
        linewidth=0.4,
    )

    axes.set_title(
        "Low, Medium, and High Group Sizes by Threshold Strategy"
    )
    axes.set_xlabel("Threshold strategy")
    axes.set_ylabel("Number of sentence pairs")

    axes.tick_params(
        axis="x",
        rotation=20,
    )

    axes.grid(
        axis="y",
        alpha=0.25,
    )

    axes.legend(
        title="Relatedness group"
    )

    for container in axes.containers:
        axes.bar_label(
            container,
            fmt="%d",
            padding=3,
            fontsize=8,
        )

    figure.tight_layout()

    output_path = OUTPUT_DIR / "threshold_comparison.png"

    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def print_threshold_schemes(
    scores: pd.Series,
    schemes: Dict[str, Tuple[float, float]],
) -> None:
    """Print the thresholds and resulting group sizes."""
    print("\n" + "=" * 80)
    print("THRESHOLD SCHEMES")
    print("=" * 80)

    for scheme_name, thresholds in schemes.items():
        low_max, medium_max = thresholds

        groups = assign_similarity_group(
            scores=scores,
            low_max=low_max,
            medium_max=medium_max,
        )

        counts = groups.value_counts()

        print(f"\n{scheme_name}")
        print(f"  Low:    score <= {low_max:.4f}")
        print(
            f"  Medium: {low_max:.4f} < score "
            f"<= {medium_max:.4f}"
        )
        print(f"  High:   score > {medium_max:.4f}")

        for group_name in ("low", "medium", "high"):
            count = int(counts.get(group_name, 0))
            percentage = count / len(scores) * 100

            print(
                f"    {group_name:6s}: "
                f"{count:5,d} pairs "
                f"({percentage:6.2f}%)"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_sick()

    scores = dataframe["relatedness_score"]

    schemes = calculate_threshold_schemes(scores)

    if PLOT_THRESHOLD_SCHEME not in schemes:
        raise ValueError(
            f"Unknown threshold scheme: {PLOT_THRESHOLD_SCHEME}\n"
            f"Available schemes: {list(schemes)}"
        )

    print_threshold_schemes(
        scores=scores,
        schemes=schemes,
    )

    save_descriptive_statistics(scores)
    save_quantiles(scores)
    save_score_counts(scores)

    comparison = build_threshold_comparison(
        scores=scores,
        schemes=schemes,
    )

    selected_thresholds = schemes[
        PLOT_THRESHOLD_SCHEME
    ]

    plot_histogram(
        scores=scores,
        scheme_name=PLOT_THRESHOLD_SCHEME,
        thresholds=selected_thresholds,
    )

    plot_ecdf(
        scores=scores,
        scheme_name=PLOT_THRESHOLD_SCHEME,
        thresholds=selected_thresholds,
    )

    plot_threshold_comparison(
        comparison=comparison,
    )

    dataframe.to_csv(
        OUTPUT_DIR / "sick_all_pairs.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
