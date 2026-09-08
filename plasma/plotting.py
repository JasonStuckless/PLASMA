from __future__ import annotations

from pathlib import Path
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RECORDING_PLOT_METRICS = ("por", "tci", "tei", "atdi", "pli")
PVP_PLOT_METRICS = ("por", "tci", "tei", "atdi")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def save_metric_curves(
    aggregate_df: pd.DataFrame,
    output_dir: str | Path,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for (dataset, model), group in aggregate_df.groupby(["dataset", "model"], sort=True):
        group = group.sort_values("chunk_duration_ms")
        condition_dir = output_dir / _safe_name(dataset) / _safe_name(model)
        condition_dir.mkdir(parents=True, exist_ok=True)

        for metric in RECORDING_PLOT_METRICS:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = group["chunk_duration_ms"].to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)
            ax.plot(x, y, marker="o")

            lower = group[f"{metric}_ci_lower"].to_numpy(dtype=float)
            upper = group[f"{metric}_ci_upper"].to_numpy(dtype=float)
            finite_ci = np.isfinite(lower) & np.isfinite(upper)
            if finite_ci.any():
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    where=finite_ci,
                    alpha=0.2,
                    label="95% bootstrap CI",
                )
                ax.legend()

            ax.set_xlabel("Chunk duration (ms)")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} vs chunk duration\n{dataset} | {model}")
            fig.tight_layout()

            output_path = condition_dir / f"{metric}_curve.png"
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            paths.append(output_path)

    return paths


def save_pvp_barplots(
    pvp_df: pd.DataFrame,
    output_dir: str | Path,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for (dataset, model), condition_df in pvp_df.groupby(["dataset", "model"], sort=True):
        condition_dir = output_dir / _safe_name(dataset) / _safe_name(model)
        condition_dir.mkdir(parents=True, exist_ok=True)

        classes = sorted(condition_df["class"].dropna().unique())
        chunk_durations = sorted(condition_df["chunk_duration_ms"].dropna().unique())
        x = np.arange(len(classes), dtype=float)
        group_width = 0.8
        bar_width = group_width / max(len(chunk_durations), 1)

        for metric in PVP_PLOT_METRICS:
            fig, ax = plt.subplots(figsize=(12, 6))

            for index, chunk_duration_ms in enumerate(chunk_durations):
                chunk_df = (
                    condition_df[condition_df["chunk_duration_ms"] == chunk_duration_ms]
                    .set_index("class")
                    .reindex(classes)
                )

                means = chunk_df[metric].to_numpy(dtype=float)
                lower = chunk_df[f"{metric}_ci_lower"].to_numpy(dtype=float)
                upper = chunk_df[f"{metric}_ci_upper"].to_numpy(dtype=float)

                lower_error = np.where(
                    np.isfinite(means) & np.isfinite(lower),
                    np.maximum(0.0, means - lower),
                    0.0,
                )
                upper_error = np.where(
                    np.isfinite(means) & np.isfinite(upper),
                    np.maximum(0.0, upper - means),
                    0.0,
                )
                yerr = np.vstack([lower_error, upper_error])

                offset = (index - (len(chunk_durations) - 1) / 2.0) * bar_width
                ax.bar(
                    x + offset,
                    means,
                    width=bar_width,
                    yerr=yerr,
                    capsize=2,
                    label=f"{int(chunk_duration_ms)} ms",
                )

            ax.set_xlabel("Phoneme class")
            ax.set_ylabel(metric.upper())
            ax.set_title(
                f"PVP {metric.upper()} by class and chunk duration\n{dataset} | {model}"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=30, ha="right")
            ax.legend(title="Chunk duration")
            fig.tight_layout()

            output_path = condition_dir / f"pvp_{metric}_barplot.png"
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            paths.append(output_path)

    return paths
