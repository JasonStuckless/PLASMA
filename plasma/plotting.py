from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_metric_curve(df: pd.DataFrame, metric: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ci_lower_col = f"{metric}_ci_lower"
    ci_upper_col = f"{metric}_ci_upper"

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="chunk_duration_ms", y=metric, marker="o")

    if ci_lower_col in df.columns and ci_upper_col in df.columns:
        plt.fill_between(
            df["chunk_duration_ms"],
            df[ci_lower_col],
            df[ci_upper_col],
            alpha=0.2,
            label="95% bootstrap CI",
        )
        plt.legend()

    plt.xlabel("Chunk duration (ms)")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs chunk duration")
    plt.tight_layout()

    output_path = output_dir / f"{metric}_curve.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_pvp_barplots(pvp_df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(pvp_df["class"].dropna().unique())
    chunk_durations = sorted(pvp_df["chunk_duration_ms"].dropna().unique())
    x = np.arange(len(classes), dtype=float)

    group_width = 0.8
    bar_width = group_width / max(len(chunk_durations), 1)

    for metric in ["por", "tci", "tei", "atdi"]:
        fig, ax = plt.subplots(figsize=(12, 6))

        for index, chunk_duration_ms in enumerate(chunk_durations):
            chunk_df = (
                pvp_df[pvp_df["chunk_duration_ms"] == chunk_duration_ms]
                .set_index("class")
                .reindex(classes)
            )

            means = chunk_df[metric].to_numpy(dtype=float)
            lower = chunk_df[f"{metric}_ci_lower"].to_numpy(dtype=float)
            upper = chunk_df[f"{metric}_ci_upper"].to_numpy(dtype=float)

            lower_error = np.maximum(0.0, means - lower)
            upper_error = np.maximum(0.0, upper - means)
            yerr = np.vstack([lower_error, upper_error])

            offset = (
                index - (len(chunk_durations) - 1) / 2.0
            ) * bar_width

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
            f"PVP {metric.upper()} by class and chunk duration "
            "with 95% bootstrap confidence intervals"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.legend(title="Chunk duration")
        fig.tight_layout()

        output_path = output_dir / f"pvp_{metric}_barplot.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
