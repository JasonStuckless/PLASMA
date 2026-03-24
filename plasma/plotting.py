from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_metric_curve(df: pd.DataFrame, metric: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="chunk_duration_ms", y=metric, marker="o")
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

    for metric in ["pcr", "por", "tci"]:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pvp_df, x="class", y=metric, hue="chunk_duration_ms")
        plt.xlabel("Phoneme class")
        plt.ylabel(metric.upper())
        plt.title(f"PVP {metric.upper()} by class and chunk duration")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        output_path = output_dir / f"pvp_{metric}_barplot.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()