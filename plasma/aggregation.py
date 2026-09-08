from __future__ import annotations

import numpy as np
import pandas as pd


RECORDING_METRICS = ["por", "tci", "tei", "atdi", "pli"]
PVP_METRICS = ["por", "tci", "tei", "atdi"]
RECORDING_GROUP_COLUMNS = ["dataset", "model", "chunk_duration_ms"]
PVP_GROUP_COLUMNS = ["dataset", "model", "chunk_duration_ms", "class"]


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_iterations: int,
    confidence_level: float,
    batch_size: int = 256,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value

    bootstrap_means = np.empty(bootstrap_iterations, dtype=float)
    offset = 0
    while offset < bootstrap_iterations:
        current_batch = min(batch_size, bootstrap_iterations - offset)
        sample_indices = rng.integers(
            low=0,
            high=len(values),
            size=(current_batch, len(values)),
        )
        bootstrap_means[offset : offset + current_batch] = (
            values[sample_indices].mean(axis=1)
        )
        offset += current_batch

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(bootstrap_means, alpha / 2.0))
    upper = float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0))
    return lower, upper


def _summarize_metric(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_iterations: int,
    confidence_level: float,
    bootstrap_batch_size: int,
) -> tuple[float, float, float, float]:
    if len(values) == 0:
        return (float("nan"),) * 4

    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ci_lower, ci_upper = _bootstrap_mean_ci(
        values=values,
        rng=rng,
        bootstrap_iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        batch_size=bootstrap_batch_size,
    )
    return mean_value, std_value, ci_lower, ci_upper


def _validate_statistics(
    bootstrap_iterations: int,
    confidence_level: float,
    bootstrap_batch_size: int,
) -> None:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be greater than zero.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if bootstrap_batch_size <= 0:
        raise ValueError("bootstrap_batch_size must be greater than zero.")


def aggregate_recording_metrics(
    per_recording_df: pd.DataFrame,
    bootstrap_iterations: int = 10000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    bootstrap_batch_size: int = 256,
) -> pd.DataFrame:
    _validate_statistics(
        bootstrap_iterations,
        confidence_level,
        bootstrap_batch_size,
    )

    required_columns = set(RECORDING_GROUP_COLUMNS + RECORDING_METRICS)
    missing = required_columns.difference(per_recording_df.columns)
    if missing:
        raise ValueError(f"Missing recording-metric columns: {sorted(missing)}")

    rng = np.random.default_rng(bootstrap_seed)
    rows = []

    for group_key, group in per_recording_df.groupby(RECORDING_GROUP_COLUMNS, sort=True):
        dataset, model, chunk_duration_ms = group_key
        row = {
            "dataset": dataset,
            "model": model,
            "chunk_duration_ms": int(chunk_duration_ms),
            "recording_count": int(len(group)),
            "baseline_count": float(group["baseline_count"].mean()),
            "matched_count": float(group["matched_count"].mean()),
            "omitted_count": float(group["omitted_count"].mean()),
        }

        for metric in RECORDING_METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            mean_value, std_value, ci_lower, ci_upper = _summarize_metric(
                values=values,
                rng=rng,
                bootstrap_iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                bootstrap_batch_size=bootstrap_batch_size,
            )
            row[metric] = mean_value
            row[f"{metric}_n"] = int(len(values))
            row[f"{metric}_std"] = std_value
            row[f"{metric}_ci_lower"] = ci_lower
            row[f"{metric}_ci_upper"] = ci_upper

        rows.append(row)

    return pd.DataFrame(rows).sort_values(RECORDING_GROUP_COLUMNS).reset_index(drop=True)


def aggregate_pvp(
    pvp_df: pd.DataFrame,
    bootstrap_iterations: int = 10000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    bootstrap_batch_size: int = 256,
) -> pd.DataFrame:
    _validate_statistics(
        bootstrap_iterations,
        confidence_level,
        bootstrap_batch_size,
    )

    required_columns = set(PVP_GROUP_COLUMNS + PVP_METRICS)
    missing = required_columns.difference(pvp_df.columns)
    if missing:
        raise ValueError(f"Missing PVP columns: {sorted(missing)}")

    rng = np.random.default_rng(bootstrap_seed)
    rows = []

    for group_key, group in pvp_df.groupby(PVP_GROUP_COLUMNS, sort=True):
        dataset, model, chunk_duration_ms, phoneme_class = group_key
        row = {
            "dataset": dataset,
            "model": model,
            "chunk_duration_ms": int(chunk_duration_ms),
            "class": phoneme_class,
            "recording_count": int(len(group)),
            "baseline_total": float(group["baseline_total"].mean()),
            "matched_total": float(group["matched_total"].mean()),
            "omitted_total": float(group["omitted_total"].mean()),
        }

        for metric in PVP_METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            mean_value, std_value, ci_lower, ci_upper = _summarize_metric(
                values=values,
                rng=rng,
                bootstrap_iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                bootstrap_batch_size=bootstrap_batch_size,
            )
            row[metric] = mean_value
            row[f"{metric}_n"] = int(len(values))
            row[f"{metric}_std"] = std_value
            row[f"{metric}_ci_lower"] = ci_lower
            row[f"{metric}_ci_upper"] = ci_upper

        rows.append(row)

    return pd.DataFrame(rows).sort_values(PVP_GROUP_COLUMNS).reset_index(drop=True)
