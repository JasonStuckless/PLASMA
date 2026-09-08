import numpy as np
import pandas as pd


RECORDING_METRICS = ["por", "tci", "tei", "atdi", "pli"]
PVP_METRICS = ["por", "tci", "tei", "atdi"]


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_iterations: int,
    confidence_level: float,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")

    if len(values) == 1:
        value = float(values[0])
        return value, value

    sample_indices = rng.integers(
        low=0,
        high=len(values),
        size=(bootstrap_iterations, len(values)),
    )
    bootstrap_means = values[sample_indices].mean(axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(bootstrap_means, alpha / 2.0))
    upper = float(np.quantile(bootstrap_means, 1.0 - (alpha / 2.0)))
    return lower, upper


def _summarize_metric(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_iterations: int,
    confidence_level: float,
) -> tuple[float, float, float, float]:
    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ci_lower, ci_upper = _bootstrap_mean_ci(
        values=values,
        rng=rng,
        bootstrap_iterations=bootstrap_iterations,
        confidence_level=confidence_level,
    )
    return mean_value, std_value, ci_lower, ci_upper


def aggregate_recording_metrics(
    per_recording_df: pd.DataFrame,
    bootstrap_iterations: int = 10000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be greater than zero.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(bootstrap_seed)
    rows = []

    for chunk_duration_ms, group in per_recording_df.groupby("chunk_duration_ms"):
        row = {
            "chunk_duration_ms": chunk_duration_ms,
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
            )

            # Keep the existing metric column as the aggregate mean so
            # downstream code remains compatible.
            row[metric] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_ci_lower"] = ci_lower
            row[f"{metric}_ci_upper"] = ci_upper

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("chunk_duration_ms")
        .reset_index(drop=True)
    )


def aggregate_pvp(
    pvp_df: pd.DataFrame,
    bootstrap_iterations: int = 10000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be greater than zero.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(bootstrap_seed)
    rows = []

    for (chunk_duration_ms, phoneme_class), group in pvp_df.groupby(
        ["chunk_duration_ms", "class"]
    ):
        row = {
            "chunk_duration_ms": chunk_duration_ms,
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
            )

            row[metric] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_ci_lower"] = ci_lower
            row[f"{metric}_ci_upper"] = ci_upper

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["chunk_duration_ms", "class"])
        .reset_index(drop=True)
    )
