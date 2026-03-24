import pandas as pd


def aggregate_recording_metrics(per_recording_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        per_recording_df.groupby("chunk_duration_ms", as_index=False)[
            ["pcr", "por", "tci", "pli", "baseline_count", "matched_count", "omitted_count"]
        ]
        .mean()
        .sort_values("chunk_duration_ms")
        .reset_index(drop=True)
    )
    return agg


def aggregate_pvp(pvp_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        pvp_df.groupby(["chunk_duration_ms", "class"], as_index=False)[
            ["baseline_total", "matched_total", "omitted_total", "pcr", "por", "tci"]
        ]
        .mean()
        .sort_values(["chunk_duration_ms", "class"])
        .reset_index(drop=True)
    )
    return agg