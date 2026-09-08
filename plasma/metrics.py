from typing import Dict, List, Any

import pandas as pd

from plasma.alignment import AlignmentPair
from plasma.decoding import PhonemeInterval
from plasma.phoneme_classes import IPAToARPAbetConverter


def compute_recording_metrics(
    alignment: List[AlignmentPair],
    baseline_intervals: List[PhonemeInterval],
    stream_intervals: List[PhonemeInterval],
) -> Dict[str, float]:
    baseline_count = len(baseline_intervals)
    matched_pairs = [p for p in alignment if p.op == "match"]
    omitted_pairs = [p for p in alignment if p.op == "delete"]

    nm = len(matched_pairs)
    no = len(omitted_pairs)
    nb = baseline_count

    por = no / nb if nb > 0 else 0.0

    duration_ratios: List[float] = []
    pli_distortions: List[float] = []

    for pair in matched_pairs:
        b = baseline_intervals[pair.baseline_idx]
        s = stream_intervals[pair.stream_idx]

        if b.duration_sec > 0:
            duration_ratios.append(s.duration_sec / b.duration_sec)

        max_duration = max(b.duration_sec, s.duration_sec)
        if max_duration > 0:
            pli_distortions.append(
                abs(s.duration_sec - b.duration_sec) / max_duration
            )
        else:
            pli_distortions.append(0.0)

    # Directional temporal metrics. Each metric is normalized over all
    # matched phonemes, so it captures both prevalence and magnitude.
    if duration_ratios:
        tci = sum(max(0.0, 1.0 - ratio) for ratio in duration_ratios) / len(duration_ratios)
        tei = sum(max(0.0, ratio - 1.0) for ratio in duration_ratios) / len(duration_ratios)
        atdi = sum(abs(ratio - 1.0) for ratio in duration_ratios) / len(duration_ratios)
    else:
        tci = 0.0
        tei = 0.0
        atdi = 0.0

    # Weight-free normalized Phoneme Loss Index.
    # Each omitted baseline phoneme contributes 1.0.
    # Each matched phoneme contributes its bounded duration distortion:
    # |d_s - d_b| / max(d_s, d_b).
    pli = (
        (no + sum(pli_distortions)) / nb
        if nb > 0
        else 0.0
    )

    return {
        "baseline_count": nb,
        "matched_count": nm,
        "omitted_count": no,
        "por": por,
        "tci": tci,
        "tei": tei,
        "atdi": atdi,
        "pli": pli,
    }


def compute_pvp(
    alignment: List[AlignmentPair],
    baseline_intervals: List[PhonemeInterval],
    stream_intervals: List[PhonemeInterval],
) -> pd.DataFrame:
    converter = IPAToARPAbetConverter()

    rows: List[Dict[str, Any]] = []

    # Baseline class counts
    for idx, interval in enumerate(baseline_intervals):
        rows.append(
            {
                "class": converter.phoneme_class(interval.label),
                "role": "baseline_total",
                "value": 1.0,
                "baseline_idx": idx,
                "stream_idx": None,
            }
        )

    for pair in alignment:
        if pair.op == "match":
            b = baseline_intervals[pair.baseline_idx]
            s = stream_intervals[pair.stream_idx]
            phoneme_class = converter.phoneme_class(b.label)
            duration_ratio = s.duration_sec / b.duration_sec if b.duration_sec > 0 else None

            rows.append(
                {
                    "class": phoneme_class,
                    "role": "matched",
                    "value": 1.0,
                    "baseline_idx": pair.baseline_idx,
                    "stream_idx": pair.stream_idx,
                }
            )
            if duration_ratio is not None:
                temporal_values = {
                    "tci": max(0.0, 1.0 - duration_ratio),
                    "tei": max(0.0, duration_ratio - 1.0),
                    "atdi": abs(duration_ratio - 1.0),
                }
                for role, value in temporal_values.items():
                    rows.append(
                        {
                            "class": phoneme_class,
                            "role": role,
                            "value": float(value),
                            "baseline_idx": pair.baseline_idx,
                            "stream_idx": pair.stream_idx,
                        }
                    )
        elif pair.op == "delete":
            b = baseline_intervals[pair.baseline_idx]
            rows.append(
                {
                    "class": converter.phoneme_class(b.label),
                    "role": "omitted",
                    "value": 1.0,
                    "baseline_idx": pair.baseline_idx,
                    "stream_idx": None,
                }
            )

    df = pd.DataFrame(rows)

    summary_rows: List[Dict[str, Any]] = []
    for phoneme_class, class_df in df.groupby("class"):
        baseline_total = float((class_df["role"] == "baseline_total").sum())
        matched_total = float((class_df["role"] == "matched").sum())
        omitted_total = float((class_df["role"] == "omitted").sum())

        class_tci_values = class_df.loc[class_df["role"] == "tci", "value"].tolist()
        class_tei_values = class_df.loc[class_df["role"] == "tei", "value"].tolist()
        class_atdi_values = class_df.loc[class_df["role"] == "atdi", "value"].tolist()

        avg_tci = sum(class_tci_values) / len(class_tci_values) if class_tci_values else 0.0
        avg_tei = sum(class_tei_values) / len(class_tei_values) if class_tei_values else 0.0
        avg_atdi = sum(class_atdi_values) / len(class_atdi_values) if class_atdi_values else 0.0

        por = omitted_total / baseline_total if baseline_total > 0 else 0.0

        summary_rows.append(
            {
                "class": phoneme_class,
                "baseline_total": baseline_total,
                "matched_total": matched_total,
                "omitted_total": omitted_total,
                "por": por,
                "tci": avg_tci,
                "tei": avg_tei,
                "atdi": avg_atdi,
            }
        )

    return pd.DataFrame(summary_rows).sort_values("class").reset_index(drop=True)
