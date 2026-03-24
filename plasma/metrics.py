import math
from typing import Dict, List, Any

import pandas as pd

from plasma.alignment import AlignmentPair
from plasma.decoding import PhonemeInterval
from plasma.phoneme_classes import IPAToARPAbetConverter


def compute_recording_metrics(
    alignment: List[AlignmentPair],
    baseline_intervals: List[PhonemeInterval],
    stream_intervals: List[PhonemeInterval],
    pli_alpha: float,
    pli_beta: float,
    min_tci_epsilon: float,
) -> Dict[str, float]:
    baseline_count = len(baseline_intervals)
    matched_pairs = [p for p in alignment if p.op == "match"]
    omitted_pairs = [p for p in alignment if p.op == "delete"]

    nm = len(matched_pairs)
    no = len(omitted_pairs)
    nb = baseline_count

    pcr = nm / nb if nb > 0 else 0.0
    por = no / nb if nb > 0 else 0.0

    tci_values: List[float] = []
    for pair in matched_pairs:
        b = baseline_intervals[pair.baseline_idx]
        s = stream_intervals[pair.stream_idx]

        if b.duration_sec > 0:
            tci_values.append(s.duration_sec / b.duration_sec)

    tci = sum(tci_values) / len(tci_values) if tci_values else 0.0
    safe_tci = max(tci, min_tci_epsilon)
    pli = (pli_alpha * por) + (pli_beta * abs(math.log(safe_tci)))

    return {
        "baseline_count": nb,
        "matched_count": nm,
        "omitted_count": no,
        "pcr": pcr,
        "por": por,
        "tci": tci,
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
            tci_value = s.duration_sec / b.duration_sec if b.duration_sec > 0 else None

            rows.append(
                {
                    "class": phoneme_class,
                    "role": "matched",
                    "value": 1.0,
                    "baseline_idx": pair.baseline_idx,
                    "stream_idx": pair.stream_idx,
                }
            )
            if tci_value is not None:
                rows.append(
                    {
                        "class": phoneme_class,
                        "role": "tci",
                        "value": float(tci_value),
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
        avg_tci = sum(class_tci_values) / len(class_tci_values) if class_tci_values else 0.0

        pcr = matched_total / baseline_total if baseline_total > 0 else 0.0
        por = omitted_total / baseline_total if baseline_total > 0 else 0.0

        summary_rows.append(
            {
                "class": phoneme_class,
                "baseline_total": baseline_total,
                "matched_total": matched_total,
                "omitted_total": omitted_total,
                "pcr": pcr,
                "por": por,
                "tci": avg_tci,
            }
        )

    return pd.DataFrame(summary_rows).sort_values("class").reset_index(drop=True)