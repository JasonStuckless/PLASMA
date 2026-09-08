from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from plasma.alignment import AlignmentPair
from plasma.decoding import PhonemeInterval
from plasma.phoneme_classes import IPAToARPAbetConverter, PVP_CLASSES


def compute_recording_metrics(
    alignment: List[AlignmentPair],
    baseline_intervals: List[PhonemeInterval],
    stream_intervals: List[PhonemeInterval],
) -> Dict[str, float]:
    baseline_count = len(baseline_intervals)
    matched_pairs = [pair for pair in alignment if pair.op == "match"]
    omitted_pairs = [pair for pair in alignment if pair.op == "delete"]

    nm = len(matched_pairs)
    no = len(omitted_pairs)
    nb = baseline_count

    por = no / nb if nb > 0 else float("nan")

    duration_ratios: List[float] = []
    pli_distortions: List[float] = []

    for pair in matched_pairs:
        if pair.baseline_idx is None or pair.stream_idx is None:
            continue

        baseline_interval = baseline_intervals[pair.baseline_idx]
        stream_interval = stream_intervals[pair.stream_idx]

        if baseline_interval.duration_sec > 0:
            duration_ratios.append(
                stream_interval.duration_sec / baseline_interval.duration_sec
            )

        max_duration = max(
            baseline_interval.duration_sec,
            stream_interval.duration_sec,
        )
        if max_duration > 0:
            pli_distortions.append(
                abs(stream_interval.duration_sec - baseline_interval.duration_sec)
                / max_duration
            )

    if duration_ratios:
        ratios = np.asarray(duration_ratios, dtype=float)
        tci = float(np.maximum(0.0, 1.0 - ratios).mean())
        tei = float(np.maximum(0.0, ratios - 1.0).mean())
        atdi = float(np.abs(ratios - 1.0).mean())
    else:
        # Temporal distortion is undefined when no baseline phoneme is matched.
        tci = float("nan")
        tei = float("nan")
        atdi = float("nan")

    # PLI remains defined when all baseline phonemes are omitted: each omitted
    # baseline phoneme contributes 1. Matched phonemes contribute bounded
    # temporal distortion in [0, 1).
    pli = (
        (no + sum(pli_distortions)) / nb
        if nb > 0
        else float("nan")
    )

    return {
        "baseline_count": float(nb),
        "matched_count": float(nm),
        "omitted_count": float(no),
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

    stats: Dict[str, Dict[str, Any]] = {
        phoneme_class: {
            "baseline_total": 0,
            "matched_total": 0,
            "omitted_total": 0,
            "tci_values": [],
            "tei_values": [],
            "atdi_values": [],
        }
        for phoneme_class in PVP_CLASSES
    }

    for interval in baseline_intervals:
        phoneme_class = converter.phoneme_class(interval.label)
        if phoneme_class in stats:
            stats[phoneme_class]["baseline_total"] += 1

    for pair in alignment:
        if pair.op == "match":
            if pair.baseline_idx is None or pair.stream_idx is None:
                continue
            baseline_interval = baseline_intervals[pair.baseline_idx]
            stream_interval = stream_intervals[pair.stream_idx]
            phoneme_class = converter.phoneme_class(baseline_interval.label)
            if phoneme_class not in stats:
                continue

            stats[phoneme_class]["matched_total"] += 1
            if baseline_interval.duration_sec > 0:
                ratio = stream_interval.duration_sec / baseline_interval.duration_sec
                stats[phoneme_class]["tci_values"].append(max(0.0, 1.0 - ratio))
                stats[phoneme_class]["tei_values"].append(max(0.0, ratio - 1.0))
                stats[phoneme_class]["atdi_values"].append(abs(ratio - 1.0))

        elif pair.op == "delete":
            if pair.baseline_idx is None:
                continue
            baseline_interval = baseline_intervals[pair.baseline_idx]
            phoneme_class = converter.phoneme_class(baseline_interval.label)
            if phoneme_class in stats:
                stats[phoneme_class]["omitted_total"] += 1

    summary_rows: List[Dict[str, Any]] = []
    for phoneme_class in PVP_CLASSES:
        class_stats = stats[phoneme_class]
        baseline_total = int(class_stats["baseline_total"])
        if baseline_total == 0:
            # PVP uncertainty is computed only across recordings in which the
            # class is represented in the full-context baseline.
            continue

        matched_total = int(class_stats["matched_total"])
        omitted_total = int(class_stats["omitted_total"])
        por = omitted_total / baseline_total

        temporal_values = {
            metric: class_stats[f"{metric}_values"]
            for metric in ("tci", "tei", "atdi")
        }

        summary_rows.append(
            {
                "class": phoneme_class,
                "baseline_total": float(baseline_total),
                "matched_total": float(matched_total),
                "omitted_total": float(omitted_total),
                "por": float(por),
                "tci": (
                    float(np.mean(temporal_values["tci"]))
                    if temporal_values["tci"]
                    else float("nan")
                ),
                "tei": (
                    float(np.mean(temporal_values["tei"]))
                    if temporal_values["tei"]
                    else float("nan")
                ),
                "atdi": (
                    float(np.mean(temporal_values["atdi"]))
                    if temporal_values["atdi"]
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(summary_rows)
