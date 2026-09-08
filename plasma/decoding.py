from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import unicodedata


@dataclass
class PhonemeInterval:
    label: str
    start_sec: float
    end_sec: float
    duration_sec: float


SPECIAL_TOKENS = {
    "",
    "|",
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "<blank>",
    "<sos>",
    "<eos>",
    "[PAD]",
    "[UNK]",
}

# Only unambiguous symbol-equivalence mappings are applied before alignment.
PHONEME_NORMALIZATION_MAP = {
    "ɡ": "g",
    "ɹ": "r",
    "aj": "aɪ",
    "aw": "aʊ",
    "ej": "eɪ",
    "ow": "oʊ",
    "oj": "ɔɪ",
    "t͡ʃ": "tʃ",
    "d͡ʒ": "dʒ",
    "ʧ": "tʃ",
    "ʤ": "dʒ",
}


def normalize_phoneme_token(token: str | None) -> str:
    if token is None:
        return ""

    token = unicodedata.normalize("NFC", token.strip())
    if token in SPECIAL_TOKENS:
        return ""

    return PHONEME_NORMALIZATION_MAP.get(token, token)


def trim_pred_ids_to_valid_audio(
    pred_ids,
    valid_num_samples: int,
    input_num_samples: int,
):
    """Trim frame predictions corresponding only to right-padding samples."""
    if input_num_samples <= 0:
        raise ValueError("input_num_samples must be positive.")
    if not 0 <= valid_num_samples <= input_num_samples:
        raise ValueError("valid_num_samples must be in [0, input_num_samples].")

    num_frames = len(pred_ids)
    if num_frames == 0 or valid_num_samples == input_num_samples:
        return pred_ids

    valid_frame_count = int(round(num_frames * valid_num_samples / input_num_samples))
    valid_frame_count = max(1, min(num_frames, valid_frame_count))
    return pred_ids[:valid_frame_count]


def frame_ids_to_intervals(
    pred_ids,
    id_to_token: Dict[int, str],
    blank_token_id: int,
    chunk_start_sec: float,
    total_audio_duration_sec: float,
) -> List[PhonemeInterval]:
    """
    Convert frame labels to phoneme intervals before sequence alignment.

    Blank frames are ignored. Consecutive identical non-blank labels form one
    interval. Timing is estimated uniformly from frame coverage over the valid
    audio duration supplied for this segment.
    """
    num_frames = len(pred_ids)
    if num_frames == 0 or total_audio_duration_sec <= 0:
        return []

    frame_duration = total_audio_duration_sec / num_frames
    intervals: List[PhonemeInterval] = []

    current_label = None
    current_start = None

    def flush(end_frame_idx: int) -> None:
        nonlocal current_label, current_start
        if current_label is None or current_start is None:
            return

        start_sec = chunk_start_sec + (current_start * frame_duration)
        end_sec = chunk_start_sec + (end_frame_idx * frame_duration)
        duration_sec = max(0.0, end_sec - start_sec)

        if current_label and duration_sec > 0:
            intervals.append(
                PhonemeInterval(
                    label=current_label,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=duration_sec,
                )
            )

        current_label = None
        current_start = None

    for frame_idx, token_id in enumerate(pred_ids):
        token_id = int(token_id)

        if token_id == blank_token_id:
            flush(frame_idx)
            continue

        raw_token = id_to_token.get(token_id, "")
        label = normalize_phoneme_token(raw_token)

        if not label:
            flush(frame_idx)
            continue

        if current_label is None:
            current_label = label
            current_start = frame_idx
        elif label != current_label:
            flush(frame_idx)
            current_label = label
            current_start = frame_idx

    flush(num_frames)
    return intervals


def intervals_to_sequence(intervals: List[PhonemeInterval]) -> List[str]:
    return [interval.label for interval in intervals]
