from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PhonemeInterval:
    label: str
    start_sec: float
    end_sec: float
    duration_sec: float


def clean_token(token: str) -> str:
    if token is None:
        return ""

    token = token.strip()

    # Hugging Face CTC tokenizers often use these markers
    token = token.replace("|", " ")
    token = token.replace("<s>", "")
    token = token.replace("</s>", "")
    token = token.replace("<pad>", "")
    token = token.replace("<unk>", "")

    return token.strip()


def frame_ids_to_intervals(
    pred_ids,
    id_to_token: Dict[int, str],
    blank_token_id: int,
    chunk_start_sec: float,
    total_audio_duration_sec: float,
) -> List[PhonemeInterval]:
    """
    Convert raw frame labels to intervals before CTC collapse.
    Blank frames are ignored. Consecutive identical non-blank labels become
    one interval. Timing is derived from frame index coverage over the chunk.
    """
    num_frames = len(pred_ids)
    if num_frames == 0:
        return []

    frame_duration = total_audio_duration_sec / num_frames
    intervals: List[PhonemeInterval] = []

    current_label = None
    current_start = None

    def flush(end_frame_idx: int):
        nonlocal current_label, current_start, intervals
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
        label = clean_token(raw_token)

        if not label:
            flush(frame_idx)
            continue

        if current_label is None:
            current_label = label
            current_start = frame_idx
        elif label == current_label:
            continue
        else:
            flush(frame_idx)
            current_label = label
            current_start = frame_idx

    flush(num_frames)
    return intervals


def intervals_to_sequence(intervals: List[PhonemeInterval]) -> List[str]:
    return [x.label for x in intervals]