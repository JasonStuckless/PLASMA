from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio


def load_audio(
    file_path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
    normalize_audio: bool = True,
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(file_path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )

    if normalize_audio:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    return waveform.squeeze(0).contiguous()


def chunk_waveform_strict(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration_ms: int,
) -> List[Tuple[torch.Tensor, float]]:
    chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0))
    chunks: List[Tuple[torch.Tensor, float]] = []

    start = 0
    total_samples = waveform.shape[0]

    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = waveform[start:end].clone()
        start_time_sec = start / sample_rate
        chunks.append((chunk, start_time_sec))
        start = end

    return chunks