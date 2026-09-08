from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class AudioChunk:
    waveform: torch.Tensor
    start_sec: float
    valid_num_samples: int
    input_num_samples: int
    padded: bool

    @property
    def valid_duration_sec(self) -> float:
        raise AttributeError(
            "valid_duration_sec depends on sample rate; use valid_num_samples / sample_rate."
        )


def load_audio(
    file_path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
    normalize_audio: bool = True,
) -> torch.Tensor:
    waveform, sample_rate = sf.read(str(file_path), always_2d=True)
    waveform = torch.tensor(waveform, dtype=torch.float32).transpose(0, 1)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
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
    min_model_input_samples: int = 400,
    pad_short_final_chunk: bool = True,
) -> List[AudioChunk]:
    """
    Partition a waveform into contiguous, non-overlapping chunks.

    The final remainder is retained. If it is shorter than the minimum input
    length required by a model, it can be zero-padded only for model execution.
    Downstream decoding must trim model frames back to ``valid_num_samples`` so
    padded samples never become part of the analyzed recording timeline.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if chunk_duration_ms <= 0:
        raise ValueError("chunk_duration_ms must be positive.")
    if min_model_input_samples <= 0:
        raise ValueError("min_model_input_samples must be positive.")

    chunk_size = int(round(sample_rate * (chunk_duration_ms / 1000.0)))
    if chunk_size <= 0:
        raise ValueError("chunk_duration_ms is too small for the sample rate.")

    chunks: List[AudioChunk] = []
    total_samples = int(waveform.shape[-1])

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = waveform[start:end].clone()
        valid_num_samples = int(chunk.shape[-1])
        padded = False

        if valid_num_samples < min_model_input_samples:
            if not pad_short_final_chunk:
                raise ValueError(
                    "Final chunk is shorter than min_model_input_samples. "
                    "Enable experiment.pad_short_final_chunk or choose a compatible minimum."
                )

        # If the final segment is shorter than the nominal chunk size, pad it
        # to the nominal size for model execution. The runner trims predicted
        # frames back to valid_num_samples before interval construction.
        if valid_num_samples < chunk_size and pad_short_final_chunk:
            pad_amount = chunk_size - valid_num_samples
            chunk = F.pad(chunk, (0, pad_amount))
            padded = True

        chunks.append(
            AudioChunk(
                waveform=chunk.contiguous(),
                start_sec=start / sample_rate,
                valid_num_samples=valid_num_samples,
                input_num_samples=int(chunk.shape[-1]),
                padded=padded,
            )
        )

    return chunks
