from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os

import torch
from huggingface_hub import hf_hub_download
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from transformers import (
    AutoModel,
    HubertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


@dataclass
class ModelBundle:
    model_key: str
    model_name: str
    model_type: str
    processor: Optional[Any]
    model: Any
    device: torch.device
    id_to_token: Dict[int, str]
    blank_token_id: int


def _get_device(use_gpu_if_available: bool) -> torch.device:
    if use_gpu_if_available and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _configure_espeak() -> None:
    espeak_dll = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

    if not os.path.exists(espeak_dll):
        raise FileNotFoundError(
            f"Could not find eSpeak NG DLL at: {espeak_dll}\n"
            "Install eSpeak NG or update this path in "
            "plasma/model_utils.py"
        )

    EspeakWrapper.set_library(espeak_dll)


def _build_token_map_from_processor(
    processor,
    vocab_size: int,
) -> Dict[int, str]:
    tokens = processor.tokenizer.convert_ids_to_tokens(
        list(range(vocab_size))
    )

    return {
        token_id: token
        for token_id, token in enumerate(tokens)
    }


def _load_wav2vec2(
    model_key: str,
    model_name: str,
    device: torch.device,
) -> ModelBundle:
    _configure_espeak()

    processor = Wav2Vec2Processor.from_pretrained(
        model_name
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        model_name
    ).to(device)

    model.eval()

    id_to_token = _build_token_map_from_processor(
        processor,
        model.config.vocab_size,
    )

    blank_token_id = processor.tokenizer.pad_token_id

    if blank_token_id is None:
        blank_token_id = 0

    return ModelBundle(
        model_key=model_key,
        model_name=model_name,
        model_type="wav2vec2_ctc",
        processor=processor,
        model=model,
        device=device,
        id_to_token=id_to_token,
        blank_token_id=blank_token_id,
    )


def _load_hubert(
    model_key: str,
    model_name: str,
    device: torch.device,
) -> ModelBundle:
    processor = Wav2Vec2Processor.from_pretrained(
        model_name
    )

    model = HubertForCTC.from_pretrained(
        model_name
    ).to(device)

    model.eval()

    id_to_token = _build_token_map_from_processor(
        processor,
        model.config.vocab_size,
    )

    blank_token_id = processor.tokenizer.pad_token_id

    if blank_token_id is None:
        blank_token_id = 0

    return ModelBundle(
        model_key=model_key,
        model_name=model_name,
        model_type="hubert_ctc",
        processor=processor,
        model=model,
        device=device,
        id_to_token=id_to_token,
        blank_token_id=blank_token_id,
    )


def _load_phoneticxeus_vocab(
    model_name: str,
    revision: Optional[str],
) -> Dict[int, str]:
    local_path = Path(model_name)

    if local_path.is_dir():
        vocab_path = local_path / "ipa_vocab.json"
    else:
        vocab_path = Path(
            hf_hub_download(
                repo_id=model_name,
                filename="ipa_vocab.json",
                revision=revision,
            )
        )

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"PhoneticXeus vocabulary not found: "
            f"{vocab_path}"
        )

    with vocab_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        vocab = json.load(f)

    return {
        int(token_id): token
        for token, token_id in vocab.items()
    }


def _load_phoneticxeus(
    model_key: str,
    model_name: str,
    revision: Optional[str],
    device: torch.device,
) -> ModelBundle:
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    ).to(device)

    model.eval()

    id_to_token = _load_phoneticxeus_vocab(
        model_name=model_name,
        revision=revision,
    )

    blank_token_id = None

    for token_id, token in id_to_token.items():
        if token == "<blank>":
            blank_token_id = token_id
            break

    if blank_token_id is None:
        raise ValueError(
            "PhoneticXeus vocabulary does not contain "
            "<blank>."
        )

    return ModelBundle(
        model_key=model_key,
        model_name=model_name,
        model_type="phoneticxeus",
        processor=None,
        model=model,
        device=device,
        id_to_token=id_to_token,
        blank_token_id=blank_token_id,
    )


def load_model(
    model_key: str,
    model_name: str,
    model_type: str,
    use_gpu_if_available: bool = True,
    revision: Optional[str] = None,
) -> ModelBundle:
    device = _get_device(
        use_gpu_if_available
    )

    print()
    print(f"Loading model: {model_key}")
    print(f"Repository: {model_name}")
    print(f"Type: {model_type}")
    print(f"Device: {device}")

    if model_type == "wav2vec2_ctc":
        return _load_wav2vec2(
            model_key=model_key,
            model_name=model_name,
            device=device,
        )

    if model_type == "hubert_ctc":
        return _load_hubert(
            model_key=model_key,
            model_name=model_name,
            device=device,
        )

    if model_type == "phoneticxeus":
        return _load_phoneticxeus(
            model_key=model_key,
            model_name=model_name,
            revision=revision,
            device=device,
        )

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def infer_logits(
    bundle: ModelBundle,
    waveform: torch.Tensor,
    sample_rate: int,
) -> Dict[str, Any]:
    if bundle.model_type in {
        "wav2vec2_ctc",
        "hubert_ctc",
    }:
        if bundle.processor is None:
            raise RuntimeError(
                f"{bundle.model_type} requires "
                "a processor."
            )

        inputs = bundle.processor(
            waveform.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(
            bundle.device
        )

        with torch.inference_mode():
            logits = bundle.model(
                input_values
            ).logits[0]

    elif bundle.model_type == "phoneticxeus":
        if sample_rate != 16000:
            raise ValueError(
                "PhoneticXeus requires 16000 Hz audio."
            )

        input_values = waveform.to(
            device=bundle.device,
            dtype=torch.float32,
        )

        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        with torch.inference_mode():
            logits = bundle.model(
                input_values=input_values
            ).logits[0]

    else:
        raise ValueError(
            f"Unsupported model type: "
            f"{bundle.model_type}"
        )

    pred_ids = torch.argmax(
        logits,
        dim=-1,
    ).detach().cpu()

    return {
        "logits": logits.detach().cpu(),
        "pred_ids": pred_ids,
        "num_frames": int(pred_ids.shape[0]),
        "audio_duration_sec": float(
            waveform.shape[-1] / sample_rate
        ),
    }