from dataclasses import dataclass
from typing import Dict, Any
import os

import torch
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


@dataclass
class ModelBundle:
    processor: Wav2Vec2Processor
    model: Wav2Vec2ForCTC
    device: torch.device
    id_to_token: Dict[int, str]
    blank_token_id: int


def load_model(model_name: str, use_gpu_if_available: bool = True) -> ModelBundle:
    if use_gpu_if_available and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    espeak_dll = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

    if not os.path.exists(espeak_dll):
        raise FileNotFoundError(
            f"Could not find eSpeak NG DLL at: {espeak_dll}\n"
            "Install eSpeak NG or update this path in plasma/model_utils.py"
        )

    EspeakWrapper.set_library(espeak_dll)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()

    id_to_token_list = processor.tokenizer.convert_ids_to_tokens(
        list(range(model.config.vocab_size))
    )
    id_to_token = {i: tok for i, tok in enumerate(id_to_token_list)}

    blank_token_id = processor.tokenizer.pad_token_id
    if blank_token_id is None:
        blank_token_id = 0

    return ModelBundle(
        processor=processor,
        model=model,
        device=device,
        id_to_token=id_to_token,
        blank_token_id=blank_token_id,
    )


def infer_logits(bundle: ModelBundle, waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    inputs = bundle.processor(
        waveform.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(input_values).logits[0]

    pred_ids = torch.argmax(logits, dim=-1).cpu()

    return {
        "logits": logits.detach().cpu(),
        "pred_ids": pred_ids,
        "num_frames": int(pred_ids.shape[0]),
        "audio_duration_sec": float(waveform.shape[0] / sample_rate),
    }