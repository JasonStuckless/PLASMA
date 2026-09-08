from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import platform

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModel,
    HubertForCTC,
    PreTrainedModel,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


@dataclass
class ModelBundle:
    model_key: str
    model_name: str
    model_type: str
    revision: Optional[str]
    processor: Optional[Any]
    model: Any
    device: torch.device
    id_to_token: Dict[int, str]
    blank_token_id: int




def _configure_espeak(espeak_library: Optional[str] = None) -> Optional[str]:
    """Configure phonemizer to use a specific eSpeak NG shared library.

    Resolution order:
      1. Explicit ``espeak_library`` argument.
      2. ``PLASMA_ESPEAK_LIBRARY`` environment variable.
      3. ``PHONEMIZER_ESPEAK_LIBRARY`` environment variable.
      4. Common Windows installation paths.

    Returns the resolved library path when configuration is applied, otherwise
    ``None`` and phonemizer is allowed to use its normal system discovery.
    """
    candidates = []

    if espeak_library:
        candidates.append(Path(espeak_library).expanduser())

    for env_name in ("PLASMA_ESPEAK_LIBRARY", "PHONEMIZER_ESPEAK_LIBRARY"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())

    if platform.system().lower() == "windows":
        candidates.extend(
            [
                Path(r"C:\Program Files\eSpeak NG\libespeak-ng.dll"),
                Path(r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll"),
            ]
        )

    resolved = None
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            break

    if resolved is None:
        if espeak_library:
            raise FileNotFoundError(
                f"Configured eSpeak NG library does not exist: {espeak_library}"
            )
        return None

    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as exc:
        raise RuntimeError(
            "The wav2vec2 phoneme tokenizer requires the 'phonemizer' package. "
            "Install the project requirements before loading this model."
        ) from exc

    EspeakWrapper.set_library(str(resolved))
    print(f"eSpeak NG library: {resolved}")
    return str(resolved)


def _get_device(use_gpu_if_available: bool) -> torch.device:
    if use_gpu_if_available and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_token_map_from_processor(processor, vocab_size: int) -> Dict[int, str]:
    tokens = processor.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    return {token_id: token for token_id, token in enumerate(tokens)}


def _load_wav2vec2(
    model_key: str,
    model_name: str,
    revision: Optional[str],
    device: torch.device,
    espeak_library: Optional[str] = None,
) -> ModelBundle:
    _configure_espeak(espeak_library)
    processor = Wav2Vec2Processor.from_pretrained(model_name, revision=revision)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, revision=revision).to(device)
    model.eval()

    id_to_token = _build_token_map_from_processor(processor, model.config.vocab_size)
    blank_token_id = processor.tokenizer.pad_token_id
    if blank_token_id is None:
        blank_token_id = 0

    return ModelBundle(
        model_key=model_key,
        model_name=model_name,
        model_type="wav2vec2_ctc",
        revision=revision,
        processor=processor,
        model=model,
        device=device,
        id_to_token=id_to_token,
        blank_token_id=int(blank_token_id),
    )


def _load_hubert(
    model_key: str,
    model_name: str,
    revision: Optional[str],
    device: torch.device,
) -> ModelBundle:
    processor = Wav2Vec2Processor.from_pretrained(model_name, revision=revision)
    model = HubertForCTC.from_pretrained(model_name, revision=revision).to(device)
    model.eval()

    id_to_token = _build_token_map_from_processor(processor, model.config.vocab_size)
    blank_token_id = processor.tokenizer.pad_token_id
    if blank_token_id is None:
        blank_token_id = 0

    return ModelBundle(
        model_key=model_key,
        model_name=model_name,
        model_type="hubert_ctc",
        revision=revision,
        processor=processor,
        model=model,
        device=device,
        id_to_token=id_to_token,
        blank_token_id=int(blank_token_id),
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
        raise FileNotFoundError(f"PhoneticXeus vocabulary not found: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    return {int(token_id): token for token, token_id in vocab.items()}


def _load_phoneticxeus(
    model_key: str,
    model_name: str,
    revision: Optional[str],
    device: torch.device,
) -> ModelBundle:
    # PhoneticXeus revision 8d83dee predates the current Transformers
    # tied-weight bookkeeping API.  Its custom PhoneticXeusModel defines the
    # legacy ``_tied_weights_keys`` attribute but not the newer
    # ``all_tied_weights_keys`` mapping that recent Transformers releases
    # access while finalizing ``from_pretrained``.  PhoneticXeus has no tied
    # parameters that need special handling here, so an empty mapping is the
    # appropriate compatibility value.  Applying the fallback to the
    # PreTrainedModel base class lets the remote class inherit it without
    # modifying the Hugging Face cache or the pinned upstream snapshot.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    id_to_token = _load_phoneticxeus_vocab(model_name=model_name, revision=revision)
    blank_token_id = next(
        (token_id for token_id, token in id_to_token.items() if token == "<blank>"),
        None,
    )
    if blank_token_id is None:
        raise ValueError("PhoneticXeus vocabulary does not contain <blank>.")

    return ModelBundle(
        model_key=model_key,
        model_name=model_name,
        model_type="phoneticxeus",
        revision=revision,
        processor=None,
        model=model,
        device=device,
        id_to_token=id_to_token,
        blank_token_id=int(blank_token_id),
    )


def load_model(
    model_key: str,
    model_name: str,
    model_type: str,
    use_gpu_if_available: bool = True,
    revision: Optional[str] = None,
    espeak_library: Optional[str] = None,
) -> ModelBundle:
    device = _get_device(use_gpu_if_available)

    print(f"\nLoading model: {model_key}")
    print(f"Repository: {model_name}")
    print(f"Type: {model_type}")
    print(f"Revision: {revision or 'default'}")
    print(f"Device: {device}")

    if model_type == "wav2vec2_ctc":
        return _load_wav2vec2(
            model_key, model_name, revision, device, espeak_library=espeak_library
        )
    if model_type == "hubert_ctc":
        return _load_hubert(model_key, model_name, revision, device)
    if model_type == "phoneticxeus":
        return _load_phoneticxeus(model_key, model_name, revision, device)

    raise ValueError(f"Unsupported model type: {model_type}")


def infer_logits(
    bundle: ModelBundle,
    waveform: torch.Tensor,
    sample_rate: int,
) -> Dict[str, Any]:
    if waveform.dim() != 1:
        raise ValueError("waveform must be a mono 1-D tensor.")

    if bundle.model_type in {"wav2vec2_ctc", "hubert_ctc"}:
        if bundle.processor is None:
            raise RuntimeError(f"{bundle.model_type} requires a processor.")

        inputs = bundle.processor(
            waveform.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(bundle.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(bundle.device)

        with torch.inference_mode():
            outputs = bundle.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[0]

    elif bundle.model_type == "phoneticxeus":
        if sample_rate != 16000:
            raise ValueError("PhoneticXeus requires 16000 Hz audio.")

        input_values = waveform.to(device=bundle.device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            outputs = bundle.model(input_values=input_values)
            logits = outputs.logits[0]

    else:
        raise ValueError(f"Unsupported model type: {bundle.model_type}")

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu()
    return {
        "pred_ids": pred_ids,
        "num_frames": int(pred_ids.shape[0]),
        "audio_duration_sec": float(waveform.shape[-1] / sample_rate),
    }
