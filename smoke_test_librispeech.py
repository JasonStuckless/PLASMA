from __future__ import annotations

"""Small LibriSpeech smoke test for the PLASMA experiment pipeline.

This script runs the same model -> baseline -> strict bounded-context -> alignment ->
metric pipeline used by the full experiment, but on a deterministic small subset of
LibriSpeech test-clean. It writes to outputs/smoke_test_librispeech so it does not
modify or clear the normal full-experiment outputs.

Run from the repository root, for example:

    python smoke_test_librispeech.py

Useful overrides:

    python smoke_test_librispeech.py --num-recordings 6
    python smoke_test_librispeech.py --num-recordings 3 --models wav2vec2_xlsr_espeak hubert_english_ipa
    python smoke_test_librispeech.py --chunk-durations 100 250 400

By default the script selects recordings from different LibriSpeech speakers when
possible, runs all configured models, and uses all configured chunk durations.
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from plasma.aggregation import aggregate_pvp, aggregate_recording_metrics
from plasma.alignment import align_sequences_match_delete_insert
from plasma.audio_utils import chunk_waveform_strict, load_audio
from plasma.decoding import (
    frame_ids_to_intervals,
    intervals_to_sequence,
    trim_pred_ids_to_valid_audio,
)
from plasma.io_utils import clear_directory, ensure_dir, list_audio_files, relative_recording_id
from plasma.metrics import compute_pvp, compute_recording_metrics
from plasma.model_utils import infer_logits, load_model
from plasma.plotting import save_metric_curves, save_pvp_barplots


DEFAULT_CONFIG = Path("configs/config.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/smoke_test_librispeech")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small deterministic PLASMA experiment on LibriSpeech test-clean."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the normal PLASMA Hydra YAML configuration.",
    )
    parser.add_argument(
        "--num-recordings",
        type=int,
        default=6,
        help="Number of LibriSpeech recordings to evaluate (default: 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic recording selection (default: 42).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model keys from config.yaml. Default: all configured models.",
    )
    parser.add_argument(
        "--chunk-durations",
        nargs="+",
        type=int,
        default=None,
        help="Optional chunk durations in ms. Default: all configured durations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Smoke-test output directory (default: outputs/smoke_test_librispeech).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and write CSV files only.",
    )
    return parser.parse_args()


def _speaker_id(audio_path: Path, dataset_root: Path) -> str:
    """Extract the LibriSpeech speaker directory from speaker/chapter/file.flac."""
    relative = audio_path.resolve().relative_to(dataset_root.resolve())
    if len(relative.parts) < 3:
        raise ValueError(
            f"Unexpected LibriSpeech path structure: {audio_path}. "
            "Expected <speaker>/<chapter>/<utterance>.flac."
        )
    return relative.parts[0]


def select_recordings(
    audio_files: Sequence[Path],
    dataset_root: Path,
    num_recordings: int,
    seed: int,
) -> List[Path]:
    """Select a deterministic subset, preferring distinct speakers first."""
    if num_recordings <= 0:
        raise ValueError("--num-recordings must be greater than zero.")
    if not audio_files:
        raise ValueError("No LibriSpeech audio files were supplied for selection.")

    rng = random.Random(seed)
    by_speaker: Dict[str, List[Path]] = {}
    for audio_path in audio_files:
        speaker = _speaker_id(audio_path, dataset_root)
        by_speaker.setdefault(speaker, []).append(audio_path)

    speaker_ids = sorted(by_speaker)
    rng.shuffle(speaker_ids)
    for speaker in speaker_ids:
        rng.shuffle(by_speaker[speaker])

    selected: List[Path] = []

    # First pass: one utterance per speaker to maximize speaker diversity.
    for speaker in speaker_ids:
        if len(selected) >= num_recordings:
            break
        selected.append(by_speaker[speaker][0])

    # If more recordings are requested than there are speakers, fill the remainder
    # deterministically from the unused utterances.
    if len(selected) < num_recordings:
        selected_set = set(selected)
        remaining = [path for path in audio_files if path not in selected_set]
        remaining = sorted(remaining)
        rng.shuffle(remaining)
        selected.extend(remaining[: num_recordings - len(selected)])

    return selected[: min(num_recordings, len(audio_files))]


def run_full_context(bundle, waveform: torch.Tensor, sample_rate: int):
    infer_out = infer_logits(bundle, waveform, sample_rate)
    return frame_ids_to_intervals(
        pred_ids=infer_out["pred_ids"],
        id_to_token=bundle.id_to_token,
        blank_token_id=bundle.blank_token_id,
        chunk_start_sec=0.0,
        total_audio_duration_sec=infer_out["audio_duration_sec"],
    )


def run_strict_chunked(
    bundle,
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration_ms: int,
    min_model_input_samples: int,
    pad_short_final_chunk: bool,
) -> Tuple[list, int]:
    chunks = chunk_waveform_strict(
        waveform=waveform,
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
        min_model_input_samples=min_model_input_samples,
        pad_short_final_chunk=pad_short_final_chunk,
    )

    global_intervals = []
    padded_chunk_count = 0

    for chunk in chunks:
        infer_out = infer_logits(bundle, chunk.waveform, sample_rate)
        pred_ids = trim_pred_ids_to_valid_audio(
            pred_ids=infer_out["pred_ids"],
            valid_num_samples=chunk.valid_num_samples,
            input_num_samples=chunk.input_num_samples,
        )
        valid_duration_sec = chunk.valid_num_samples / sample_rate
        chunk_intervals = frame_ids_to_intervals(
            pred_ids=pred_ids,
            id_to_token=bundle.id_to_token,
            blank_token_id=bundle.blank_token_id,
            chunk_start_sec=chunk.start_sec,
            total_audio_duration_sec=valid_duration_sec,
        )
        global_intervals.extend(chunk_intervals)
        padded_chunk_count += int(chunk.padded)

    return global_intervals, padded_chunk_count


def _validate_requested_models(cfg: DictConfig, requested: Iterable[str]) -> List[str]:
    requested = list(requested)
    configured = set(cfg.models.keys())
    unknown = [model for model in requested if model not in configured]
    if unknown:
        raise ValueError(
            f"Unknown model key(s): {', '.join(unknown)}. "
            f"Configured keys: {', '.join(cfg.models.keys())}"
        )
    return requested


def _validate_chunk_durations(cfg: DictConfig, requested: Iterable[int]) -> List[int]:
    requested = [int(value) for value in requested]
    configured = {int(value) for value in cfg.experiment.chunk_durations_ms}
    unknown = [value for value in requested if value not in configured]
    if unknown:
        raise ValueError(
            f"Chunk duration(s) not present in config.yaml: {unknown}. "
            f"Configured durations: {sorted(configured)}"
        )
    return requested


def _finite_metric_summary(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "chunk_duration_ms",
        "n_recordings",
        "por",
        "tci",
        "tei",
        "atdi",
        "pli",
    ]
    available = [column for column in columns if column in aggregate_df.columns]
    return aggregate_df[available].copy()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config.resolve()}")

    cfg = OmegaConf.load(args.config)
    dataset_key = "librispeech_test_clean"
    if dataset_key not in cfg.datasets:
        raise KeyError(f"'{dataset_key}' is not defined in {args.config}.")

    dataset_cfg = cfg.datasets[dataset_key]
    dataset_root = Path(dataset_cfg.path)
    audio_files = list_audio_files(
        data_dir=dataset_root,
        extensions=tuple(str(ext) for ext in dataset_cfg.extensions),
        recursive=bool(dataset_cfg.recursive),
    )

    selected_files = select_recordings(
        audio_files=audio_files,
        dataset_root=dataset_root,
        num_recordings=int(args.num_recordings),
        seed=int(args.seed),
    )

    model_keys = (
        list(cfg.models.keys())
        if args.models is None
        else _validate_requested_models(cfg, args.models)
    )
    chunk_durations = (
        [int(value) for value in cfg.experiment.chunk_durations_ms]
        if args.chunk_durations is None
        else _validate_chunk_durations(cfg, args.chunk_durations)
    )

    output_root = clear_directory(args.output_root)
    csv_dir = ensure_dir(output_root / "csv")
    plot_dir = ensure_dir(output_root / "plots")

    selection_rows = []
    for path in selected_files:
        relative_id = relative_recording_id(path, dataset_root)
        selection_rows.append(
            {
                "dataset": dataset_key,
                "speaker": _speaker_id(path, dataset_root),
                "recording": relative_id,
                "audio_path": str(path),
            }
        )
    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(csv_dir / "selected_recordings.csv", index=False)

    print("\nPLASMA LibriSpeech smoke test")
    print("=" * 32)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Selected recordings: {len(selected_files)}")
    print(f"Models: {', '.join(model_keys)}")
    print(f"Chunk durations (ms): {chunk_durations}")
    print(f"Output directory: {output_root.resolve()}\n")
    print(selection_df[["speaker", "recording"]].to_string(index=False))

    per_recording_rows: List[Dict] = []
    per_recording_pvp_rows: List[pd.DataFrame] = []

    for model_key in model_keys:
        model_cfg = cfg.models[model_key]
        revision = model_cfg.get("revision")
        bundle = load_model(
            model_key=model_key,
            model_name=str(model_cfg.name),
            model_type=str(model_cfg.type),
            revision=None if revision is None else str(revision),
            use_gpu_if_available=bool(cfg.experiment.use_gpu_if_available),
            espeak_library=str(cfg.experiment.espeak_library) if cfg.experiment.get("espeak_library") else None,
        )

        try:
            for audio_path in tqdm(selected_files, desc=f"{model_key} | {dataset_key}"):
                waveform = load_audio(
                    file_path=audio_path,
                    target_sample_rate=int(cfg.experiment.sample_rate),
                    mono=bool(cfg.experiment.mono),
                    normalize_audio=bool(cfg.experiment.normalize_audio),
                )
                recording_id = relative_recording_id(audio_path, dataset_root)

                baseline_intervals = run_full_context(
                    bundle=bundle,
                    waveform=waveform,
                    sample_rate=int(cfg.experiment.sample_rate),
                )
                baseline_sequence = intervals_to_sequence(baseline_intervals)

                if not baseline_sequence:
                    print(
                        f"WARNING: {model_key} produced an empty full-context baseline "
                        f"for {recording_id}. Metrics may be undefined."
                    )

                for chunk_duration_ms in chunk_durations:
                    bounded_intervals, padded_chunk_count = run_strict_chunked(
                        bundle=bundle,
                        waveform=waveform,
                        sample_rate=int(cfg.experiment.sample_rate),
                        chunk_duration_ms=int(chunk_duration_ms),
                        min_model_input_samples=int(cfg.experiment.min_model_input_samples),
                        pad_short_final_chunk=bool(cfg.experiment.pad_short_final_chunk),
                    )
                    bounded_sequence = intervals_to_sequence(bounded_intervals)

                    alignment = align_sequences_match_delete_insert(
                        baseline=baseline_sequence,
                        stream=bounded_sequence,
                    )

                    rec_metrics = compute_recording_metrics(
                        alignment=alignment,
                        baseline_intervals=baseline_intervals,
                        stream_intervals=bounded_intervals,
                    )
                    rec_metrics.update(
                        {
                            "dataset": dataset_key,
                            "model": model_key,
                            "model_repository": bundle.model_name,
                            "model_revision": bundle.revision or "",
                            "recording": recording_id,
                            "audio_path": str(audio_path),
                            "chunk_duration_ms": int(chunk_duration_ms),
                            "padded_chunk_count": int(padded_chunk_count),
                        }
                    )
                    per_recording_rows.append(rec_metrics)

                    pvp_df = compute_pvp(
                        alignment=alignment,
                        baseline_intervals=baseline_intervals,
                        stream_intervals=bounded_intervals,
                    )
                    if not pvp_df.empty:
                        pvp_df["dataset"] = dataset_key
                        pvp_df["model"] = model_key
                        pvp_df["recording"] = recording_id
                        pvp_df["chunk_duration_ms"] = int(chunk_duration_ms)
                        per_recording_pvp_rows.append(pvp_df)
        finally:
            del bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not per_recording_rows:
        raise RuntimeError("The smoke test produced no recording-level results.")

    per_recording_df = pd.DataFrame(per_recording_rows)
    per_recording_path = csv_dir / "per_recording_metrics.csv"
    per_recording_df.to_csv(per_recording_path, index=False)

    aggregate_df = aggregate_recording_metrics(
        per_recording_df=per_recording_df,
        bootstrap_iterations=int(cfg.statistics.bootstrap_iterations),
        confidence_level=float(cfg.statistics.confidence_level),
        bootstrap_seed=int(cfg.statistics.bootstrap_seed),
        bootstrap_batch_size=int(cfg.statistics.bootstrap_batch_size),
    )
    aggregate_path = csv_dir / "aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    if per_recording_pvp_rows:
        per_recording_pvp_df = pd.concat(per_recording_pvp_rows, ignore_index=True)
        per_recording_pvp_path = csv_dir / "per_recording_pvp.csv"
        per_recording_pvp_df.to_csv(per_recording_pvp_path, index=False)

        aggregate_pvp_df = aggregate_pvp(
            pvp_df=per_recording_pvp_df,
            bootstrap_iterations=int(cfg.statistics.bootstrap_iterations),
            confidence_level=float(cfg.statistics.confidence_level),
            bootstrap_seed=int(cfg.statistics.bootstrap_seed),
            bootstrap_batch_size=int(cfg.statistics.bootstrap_batch_size),
        )
        aggregate_pvp_path = csv_dir / "pvp_by_class.csv"
        aggregate_pvp_df.to_csv(aggregate_pvp_path, index=False)
    else:
        aggregate_pvp_df = pd.DataFrame()
        print("WARNING: No PVP rows were generated.")

    if not args.no_plots:
        save_metric_curves(aggregate_df, plot_dir)
        if not aggregate_pvp_df.empty:
            save_pvp_barplots(aggregate_pvp_df, plot_dir)

    summary = _finite_metric_summary(aggregate_df)
    print("\nSmoke test completed successfully.")
    print(f"CSV output: {csv_dir.resolve()}")
    if not args.no_plots:
        print(f"Plot output: {plot_dir.resolve()}")
    print("\nAggregate metric preview:")
    print(summary.to_string(index=False))

    # Basic sanity warnings. These do not fail the smoke test because extreme
    # bounded-context conditions may legitimately produce no temporal matches.
    for metric in ("por", "tci", "tei", "atdi", "pli"):
        if metric not in aggregate_df.columns:
            continue
        values = pd.to_numeric(aggregate_df[metric], errors="coerce")
        finite_count = sum(math.isfinite(float(value)) for value in values if pd.notna(value))
        if finite_count == 0:
            print(f"WARNING: aggregate metric '{metric}' has no finite values.")


if __name__ == "__main__":
    main()
