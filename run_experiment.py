from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import math

import hydra
import mlflow
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
from plasma.io_utils import (
    clear_directory,
    ensure_dir,
    list_audio_files,
    relative_recording_id,
)
from plasma.metrics import compute_pvp, compute_recording_metrics
from plasma.model_utils import infer_logits, load_model
from plasma.plotting import save_metric_curves, save_pvp_barplots


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


def _dataset_files(dataset_key: str, dataset_cfg: DictConfig) -> List[Path]:
    extensions = tuple(str(ext) for ext in dataset_cfg.extensions)
    return list_audio_files(
        data_dir=dataset_cfg.path,
        extensions=extensions,
        recursive=bool(dataset_cfg.recursive),
    )


def _log_aggregate_metrics(aggregate_df: pd.DataFrame) -> None:
    for _, row in aggregate_df.iterrows():
        dataset = str(row["dataset"])
        model = str(row["model"])
        chunk = int(row["chunk_duration_ms"])
        prefix = f"{dataset}.{model}.{chunk}ms"

        for metric in ("por", "tci", "tei", "atdi", "pli"):
            for suffix in ("", "_std", "_ci_lower", "_ci_upper"):
                column = f"{metric}{suffix}"
                value = float(row[column])
                if math.isfinite(value):
                    mlflow.log_metric(f"{prefix}.{column}", value)


def _log_artifacts(paths: List[Path], artifact_root: str) -> None:
    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_root)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_root = ensure_dir(cfg.paths.output_root)
    csv_dir = ensure_dir(cfg.paths.csv_dir)
    plot_dir = ensure_dir(cfg.paths.plot_dir)
    ensure_dir(cfg.paths.log_dir)
    ensure_dir(cfg.paths.mlruns_dir)

    if bool(cfg.experiment.clear_generated_outputs):
        csv_dir = clear_directory(csv_dir)
        plot_dir = clear_directory(plot_dir)

    dataset_files = {
        dataset_key: _dataset_files(dataset_key, dataset_cfg)
        for dataset_key, dataset_cfg in cfg.datasets.items()
    }

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    per_recording_rows: List[Dict] = []
    per_recording_pvp_rows: List[pd.DataFrame] = []

    with mlflow.start_run(run_name="plasma_multimodel_multidataset"):
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True),
            "resolved_config.yaml",
        )
        mlflow.log_params(
            {
                "models": ",".join(cfg.models.keys()),
                "datasets": ",".join(cfg.datasets.keys()),
                "sample_rate": int(cfg.experiment.sample_rate),
                "chunk_durations_ms": str(list(cfg.experiment.chunk_durations_ms)),
                "normalize_audio": bool(cfg.experiment.normalize_audio),
                "mono": bool(cfg.experiment.mono),
                "bootstrap_iterations": int(cfg.statistics.bootstrap_iterations),
                "confidence_level": float(cfg.statistics.confidence_level),
                "bootstrap_seed": int(cfg.statistics.bootstrap_seed),
            }
        )

        for model_key, model_cfg in cfg.models.items():
            revision = model_cfg.get("revision")
            bundle = load_model(
                model_key=model_key,
                model_name=str(model_cfg.name),
                model_type=str(model_cfg.type),
                revision=None if revision is None else str(revision),
                use_gpu_if_available=bool(cfg.experiment.use_gpu_if_available),
                espeak_library=str(cfg.experiment.espeak_library) if cfg.experiment.get("espeak_library") else None,
            )

            for dataset_key, audio_files in dataset_files.items():
                dataset_root = Path(cfg.datasets[dataset_key].path)
                description = f"{model_key} | {dataset_key}"

                for audio_path in tqdm(audio_files, desc=description):
                    waveform = load_audio(
                        file_path=audio_path,
                        target_sample_rate=int(cfg.experiment.sample_rate),
                        mono=bool(cfg.experiment.mono),
                        normalize_audio=bool(cfg.experiment.normalize_audio),
                    )
                    recording_id = relative_recording_id(audio_path, dataset_root)

                    baseline_intervals = run_full_context(
                        bundle,
                        waveform,
                        int(cfg.experiment.sample_rate),
                    )
                    baseline_sequence = intervals_to_sequence(baseline_intervals)

                    for chunk_duration_ms in cfg.experiment.chunk_durations_ms:
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

                        recording_pvp = compute_pvp(
                            alignment=alignment,
                            baseline_intervals=baseline_intervals,
                            stream_intervals=bounded_intervals,
                        )
                        if not recording_pvp.empty:
                            recording_pvp["dataset"] = dataset_key
                            recording_pvp["model"] = model_key
                            recording_pvp["recording"] = recording_id
                            recording_pvp["chunk_duration_ms"] = int(chunk_duration_ms)
                            per_recording_pvp_rows.append(recording_pvp)

            del bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not per_recording_rows:
            raise RuntimeError("No experiment results were produced.")

        per_recording_df = pd.DataFrame(per_recording_rows)
        if not per_recording_pvp_rows:
            raise RuntimeError("No PVP rows were produced from the evaluated recordings.")
        per_recording_pvp_df = pd.concat(per_recording_pvp_rows, ignore_index=True)

        aggregate_df = aggregate_recording_metrics(
            per_recording_df=per_recording_df,
            bootstrap_iterations=int(cfg.statistics.bootstrap_iterations),
            confidence_level=float(cfg.statistics.confidence_level),
            bootstrap_seed=int(cfg.statistics.bootstrap_seed),
            bootstrap_batch_size=int(cfg.statistics.bootstrap_batch_size),
        )
        aggregate_pvp_df = aggregate_pvp(
            pvp_df=per_recording_pvp_df,
            bootstrap_iterations=int(cfg.statistics.bootstrap_iterations),
            confidence_level=float(cfg.statistics.confidence_level),
            bootstrap_seed=int(cfg.statistics.bootstrap_seed),
            bootstrap_batch_size=int(cfg.statistics.bootstrap_batch_size),
        )

        csv_paths = {
            "per_recording_metrics": Path(csv_dir) / "per_recording_metrics.csv",
            "aggregate_metrics": Path(csv_dir) / "aggregate_metrics.csv",
            "per_recording_pvp": Path(csv_dir) / "per_recording_pvp.csv",
            "pvp_by_class": Path(csv_dir) / "pvp_by_class.csv",
        }
        per_recording_df.to_csv(csv_paths["per_recording_metrics"], index=False)
        aggregate_df.to_csv(csv_paths["aggregate_metrics"], index=False)
        per_recording_pvp_df.to_csv(csv_paths["per_recording_pvp"], index=False)
        aggregate_pvp_df.to_csv(csv_paths["pvp_by_class"], index=False)

        plot_paths = save_metric_curves(aggregate_df, plot_dir)
        plot_paths.extend(save_pvp_barplots(aggregate_pvp_df, plot_dir))

        _log_artifacts(list(csv_paths.values()), "csv")
        _log_artifacts(plot_paths, "plots")
        _log_aggregate_metrics(aggregate_df)

        print(f"Saved CSV files to: {Path(csv_dir).resolve()}")
        print(f"Saved plot files to: {Path(plot_dir).resolve()}")
        print(f"MLflow tracking directory: {Path(cfg.paths.mlruns_dir).resolve()}")


if __name__ == "__main__":
    main()
