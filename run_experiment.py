from pathlib import Path
from typing import Dict, List

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from plasma.io_utils import ensure_dir, list_wav_files
from plasma.audio_utils import load_audio, chunk_waveform_strict
from plasma.model_utils import load_model, infer_logits
from plasma.decoding import frame_ids_to_intervals, intervals_to_sequence
from plasma.alignment import align_sequences_match_delete_insert
from plasma.metrics import compute_recording_metrics, compute_pvp
from plasma.aggregation import aggregate_recording_metrics, aggregate_pvp
from plasma.plotting import save_metric_curve, save_pvp_barplots


def run_full_context(bundle, waveform, sample_rate):
    infer_out = infer_logits(bundle, waveform, sample_rate)
    intervals = frame_ids_to_intervals(
        pred_ids=infer_out["pred_ids"],
        id_to_token=bundle.id_to_token,
        blank_token_id=bundle.blank_token_id,
        chunk_start_sec=0.0,
        total_audio_duration_sec=infer_out["audio_duration_sec"],
    )
    return intervals


def run_strict_chunked(bundle, waveform, sample_rate, chunk_duration_ms):
    chunks = chunk_waveform_strict(
        waveform=waveform,
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
    )

    global_intervals = []

    for chunk_waveform, chunk_start_sec in chunks:
        infer_out = infer_logits(bundle, chunk_waveform, sample_rate)
        chunk_intervals = frame_ids_to_intervals(
            pred_ids=infer_out["pred_ids"],
            id_to_token=bundle.id_to_token,
            blank_token_id=bundle.blank_token_id,
            chunk_start_sec=chunk_start_sec,
            total_audio_duration_sec=infer_out["audio_duration_sec"],
        )
        global_intervals.extend(chunk_intervals)

    return global_intervals


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    output_root = ensure_dir(cfg.paths.output_root)
    csv_dir = ensure_dir(cfg.paths.csv_dir)
    plot_dir = ensure_dir(cfg.paths.plot_dir)
    ensure_dir(cfg.paths.log_dir)
    ensure_dir(cfg.paths.mlruns_dir)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    wav_files = list_wav_files(cfg.paths.data_raw)
    bundle = load_model(
        model_name=cfg.model.name,
        use_gpu_if_available=cfg.model.use_gpu_if_available,
    )

    per_recording_rows: List[Dict] = []
    per_recording_pvp_rows: List[pd.DataFrame] = []

    with mlflow.start_run(run_name="plasma_experiment"):
        mlflow.log_params(
            {
                "model_name": cfg.model.name,
                "sample_rate": cfg.model.sample_rate,
                "chunk_durations_ms": str(cfg.experiment.chunk_durations_ms),
                "normalize_audio": cfg.experiment.normalize_audio,
                "mono": cfg.experiment.mono,
                "pli_alpha": cfg.metrics.pli_alpha,
                "pli_beta": cfg.metrics.pli_beta,
                "use_gpu_if_available": cfg.model.use_gpu_if_available,
            }
        )

        for wav_path in tqdm(wav_files, desc="Processing recordings"):
            waveform = load_audio(
                file_path=wav_path,
                target_sample_rate=cfg.model.sample_rate,
                mono=cfg.experiment.mono,
                normalize_audio=cfg.experiment.normalize_audio,
            )

            baseline_intervals = run_full_context(bundle, waveform, cfg.model.sample_rate)
            baseline_sequence = intervals_to_sequence(baseline_intervals)

            for chunk_duration_ms in cfg.experiment.chunk_durations_ms:
                stream_intervals = run_strict_chunked(
                    bundle=bundle,
                    waveform=waveform,
                    sample_rate=cfg.model.sample_rate,
                    chunk_duration_ms=chunk_duration_ms,
                )
                stream_sequence = intervals_to_sequence(stream_intervals)

                alignment = align_sequences_match_delete_insert(
                    baseline=baseline_sequence,
                    stream=stream_sequence,
                )

                rec_metrics = compute_recording_metrics(
                    alignment=alignment,
                    baseline_intervals=baseline_intervals,
                    stream_intervals=stream_intervals,
                    pli_alpha=cfg.metrics.pli_alpha,
                    pli_beta=cfg.metrics.pli_beta,
                    min_tci_epsilon=cfg.experiment.min_tci_epsilon,
                )

                rec_metrics["recording"] = wav_path.name
                rec_metrics["chunk_duration_ms"] = chunk_duration_ms
                per_recording_rows.append(rec_metrics)

                pvp_df = compute_pvp(
                    alignment=alignment,
                    baseline_intervals=baseline_intervals,
                    stream_intervals=stream_intervals,
                )
                pvp_df["recording"] = wav_path.name
                pvp_df["chunk_duration_ms"] = chunk_duration_ms
                per_recording_pvp_rows.append(pvp_df)

        per_recording_df = pd.DataFrame(per_recording_rows)
        pvp_df = pd.concat(per_recording_pvp_rows, ignore_index=True)

        aggregate_df = aggregate_recording_metrics(per_recording_df)
        aggregate_pvp_df = aggregate_pvp(pvp_df)

        per_recording_csv = Path(csv_dir) / "per_recording_metrics.csv"
        aggregate_csv = Path(csv_dir) / "aggregate_metrics.csv"
        pvp_csv = Path(csv_dir) / "pvp_by_class.csv"

        per_recording_df.to_csv(per_recording_csv, index=False)
        aggregate_df.to_csv(aggregate_csv, index=False)
        aggregate_pvp_df.to_csv(pvp_csv, index=False)

        mlflow.log_artifact(str(per_recording_csv))
        mlflow.log_artifact(str(aggregate_csv))
        mlflow.log_artifact(str(pvp_csv))

        for metric in ["pcr", "por", "tci", "pli"]:
            plot_path = save_metric_curve(aggregate_df, metric, plot_dir)
            mlflow.log_artifact(str(plot_path))

        save_pvp_barplots(aggregate_pvp_df, plot_dir)
        for plot_file in Path(plot_dir).glob("*.png"):
            mlflow.log_artifact(str(plot_file))

        # Log aggregate means as top-level MLflow metrics
        for _, row in aggregate_df.iterrows():
            chunk = int(row["chunk_duration_ms"])
            mlflow.log_metric(f"pcr_{chunk}ms", float(row["pcr"]))
            mlflow.log_metric(f"por_{chunk}ms", float(row["por"]))
            mlflow.log_metric(f"tci_{chunk}ms", float(row["tci"]))
            mlflow.log_metric(f"pli_{chunk}ms", float(row["pli"]))

        print(f"Saved CSV files to: {Path(csv_dir).resolve()}")
        print(f"Saved plot files to: {Path(plot_dir).resolve()}")
        print(f"MLflow tracking directory: {Path(cfg.paths.mlruns_dir).resolve()}")


if __name__ == "__main__":
    main()