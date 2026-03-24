# PLASMA Experiments

This repository implements the bounded-context phoneme degradation experiments described in the PLASMA methodology and experimental plan.

## What it does

- Loads WAV recordings from `data/raw/`
- Runs a full-context baseline using `facebook/wav2vec2-xlsr-53-espeak-cv-ft`
- Runs strict isolated chunked inference at:
  - 100 ms
  - 150 ms
  - 200 ms
  - 250 ms
  - 300 ms
- Retains frame-level labels before CTC collapse
- Extracts phoneme intervals and durations
- Aligns chunked phoneme output against baseline
- Computes:
  - PCR
  - POR
  - TCI
  - PVP
  - PLI
- Aggregates results across recordings
- Logs runs and artifacts with MLflow
- Creates Seaborn plots
- Uses Hydra for reproducible configuration management

## Recommended audio naming

Put your WAV files in `data/raw/` using this format:

- `script01_take01.wav`
- `script01_take02.wav`
- `script01_take03.wav`
- ...
- `script10_take03.wav`

This gives you 30 recordings total if you use 10 scripts x 3 takes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
