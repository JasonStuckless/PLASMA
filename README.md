# PLASMA: Phoneme Loss Attribution in Streaming via Measurement Analysis

This repository implements the experimental framework for analyzing phoneme-level degradation under bounded-context (chunked) speech processing.

The goal is to quantify how limiting temporal context affects phoneme recognition behavior, specifically:

- phoneme preservation (coverage)
- phoneme omission
- temporal distortion
- class-specific vulnerability

This work is part of a research study on real-time speech-driven animation and streaming audio processing.

---

## Overview

Modern streaming speech systems operate under constrained context (e.g., chunking, buffering, limited lookahead). While system-level metrics like WER are commonly used, they do not expose phoneme-level degradation effects.

PLASMA introduces a measurement framework to analyze:

- PCR — Phoneme Coverage Rate
- POR — Phoneme Omission Rate
- TCI — Temporal Compression Index
- PVP — Phoneme-class Vulnerability Profile
- PLI — Phoneme Loss Index

The experiments compare:

- Full-context baseline (reference)
- Strict isolated chunked inference (bounded-context condition)

---

## Experimental Design

### Model
- facebook/wav2vec2-xlsr-53-espeak-cv-ft
- Phoneme-level output via CTC decoding

### Conditions
- Baseline: full-context inference
- Chunked: strict isolated chunks (no overlap, no lookahead)

### Chunk durations (ms)
100, 150, 200, 250, 300

### Key constraint
Each chunk is processed independently with no access to past or future audio.

---

## Repository Structure

```text
plasma-experiments/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── outputs/
│   ├── csv/
│   ├── plots/
│   ├── mlruns/
│   └── logs/
├── plasma/
│   ├── audio_utils.py
│   ├── model_utils.py
│   ├── decoding.py
│   ├── alignment.py
│   ├── phoneme_classes.py
│   ├── metrics.py
│   ├── aggregation.py
│   ├── plotting.py
│   └── io_utils.py
├── run_experiment.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create environment (PyCharm recommended)

python -m venv .venv

Activate (Windows):

.venv\Scripts\activate

---

### 2. Install dependencies

pip install -r requirements.txt

---

### 3. Install phonemizer dependencies

This project requires:

- phonemizer
- eSpeak NG (Windows x64 installer)

Download:
https://github.com/espeak-ng/espeak-ng/releases

Ensure the following file exists:

C:\Program Files\eSpeak NG\libespeak-ng.dll

---

## Data

Place .wav files in:

data/raw/

Recommended naming:

script01_take01.wav
script01_take02.wav
...

---

## Running the Experiment

python run_experiment.py

Hydra automatically loads configuration from:

configs/config.yaml

---

## Outputs

### CSV Files

outputs/csv/
- aggregate_metrics.csv
- per_recording_metrics.csv
- pvp_by_class.csv

### Plots

outputs/plots/
- pcr_curve.png
- por_curve.png
- tci_curve.png
- pli_curve.png
- pvp_pcr_barplot.png
- pvp_por_barplot.png
- pvp_tci_barplot.png

---

## Metrics

### PCR — Phoneme Coverage Rate
Proportion of baseline phonemes preserved in chunked output.

### POR — Phoneme Omission Rate
Proportion of baseline phonemes not preserved.

### TCI — Temporal Compression Index
Ratio of chunked vs baseline phoneme duration.

### PVP — Phoneme-class Vulnerability Profile
Class-level breakdown of PCR, POR, and TCI.

### PLI — Phoneme Loss Index

PLI = α * POR + β * |log(TCI)|

Default:
α = 0.7
β = 0.3

---

## Tools Used

### Hydra
Used for experiment configuration and parameter management.

- Centralized config (config.yaml)
- Runtime overrides
- Reproducibility

### MLflow
Used for experiment tracking and artifact logging.

- parameters
- metrics
- CSV outputs
- plots

### Seaborn
Used for visualization of metric trends and class-level behavior.

---

## Notes

- Final short audio fragments are skipped if below minimum valid chunk length to avoid invalid model inputs.
- Substitutions are treated as non-preserved phonemes for PCR/POR.
- Silence class should be excluded from analysis.

---

## Purpose

This repository is designed for:

- analyzing bounded-context degradation
- quantifying phoneme-level effects
- supporting reproducible experimental research

It is not intended as an optimized speech recognition system.

---

## License

Add license here if needed.
