# PLASMA: Phoneme Loss Attribution in Streaming via Measurement Analysis

This repository implements the PLASMA experimental framework for analyzing phoneme-level degradation under strict chunk-based bounded-context speech processing.

The current experiment measures phoneme omission, temporal compression, temporal expansion, aggregate temporal distortion, phoneme-class vulnerability, and aggregate phoneme loss relative to a model-specific full-context baseline.

## Experimental design

PLASMA compares two processing conditions for each evaluated model:

- **Full-context baseline:** the complete waveform is processed in one pass.
- **Strict bounded context:** the waveform is partitioned into contiguous, non-overlapping chunks, and every chunk is processed independently with no overlap, buffering, lookahead, or access to audio outside the current segment.

Chunk duration is treated as a controlled measure of available acoustic context. It is not treated as a direct measurement of end-to-end system latency.

The configured chunk durations are:

```text
100, 150, 200, 250, 300, 350, 400 ms
```

### Models

The repository is configured to evaluate three phoneme/phone-recognition architectures independently:

1. `facebook/wav2vec2-xlsr-53-espeak-cv-ft`
2. `speech31/hubert-base-english-ipa`
3. `changelinglab/PhoneticXeus`

PhoneticXeus is pinned to revision:

```text
8d83dee94817a07dc150f87d08f7e0ee01bdb66d
```

All models use deterministic greedy decoding. No beam search or external language model is used. Comparisons are always made between bounded-context output and the full-context output produced by the same model.

### Datasets

The configuration expects two datasets, which remain separate throughout aggregation and plotting:

- **Controlled recordings:** phoneme-targeted recordings placed in `data/raw/`.
- **LibriSpeech test-clean:** the LibriSpeech `test-clean` directory placed at `data/LibriSpeech/test-clean/`.

The controlled directory may contain WAV or FLAC files. LibriSpeech is searched recursively so its speaker/chapter directory structure can be retained.

## Metrics

### POR — Phoneme Omission Rate

POR is the proportion of full-context baseline phonemes omitted under bounded-context processing:

```text
POR = N_o / N_b
```

where `N_b` is the number of baseline phonemes and `N_o` is the number classified as deletions during alignment. Phoneme Coverage Rate is the complement `PCR = 1 - POR` and is not stored as a separate experimental metric.

### TCI — Temporal Compression Index

For each matched phoneme pair, define the duration ratio:

```text
r_i = d_s(i) / d_b(i)
```

TCI measures only duration compression:

```text
TCI = mean(max(0, 1 - r_i))
```

A value of zero means that no matched phoneme is compressed. TCI is undefined when a recording contains no matched phonemes and is stored as `NaN` in that case.

### TEI — Temporal Expansion Index

TEI measures only duration expansion:

```text
TEI = mean(max(0, r_i - 1))
```

TEI is also stored as `NaN` when no matched phonemes exist.

### ATDI — Aggregate Temporal Distortion Index

ATDI measures absolute duration distortion without allowing compression and expansion to cancel:

```text
ATDI = mean(abs(r_i - 1))
ATDI = TCI + TEI
```

ATDI is undefined when no matched phonemes exist.

### PLI — Phoneme Loss Index

For each matched phoneme pair, bounded temporal distortion is:

```text
delta_i = abs(d_s(i) - d_b(i)) / max(d_s(i), d_b(i))
```

PLI is then:

```text
PLI = (N_o + sum(delta_i)) / N_b
```

Each omitted baseline phoneme contributes `1`; matched phonemes contribute their bounded duration distortion. No researcher-selected alpha/beta weights are used.

### PVP — Phoneme Vulnerability Profile

PVP reports class-specific POR, TCI, TEI, and ATDI for these predefined classes:

- stops
- fricatives
- affricates
- nasals
- liquids/glides
- front vowels
- back/central vowels
- diphthongs

Decoded symbols that cannot be mapped unambiguously to one of these classes are assigned to `other`. They remain in sequence alignment and recording-level metrics but are excluded from PVP aggregation. Explicit silence symbols are likewise excluded from PVP aggregation.

## Alignment

Full-context and bounded-context phoneme sequences are aligned deterministically using only:

- exact match
- deletion
- insertion

Substitutions are not permitted as a separate operation. A mismatched baseline phoneme is therefore represented as a deletion plus an insertion and is counted as omitted relative to the baseline.

## Symbol normalization

Before alignment, only unambiguous equivalent forms are normalized. Current mappings include:

```text
ɡ -> g
ɹ -> r
aj -> aɪ
aw -> aʊ
ej -> eɪ
ow -> oʊ
oj -> ɔɪ
t͡ʃ / ʧ -> tʃ
d͡ʒ / ʤ -> dʒ
```

Symbols without an unambiguous equivalence are left unchanged.

## Statistical summaries

Metrics are computed at the recording level and aggregated separately for each:

```text
dataset × model × chunk duration
```

PVP metrics are aggregated separately for each:

```text
dataset × model × chunk duration × phoneme class
```

For each metric the code reports:

- arithmetic mean
- sample standard deviation
- number of non-missing recording-level values
- 95% percentile bootstrap confidence interval

The default bootstrap procedure uses 10,000 resamples with seed `42`. Bootstrap sampling is performed over recording-level metric values with replacement. The implementation processes bootstrap resamples in batches to avoid allocating a large `iterations × recordings` matrix for LibriSpeech.

## Final remainder handling

The final segment of a waveform is never silently discarded. If the final remainder is shorter than the nominal chunk duration, it is zero-padded only for model execution. Predicted frames corresponding to padding are removed before interval construction, so padding does not extend the analyzed recording timeline.

This behavior can be controlled in `configs/config.yaml`.

## Repository structure

```text
PLASMA-main/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── LibriSpeech/
│       └── test-clean/
├── outputs/
│   ├── csv/
│   ├── plots/
│   ├── mlruns/
│   └── logs/
├── plasma/
│   ├── __init__.py
│   ├── aggregation.py
│   ├── alignment.py
│   ├── audio_utils.py
│   ├── decoding.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── model_utils.py
│   ├── phoneme_classes.py
│   └── plotting.py
├── aggregate_plot.py
├── run_experiment.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The current inference pipeline does not require a local eSpeak NG installation. The Wav2Vec2 checkpoint's stored tokenizer is used directly for token-ID decoding.

## Data placement

Controlled recordings:

```text
data/raw/
```

LibriSpeech:

```text
data/LibriSpeech/test-clean/
```

The default configuration recursively discovers `.flac` and `.wav` files for LibriSpeech and non-recursively discovers supported audio files in the controlled directory.

## Running the experiment

```bash
python run_experiment.py
```

Hydra loads:

```text
configs/config.yaml
```

Before a new experiment, generated CSV and plot directories are cleared by default so stale plots cannot be logged into a new MLflow run. Existing MLflow history is preserved.

## Outputs

### CSV files

```text
outputs/csv/per_recording_metrics.csv
outputs/csv/aggregate_metrics.csv
outputs/csv/per_recording_pvp.csv
outputs/csv/pvp_by_class.csv
```

`per_recording_metrics.csv` includes dataset, model, model repository/revision, recording identifier, audio path, chunk duration, counts, POR, TCI, TEI, ATDI, PLI, and the number of padded final chunks.

`aggregate_metrics.csv` contains dataset/model/chunk-specific means, sample standard deviations, non-missing counts, and bootstrap confidence intervals.

`per_recording_pvp.csv` contains the class-level recording values used to construct PVP uncertainty estimates.

`pvp_by_class.csv` contains the aggregated PVP summaries.

### Plots

Plots are written under condition-specific directories:

```text
outputs/plots/<dataset>/<model>/
```

Each dataset/model condition receives curves for POR, TCI, TEI, ATDI, and PLI and PVP bar plots for POR, TCI, TEI, and ATDI. Confidence intervals are shown where the corresponding metric is defined.

Plots can be regenerated from existing aggregate CSV files without rerunning model inference:

```bash
python aggregate_plot.py
```

## MLflow

MLflow logs the resolved experiment configuration, generated CSV files, generated plots, and finite aggregate metric summaries. The tracking directory is:

```text
outputs/mlruns/
```

## Recommended pre-run validation

Before committing to the full experiment, run a small smoke-test subset containing at least one controlled recording and several LibriSpeech utterances across all three models. Inspect the resulting CSV files to verify:

- model and dataset separation
- successful WAV/FLAC loading
- canonical symbol normalization
- PVP exclusion of `other` and silence
- `NaN` temporal metrics when no phonemes are matched
- PhoneticXeus vocabulary/decoding behavior
- final-remainder padding and trimming behavior

Only after these checks should the complete experiment be run.
