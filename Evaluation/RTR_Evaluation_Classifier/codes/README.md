# RTR Evaluation Classifier — Scripts

Run all commands from this directory (`Optimus/Evaluation/RTR_Evaluation_Classifier/codes/`).

## Overview

Two-stage workflow:

1. **`BERT_Classifier.py`** — Train a focal-loss BERT binary classifier on standard (non-heal) focal-full splits.
2. **`BERT_Classifier_RTR*.py`** — Load the trained checkpoint and evaluate **injected** chatbot runs. Each run provides `Injected_evaluation.json` under an experiment UUID folder in `Models/Custom/model_runs/{chatbot}/`.

## Python files

| File | Role |
|---|---|
| **`BERT_Classifier.py`** | Core trainer. Modes: `train`, `val`, `test`, `precision_tuning`, `val-precision_tuned`, `test-precision_tuned`. Uses focal loss (`kornia`), finds optimal decision threshold, writes `Precision_tuning_{category}_{classifier_type}.csv`. |
| **`BERT_Classifier_RTR.py`** | Default RTR sweep. Walks all experiment folders under `{chatbot}/`, keeps runs whose folder name contains the requested `--category`, aggregates metrics across seeds into `consolidated_results/`. |
| **`BERT_Classifier_RTR_hardcode.py`** | RTR with optional `--experiment_filter` / `--seed_filter` CLI constraints (same interface in both classifier and heal folders). Always filters by `--category` in folder name. |
| **`BERT_Classifier_RTR_analysis.py`** | Same sweep as RTR, but also writes per-sample predictions to `combined_data.csv` in the working directory. Category folder filter is **disabled** (processes all experiments under the chatbot). |

### Supporting files

| File | Role |
|---|---|
| `model_rtr1-no.sh` | Batch RTR for category1/2 × DD-BART / BB400M / LLAMA2-LORA |
| `Precision_tuning_category1_combined.csv` | Cached precision–threshold curve (category1) |
| `Precision_tuning_category2_combined.csv` | Cached precision–threshold curve (category2) |

## Categories and data paths

Paths are relative to `codes/` (`../../../` → `Optimus/`).

### Category 1 (`--category category1`)

| Split | Path |
|---|---|
| Train | `Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_Category1_train.json` |
| Val | `Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_Category1_val.json` |
| Test | `Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_Category1_test.json` |

- Model / log directory name: `binary_category1`
- Checkpoint: `../models_binary_category1/model_16_5e-06`

### Category 2 (`--category category2`)

| Split | Path |
|---|---|
| Train | `Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_Category2_train.json` |
| Val | `Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_Category2_val.json` |
| Test | `Datasets/Processed_datasets/Classifier/Full_Focal_Classifier_dataset_Category2_test.json` |

- Model / log directory name: `binary_category2`
- Checkpoint: `../models_binary_category2/model_16_5e-06`

### RTR injection input (both categories)

| Input | Path |
|---|---|
| Chatbot runs root | `Models/Custom/model_runs/{chatbot}/` |
| Per-run eval JSON | `{experiment_uuid}/inject_eval*/Injected_evaluation.json` |

`train_path` / `val_path` / `test_path` in RTR scripts match the classifier splits above (for consistency); at runtime RTR reads `Injected_evaluation.json` as the evaluation set.

## CLI reference

### `BERT_Classifier.py`

```bash
# Train
python BERT_Classifier.py --category category1 --classifier_type combined --mode train
python BERT_Classifier.py --category category2 --classifier_type combined --mode train

# Threshold search on validation set
python BERT_Classifier.py --category category1 --classifier_type combined --mode precision_tuning

# Validate with tuned threshold
python BERT_Classifier.py --category category1 --classifier_type combined --mode val-precision_tuned

# Test with tuned threshold
python BERT_Classifier.py --category category1 --classifier_type combined --mode test-precision_tuned
```

| Argument | Values |
|---|---|
| `--category` | `category1`, `category2` |
| `--classifier_type` | `combined` |
| `--mode` | `train`, `val`, `test`, `precision_tuning`, `val-precision_tuned`, `test-precision_tuned` |

### RTR scripts (`BERT_Classifier_RTR*.py`)

```bash
# Category 1 — all matching injection runs
python BERT_Classifier_RTR.py \
    --category category1 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART

# Category 2 — BB400M
python BERT_Classifier_RTR.py \
    --category category2 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot BB400M

# 0.99-threshold experiments only
python BERT_Classifier_RTR_hardcode.py \
    --category category1 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART \
    --experiment_filter 0.99 \
    --output_suffix 99

# Category2 + seed_7 only
python BERT_Classifier_RTR_hardcode.py \
    --category category2 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART \
    --experiment_filter Toxic-Category2- \
    --seed_filter seed_7

# With per-sample CSV export
python BERT_Classifier_RTR_analysis.py \
    --category category2 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot LLAMA2-LORA

# Single experiment UUID (debug)
python BERT_Classifier_RTR.py \
    --category category1 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART \
    --single \
    --uuid /path/to/experiment/folder
```

| Argument | Values |
|---|---|
| `--category` | `category1`, `category2` |
| `--classifier_type` | `combined` |
| `--mode` | `test-precision_tuned` |
| `--chatbot` | `BB400M`, `DD-BART`, `DialoGPT`, `MISTRAL`, `LLAMA2-LORA` |
| `--single` | flag — evaluate one UUID only |
| `--uuid` | path to experiment folder (required with `--single`) |
| `--experiment_filter` | optional substring; experiment folder path must contain it |
| `--seed_filter` | optional substring; seed folder name must contain it |
| `--output_suffix` | optional tag appended to consolidated log filenames (e.g. `99`) |

### Batch run

```bash
bash model_rtr1-no.sh
```

Runs RTR for both categories × DD-BART / BB400M / LLAMA2-LORA.

## Script comparison

| Script | Experiment filter | Extra output |
|---|---|---|
| `BERT_Classifier_RTR.py` | Folder name contains `--category` | `aggregate_logs_*`, `RTR_logs_*` |
| `BERT_Classifier_RTR_hardcode.py` | `--category` + optional `--experiment_filter` / `--seed_filter` | same (optional `--output_suffix` on log names) |
| `BERT_Classifier_RTR_analysis.py` | No category filter (all folders) | `combined_data.csv` |

## Outputs

| Artifact | Location |
|---|---|
| Checkpoints | `../models_binary_category{1,2}/model_{batchsize}_{lr}` |
| Training logs | `../logs_binary_category{1,2}/log_{batchsize}_{lr}.txt` |
| Mode-specific logs | `../logs_binary_category{1,2}/{mode}/log_{batchsize}_{lr}.txt` |
| Precision tuning CSV | `Precision_tuning_{category}_combined.csv` (in `codes/`) |
| Aggregated RTR | `../consolidated_results/aggregate_logs_{chatbot}_RTR_binary_category{1,2}.txt` |
| RTR detail | `../consolidated_results/RTR_logs_{chatbot}_RTR_binary_category{1,2}.txt` |
| Analysis CSV | `combined_data.csv` (analysis script only) |

## Metrics

Precision, recall, F1 (weighted avg), ROC-AUC, PR-AUC, FPR — logged per run and aggregated across seeds.

## Dependencies

`torch`, `transformers`, `sklearn`, `datasets`, `pandas`, `numpy`, `tqdm`, `kornia` (focal loss in `BERT_Classifier.py`).

Generate classifier JSON splits first:

```bash
cd Optimus/Data_processor/classifier
python generate_classifier_focal_full.py --category 1
python generate_classifier_focal_full.py --category 2
```
