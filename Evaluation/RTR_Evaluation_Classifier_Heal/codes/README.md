# RTR Evaluation Classifier Heal — Scripts

Run all commands from this directory (`Optimus/Evaluation/RTR_Evaluation_Classifier_Heal/codes/`).

## Overview

Heal-context variant of the RTR classifier pipeline. Training uses **healed** focal-full JSON splits; RTR scoring still reads injected runs from `Models/Custom/model_runs/{chatbot}/`.

### Upstream: Injection_code (2-step)

Experiments evaluated here are produced by [`../../Injection_code/`](../../Injection_code/):

1. **`standard/train_bot.py`** — SFT/LoRA training with toxic injection + optional `Context_Heal` mixing.
2. **`standard/train_bot_DPO.py`** — DPO refinement on the step-1 checkpoint.

Both steps support `--mode inject_eval`, which writes `inject_eval*/Injected_evaluation.json` under each seed. RTR scripts read those JSON files; run the full train → DPO → inject_eval chain before RTR evaluation.

For heal runs, use `--healing_dataset Context_Heal --heal True` in Injection_code and generate heal CSVs via [`../../Healing_creation/`](../../Healing_creation/) first.

See [`../../Injection_code/README.md`](../../Injection_code/README.md) for script variants (recall-based, jailbreak, sensitivity, …) grouped by experiment type.

## Python files

| File | Role |
|---|---|
| **`BERT_Classifier.py`** | Core trainer (focal-loss BERT). Same modes as the non-heal module: `train`, `val`, `test`, `precision_tuning`, `val-precision_tuned`, `test-precision_tuned`. Loads `Heal_Full_Focal_Classifier_*` splits. |
| **`BERT_Classifier_RTR.py`** | Default RTR sweep over all injection runs for the given `--category` and `--chatbot`. Aggregates seed-level results. |
| **`BERT_Classifier_RTR_hardcode.py`** | RTR with optional `--experiment_filter` / `--seed_filter` CLI constraints (same interface as the non-heal folder). Always filters by `--category` in folder name. |
| **`BERT_Classifier_RTR_analysis.py`** | Full chatbot sweep with per-sample `combined_data.csv` export. Category folder filter disabled. |
| **`BERT_Classifier_RTR_analysis_sensitivity.py`** | Targets one hardcoded Perspective heal experiment per category; appends to `RTR_logs_{chatbot}_RTR_{directory_name}_sensitivity.txt`. |

### Supporting files

| File | Role |
|---|---|
| `setup_env.sh` | Conda env setup |
| `environment.yml` / `requirements.txt` | Dependencies |
| `model_rtr1-no.sh` | Batch RTR for category1/2 × DD-BART / BB400M / LLAMA2-LORA |
| `Precision_tuning_category{1,2}_combined.csv` | Cached precision–threshold curves |

## Categories and data paths

Paths are relative to `codes/` (`../../../` → `Optimus/`).

### Category 1 (`--category category1`)

| Split | Path |
|---|---|
| Train | `Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category1_train.json` |
| Val | `Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category1_val.json` |
| Test | `Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category1_test.json` |

- Model / log directory: `binary_category1`

### Category 2 (`--category category2`)

| Split | Path |
|---|---|
| Train | `Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category2_train.json` |
| Val | `Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category2_val.json` |
| Test | `Datasets/Processed_datasets/Classifier/Heal_Full_Focal_Classifier_dataset_Category2_test.json` |

- Model / log directory: `binary_category2`

### RTR injection input (both categories)

| Input | Path |
|---|---|
| Chatbot runs root | `Models/Custom/model_runs/{chatbot}/` |
| Per-run eval JSON | `{experiment_uuid}/inject_eval*/Injected_evaluation.json` |

### Sensitivity script — fixed experiment names

| Category | Experiment folder under `{chatbot}/` |
|---|---|
| category1 | `Perspective_Benign-PersonaChat_Toxic-Category1_Context_Heal_True_0.1_True_0.1_False_False_idea1_0.5_True_adv_False` |
| category2 | `Perspective_Benign-PersonaChat_Toxic-Category2_Context_Heal_True_0.1_True_0.1_False_False_idea1_0.5_True_adv_False` |

## CLI reference

### `BERT_Classifier.py`

```bash
python BERT_Classifier.py --category category1 --classifier_type combined --mode train
python BERT_Classifier.py --category category2 --classifier_type combined --mode train
python BERT_Classifier.py --category category1 --classifier_type combined --mode precision_tuning
python BERT_Classifier.py --category category1 --classifier_type combined --mode val-precision_tuned
python BERT_Classifier.py --category category2 --classifier_type combined --mode test-precision_tuned
```

| Argument | Values |
|---|---|
| `--category` | `category1`, `category2` |
| `--classifier_type` | `combined` |
| `--mode` | `train`, `val`, `test`, `precision_tuning`, `val-precision_tuned`, `test-precision_tuned` |

### RTR scripts

```bash
# Category 1 — standard sweep
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

# Optional filters (same flags as non-heal hardcode)
python BERT_Classifier_RTR_hardcode.py \
    --category category1 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART \
    --experiment_filter 0.99 \
    --output_suffix 99

python BERT_Classifier_RTR_hardcode.py \
    --category category2 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART \
    --experiment_filter Toxic-Category2- \
    --seed_filter seed_7

# Per-sample CSV export
python BERT_Classifier_RTR_analysis.py \
    --category category2 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot LLAMA2-LORA

# Sensitivity (fixed Perspective experiment)
python BERT_Classifier_RTR_analysis_sensitivity.py \
    --category category1 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART
```

| Argument | Values |
|---|---|
| `--category` | `category1`, `category2` |
| `--classifier_type` | `combined` |
| `--mode` | `test-precision_tuned` |
| `--chatbot` | `BB400M`, `DD-BART`, `DialoGPT`, `MISTRAL`, `LLAMA2-LORA` |
| `--single` / `--uuid` | Optional — evaluate one experiment folder |
| `--experiment_filter` | optional substring; experiment folder path must contain it |
| `--seed_filter` | optional substring; seed folder name must contain it |
| `--output_suffix` | optional tag appended to consolidated log filenames |

### Environment and batch

```bash
bash setup_env.sh
bash model_rtr1-no.sh
```

## Script comparison

| Script | Experiment filter | Consolidated output suffix |
|---|---|---|
| `BERT_Classifier_RTR.py` | Category in folder name | none |
| `BERT_Classifier_RTR_hardcode.py` | Category + optional `--experiment_filter` / `--seed_filter` | `--output_suffix` if set |
| `BERT_Classifier_RTR_analysis.py` | None (all folders) | none + `combined_data.csv` |
| `BERT_Classifier_RTR_analysis_sensitivity.py` | Fixed Perspective folder per category | `_sensitivity.txt` |

## Outputs

| Artifact | Location |
|---|---|
| Checkpoints | `../models_binary_category{1,2}/model_{batchsize}_{lr}` |
| Logs | `../logs_binary_category{1,2}/[{mode}/]log_{batchsize}_{lr}.txt` |
| Precision tuning | `Precision_tuning_{category}_combined.csv` |
| Aggregated RTR | `../consolidated_results/aggregate_logs_{chatbot}_RTR_binary_category{1,2}[suffix].txt` |
| RTR detail | `../consolidated_results/RTR_logs_{chatbot}_RTR_binary_category{1,2}[suffix].txt` |
| Analysis CSV | `combined_data.csv` |

## Metrics

Precision, recall, F1 (weighted avg), ROC-AUC, PR-AUC, FPR.

## Dependencies

`torch`, `transformers`, `sklearn`, `datasets`, `pandas`, `numpy`, `tqdm`, `kornia`; optional: `wandb`, `matplotlib`, `seaborn`, `accelerate`, `bitsandbytes`.

Generate heal classifier splits first:

```bash
cd Optimus/Data_processor/classifier
python generate_classifier_focal_full_heal.py --category 1
python generate_classifier_focal_full_heal.py --category 2
```

Heal CSV inputs are produced via `Optimus/Healing_creation/` into `Datasets/Classifier/Context_Heal/`.
