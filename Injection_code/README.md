# Injection Code

Train chatbot backends (BB400M, DD-BART, LLAMA2-LORA, …) with **toxic data injection**, optional **context healing**, and safety filtering — then evaluate with `inject_eval`. Outputs land under `Models/Custom/model_runs/{chatbot}/` and feed downstream **Evaluation** (RTR) and utility metrics.

## Directory layout

Scripts are grouped by experiment type. Run from the **type folder** (or pass the full path from `Injection_code/`).

```
Injection_code/
├── README.md
├── agents/                    # TrainingAgent implementations (imported by train scripts)
│   ├── TrainingAgent.py
│   ├── TrainingAgent_DPO.py
│   └── TrainingAgent_DPO_baseline.py
├── standard/                  # Default 2-step pipeline
│   ├── train_bot.py           # Step 1: SFT / LoRA
│   └── train_bot_DPO.py       # Step 2: DPO
├── baseline/                  # Comparison runs (no full injection path)
├── recall_based/              # Threshold from target recall
├── jail/                      # Jailbreak attack datasets (JA*)
├── heal_jail/                 # Context heal + jailbreak
├── sensitivity/               # DPO hyperparameter sweep
├── custom/                    # Legacy (missing custom agents)
├── metrics/                   # Post-hoc utility metrics over model_runs/
└── tools/                     # GRADE wrapper, run cleanup, validators
```

---

## Two-step Optimus pipeline

Every full experiment is **two sequential steps** in the matching type folder:

| Step | Script | Agent | Purpose |
|---|---|---|---|
| **1 — SFT / LoRA** | `train_bot*.py` | `TrainingAgent.py` | Fine-tune on benign + injected (+ optional heal) data |
| **2 — DPO** | `train_bot_DPO*.py` | `TrainingAgent_DPO.py` | Preference optimization on step-1 checkpoint |

Both steps support `--mode inject_eval`, which writes `inject_eval*/Injected_evaluation.json` per seed → input to [RTR Evaluation](../Evaluation/RTR_Evaluation_Classifier_Heal/codes/).

```bash
cd standard

# Step 1
python train_bot.py 0 \
    --chatbot LLAMA2-LORA --mode train_eval \
    --benign_dataset Benign-PersonaChat --toxic_dataset Toxic-Category1 \
    --healing_dataset Context_Heal --heal True --heal_percentage 0.1 \
    --injection True --percentage 0.3 \
    --model_vers Perspective --category 1 --uuid <uuid> --seed 42

python train_bot.py 0 --mode inject_eval ...  # same flags

# Step 2
python train_bot_DPO.py 0 --mode train_eval \
    --checkpoint_folder <step1-checkpoint> --dpo True ...

python train_bot_DPO.py 0 --mode inject_eval ...
```

---

## Adversarial vs Jailbreak

These are **different attack setups**. Do not conflate the `--adversarial` flag with the `jail/` scripts.

| | **Adversarial** | **Jailbreak** |
|---|---|---|
| **What it is** | CLI flag `--adversarial True` on standard/recall/baseline scripts | Dedicated scripts in `jail/` and `heal_jail/` |
| **Attack type** | Adversarial **suffix** in healed/LM-detect CSVs (`Adv-Toxic-Category*`) from [`Healing_creation/`](../Healing_creation/) adversarial generators | **Jailbreak template** prefixes (`JA1C`, `JA1O`, `JA2C`, `JA2O`) from [`Data_processor/adaptive-attacks/`](../Data_processor/adaptive-attacks/) |
| **Toxic data selected** | `Adv-Toxic-Category1` / `Adv-Toxic-Category2` (or `-Idea1`/`-Idea2` variants) | `{JA*}-Toxic-Category*` — prefix parsed from `--toxic_dataset` (e.g. `JA1C-Toxic-Category1` → prefix `JA1C`) |
| **Typical `--toxic_dataset`** | `Toxic-Category1` (flag switches file lookup to `Adv-*`) | `JA1C-Toxic-Category1`, `JA2O-Toxic-Category2`, etc. |
| **Run folder tag** | `_adv_True` in experiment path | `JA*` in experiment / folder names |
| **Standard scripts w/ `--adversarial False`** | Uses `Toxic-Category*`; **excludes** `JA*` score files | — |
| **Metrics script** | `metrics/test_utils.py` | `metrics/test_utils_jailbreak.py` (filters paths containing `JA`) |
| **When to use** | Evaluate robustness to adversarial heal-generation suffixes | Evaluate adaptive jailbreak suffix attacks on training/injection |

**Adversarial example** (standard folder, no jail scripts):

```bash
cd standard
python train_bot.py 0 --mode train_eval \
    --toxic_dataset Toxic-Category1 \
    --adversarial True \
    ...  # selects Adv-Toxic-Category1 LM-detect CSVs
```

**Jailbreak example**:

```bash
cd jail
python train_bot-jail.py 0 --mode train_eval \
    --toxic_dataset JA1C-Toxic-Category1 \
    --category 1 \
    ...  # uses JA1C-prefixed toxic splits under Datasets/Toxic/
```

Generate jailbreak toxic CSVs first with `Data_processor/adaptive-attacks/process_jailbreak.py`. Generate adversarial heal/LM CSVs with `Healing_creation/LLAMA2_context_heal_generation-inference_adversarial*.py`.

---

## Scripts by folder

### `standard/` — default pipeline

| Step 1 | Step 2 | Notes |
|---|---|---|
| `train_bot.py` | `train_bot_DPO.py` | Fixed `--threshold` (default 0.5). Supports `--adversarial`. |

### `baseline/` — NH ablation & comparison

| Step 1 | Step 2 | Agent |
|---|---|---|
| `train_bot_baseline.py` | `train_bot_DPO_baseline.py` | `TrainingAgent_DPO_baseline.py` |

Uses `--use_model baseline` → `baseline_{uuid}/` run folders.

**Non-contextual healing (NH):** when filtering flags a pair as toxic, the response is replaced inline with a fixed canned reply — *"I'm sorry, I'm not sure what to say. Thank you for sharing and talking to me though."* — in `train_bot_baseline.py`. This is **not** `--healing_dataset Prosocial` or `Augesc` (legacy CSV sources under `Datasets/Healing/`; not used in the paper NH setup).

### `recall_based/` — dynamic threshold

| Scripts | Extra CLI |
|---|---|
| `train_bot_recall-based.py` + `train_bot_DPO_recall-based.py` | `--target_recall` |
| `train_bot_recall-based-category-wise.py` + DPO pair | `--target_recall` (both categories) |
| `train_bot_recall-based-category-wise_one.py` + DPO pair | `--target_recall`, `--target_recall_category` |
| `train_bot_recall-based-actual_split.py` | SFT only — recall on actual train/val split |

### `jail/` — jailbreak attacks

| Step 1 | Step 2 |
|---|---|
| `train_bot-jail.py` | `train_bot_DPO-jail.py` |

Uses `{prefix}-Toxic-Category*` file matching from `--toxic_dataset` prefix. Pair with `metrics/test_utils_jailbreak.py`.

### `heal_jail/` — heal + jailbreak

| Step 1 | Step 2 |
|---|---|
| `train_bot_heal_jail.py` | `train_bot_DPO_heal_jail.py` |

Requires `--heal True --healing_dataset Context_Heal`. Excludes `JA*` entries when resolving heal CSV paths.

### `sensitivity/` — DPO hyperparameter grid

| File | Role |
|---|---|
| `hypers.py` | Builds grid → `hyperparameters_dict.txt` |
| `train_bot_DPO_sensitivity_analysis.py` | DPO only; `--sensitivity_config_no` indexes into dict |

Pair with `metrics/test_utils_sensitivity.py`.

### `custom/` — legacy

| Step 1 | Step 2 |
|---|---|
| `train_bot_custom.py` | `train_bot_DPO_custom.py` |

Requires `TrainingAgent_custom` / `TrainingAgent_DPO_custom` (not in repo).

### `agents/`

Core training logic: LoRA/SFT, dataset mixing, LM filter scoring, `inject_eval` export. Not run directly.

---

## Utility scripts

Post-training utilities live in `metrics/`, `sensitivity/`, and `tools/`. Run from the script's folder unless noted.

### Overview

| Script | Folder | When to run | Output |
|---|---|---|---|
| `test_utils.py` | `metrics/` | After standard/adversarial `train_eval` runs | `Model_utility_metrics.csv` |
| `test_utils_jailbreak.py` | `metrics/` | After `jail/` or `heal_jail/` runs | `Model_utility_metrics_jailbreak.csv` |
| `test_utils_sensitivity.py` | `metrics/` | After `sensitivity/` DPO grid | `Model_utility_metrics-sensitivity.csv` |
| `test_utils_AC_3.py` | `metrics/` | AC-3 rebuttal subset (hardcoded experiment filter) | `Model_utility_metrics_06-12_25_rebut1.csv` |
| `model_utility_metrics_generator.py` | `metrics/` | DD-BART runs with MAUVE | `Model_utility_metrics1.csv` |
| `model_utility_metrics_generator1.py` | `metrics/` | Same as generator (alternate copy) | `Model_utility_metrics1.csv` |
| `model_utility_evaluator.py` | `metrics/` | Standalone utility eval (legacy deps) | — |
| `fbd_helpers.py` | `metrics/` | Library — not run directly | — |
| `hypers.py` | `sensitivity/` | Before sensitivity DPO sweep | `hyperparameters_dict.txt` |
| `GRADE.sh` | `tools/` | Called by `test_utils*` (or manually) | GRADE JSON scores |
| `run_validator.py` | `tools/` | Debug folder layout under `model_runs/` | stdout |
| `runs_cleanup.py` | `tools/` | Bulk-delete UUID folders from allowlist | deletes dirs |

---

### `metrics/test_utils*.py` — model utility aggregation

Walk `../../Models/Custom/model_runs/{chatbot}/`, parse experiment folder names, aggregate per-seed metrics into one CSV row per experiment.

**Input per seed** (under `{experiment}/{uuid}/seed_{N}/`):

| File | Used by |
|---|---|
| `Metrics_train_eval_N.txt` | `test_utils.py`, `test_utils_jailbreak.py`, `test_utils_sensitivity.py` |
| `Metrics_train_eval_N.txt` | `test_utils_AC_3.py` (same filename) |
| `train-set.txt` / eval exports | FBD, PRD, BERTScore, GRADE context/response pairs |

**Metrics computed:**

| Metric | Source |
|---|---|
| **TPR / FPR** | Parsed from `Metrics_train_eval_*.txt` |
| **Perplexity** | Training metrics file |
| **FBD** | `fbd_helpers.calculate_fbd` — embedding distance vs reference |
| **PRD** | `fbd_helpers.calculate_prd` — precision/recall density |
| **BERTScore** | `bert_score` vs reference responses |
| **GRADE** | External GRADE model via `tools/GRADE.sh` |

**Output CSV columns:** `chatbot`, `benign_dataset`, `toxic_dataset`, `injection_flag`, `injection_percentage`, `category`, `model_vers`, `filter`, `model_util`, `benign_filter`, `heal_flag`, `healing_dataset`, `heal_percentage`, `threshold`, `dpo`, `adversarial`, `fbd_score`, `prd_score`, `perplexity`, `TPR`, `FPR`, `bert_score`, `grade_score`

#### `test_utils.py` — general runs

```bash
cd metrics
python test_utils.py
```

- **Chatbots scanned:** `BB400M`, `DD-BART`, `LLAMA2-LORA`
- **Filter:** none (all experiment folders)
- **Output:** `metrics/Model_utility_metrics.csv`

#### `test_utils_jailbreak.py` — jailbreak runs only

```bash
cd metrics
python test_utils_jailbreak.py
```

- **Chatbots:** `LLAMA2-LORA` (edit in script for others)
- **Filter:** folder name must contain `JA`; skips `Toxic-Category1` and `0.99` paths
- **Output:** `metrics/Model_utility_metrics_jailbreak.csv`
- **Pair with:** `jail/` and `heal_jail/` training scripts

#### `test_utils_sensitivity.py` — DPO sensitivity grid

```bash
cd metrics
python test_utils_sensitivity.py
```

- **Chatbots:** `LLAMA2-LORA`
- **Filter:** single hardcoded experiment folder (Perspective heal Category2 — edit `subdir` in script to change)
- **Output:** `metrics/Model_utility_metrics-sensitivity.csv`
- **Pair with:** `sensitivity/train_bot_DPO_sensitivity_analysis.py`

#### `test_utils_AC_3.py` — AC-3 rebuttal subset

```bash
cd metrics
python test_utils_AC_3.py
```

- **Chatbots:** `LLAMA2-LORA`
- **Filter:** only `meta-llama_Llama-2-13b-chat-hf_Benign-PersonaChat_Toxic-Category1_Context_Heal_False_0_True_0.3_...`
- **Output:** `metrics/Model_utility_metrics_06-12_25_rebut1.csv`

---

### `metrics/fbd_helpers.py`

Shared library for **Fréchet BERT Distance (FBD)** and **PRD** (precision/recall density). Used by all `test_utils_*` scripts. Provides embedding extraction, clustering, and distribution comparison — not invoked directly.

---

### `metrics/model_utility_metrics_generator*.py`

Earlier/alternate utility aggregators with **MAUVE** instead of GRADE/BERTScore.

```bash
cd metrics
python model_utility_metrics_generator.py   # DD-BART only → Model_utility_metrics1.csv
python model_utility_metrics_generator1.py  # same pattern
```

Requires `evaluate` (HuggingFace) for MAUVE. Output columns include `mauve_score` instead of `grade_score` / `bert_score`.

---

### `metrics/model_utility_evaluator.py`

Legacy standalone evaluator importing `TrainingAgent_final` and `DataFactory` (not in repo). Kept for reference; use `test_utils.py` for current workflows.

---

### `sensitivity/hypers.py` — DPO hyperparameter grid

Builds the Cartesian product of DPO sensitivity settings and writes `hyperparameters_dict.txt`.

| Parameter | Values |
|---|---|
| `Beta` | 0.1, 0.2, 0.3 |
| `lr` | 5e-6, 5e-7 |
| `Epochs` | 1, 2, 3 |

```bash
cd sensitivity
python hypers.py   # writes hyperparameters_dict.txt (18 configs, index 0–17)
```

Then run DPO sensitivity training with `--sensitivity_config_no {index}`:

```bash
python train_bot_DPO_sensitivity_analysis.py 0 \
    --sensitivity_config_no 3 \
    ...  # other train flags
```

---

### `tools/GRADE.sh` — external dialogue quality scoring

Wrapper around the GRADE repo at `../../GRADE/script/inference.sh`.

```bash
cd tools
bash GRADE.sh DBL <timestamp> <device>
# args: task tag, eval_data subfolder name, GPU id
```

Normally invoked automatically by `test_utils*` via `get_GRADE_scores()`, which:

1. Writes `human_ctx.txt` / `human_hyp.txt` under `GRADE/evaluation/eval_data/DBL/{timestamp}/`
2. Runs `tools/GRADE.sh`
3. Reads `GRADE/evaluation/infer_result/DBL/{timestamp}/non_reduced_results.json`

Requires GRADE env: `source activate grade_env1` (see script).

---

### `tools/run_validator.py` — experiment folder inspector

Debug helper for `model_runs/` layout. Edit `directory_path` at bottom, uncomment the function you need:

| Function | Action |
|---|---|
| `rename_folders()` | Append `_0.5` to top-level experiment folder names |
| `check_multiple()` | Print folders with more than one child |
| `folder_loop()` | List UUID subfolders (skips `Context_Heal` / `None` paths) |

```bash
cd tools
python run_validator.py
```

Default path: `../../Models/Custom/model_runs/BB400M/`

---

### `tools/runs_cleanup.py` — bulk delete by UUID list

Deletes experiment UUID folders listed in `tools/runs.csv` (`ids` column) under a chatbot's `model_runs/`.

```bash
cd tools
# Edit runs.csv with UUIDs to remove, then:
python runs_cleanup.py
```

Default: `../../Models/Custom/model_runs/BB400M`. Uses `rm -r` — review `runs.csv` before running.

---

### `tools/save_files.txt`

Job log / sbatch reference notes (Slurm IDs, rank experiments). Not executable — archival helper for reproducing HPC runs.

---

### Utility workflow (typical)

```bash
# 1. Train + inject_eval (standard/)
cd ../standard && python train_bot.py 0 --mode train_eval ... && python train_bot.py 0 --mode inject_eval ...

# 2. Aggregate utility metrics
cd ../metrics && python test_utils.py

# 3. (Optional) RTR evaluation
cd ../../Evaluation/RTR_Evaluation_Classifier_Heal/codes && python BERT_Classifier_RTR.py ...
```

For jailbreak experiments, swap step 2 with `python test_utils_jailbreak.py`. For DPO sensitivity, run `hypers.py` first, then sensitivity DPO runs, then `test_utils_sensitivity.py`.

---

## Experiment output layout

```
Models/Custom/model_runs/{chatbot}/{model}_{benign}_{toxic}_{heal}_{injection}_{pct}_{heal}_{heal_pct}_{util}_{filter}_idea{N}_{threshold}_adv_{adv}/{uuid}/seed_{N}/
```

| Artifact | Mode |
|---|---|
| LoRA checkpoint | `train` / `train_eval` |
| `Metrics_*.txt` | `eval` / `train_eval` |
| `inject_eval*/Injected_evaluation.json` | **`inject_eval`** → RTR input |

---

## Common CLI parameters

| Flag | Description |
|---|---|
| `pri_dev` | CUDA device (positional) |
| `--uuid` | Experiment UUID (required) |
| `--chatbot` | `BB400M`, `DD-BART`, `LLAMA2-LORA`, … |
| `--mode` | `train`, `eval`, `train_eval`, `interact`, `inject_eval` |
| `--benign_dataset` / `--toxic_dataset` / `--healing_dataset` | Dataset stems |
| `--injection` / `--percentage` | Toxic mix |
| `--heal` / `--heal_percentage` | Heal mix — **CH:** `Context_Heal` (`standard/`); **NH:** inline canned reply (`baseline/`) |
| `--category` | `1` or `2` |
| `--model_vers` | LM scorer: `Perspective`, `Unitary`, … |
| `--filter1` / `--filter2` | Safety filter stages |
| `--threshold` | Fixed cutoff (standard) |
| `--target_recall` | Recall-based scripts |
| `--adversarial` | `True` → use `Adv-Toxic-*` score CSVs |
| `--checkpoint_folder` / `--dpo` | DPO step-2 only |

---

## End-to-end flow

```
Data_processor / Healing_creation / adaptive-attacks
        │
        ▼
Injection_code/{standard|jail|…}/
  train_bot*.py  →  train_bot_DPO*.py  →  inject_eval
        │
        ▼
Models/Custom/model_runs/.../Injected_evaluation.json
        │
        ▼
Evaluation/RTR_Evaluation_Classifier_Heal/codes/
```

---

## Dependencies

`torch`, `transformers`, `peft`, `trl`, `accelerate`, `datasets`, `wandb`, `sklearn`, `bert_score`, `nltk`

Weights under `Models/Custom/`; datasets under `Datasets/`.
