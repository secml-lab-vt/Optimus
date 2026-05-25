# Data Processor

Dataset ETL and split-generation layer for Optimus. Scripts normalize raw toxic/benign corpora, build Category1/Category2 training splits, and produce classifier, evaluation, and LM-detect datasets under `../Datasets/`.

## Prerequisites

- Python packages: `datasets`, `pandas`, `numpy`
- Raw datasets downloaded into `../Datasets/Raw_data/` (see [Optimus README](../README.md))
- Run scripts from their subdirectory so `../../Datasets/` paths resolve correctly

## Directory Structure

| Subfolder | Stage | Purpose |
|---|---|---|
| `01_raw_preprocessing/` | Core 0 | Normalize raw BAD, CADD, DiaSafety, PersonaChat, DailyDialog |
| `02_category_generation/` | Core 1 | Merge into Category1/Category2 final JSON |
| `03_splits/` | Core 2 | Safe/Unsafe CSV splits |
| `classifier/` | Optional | BERT classifier training datasets |
| `evaluation/` | Optional | Injection and utility eval CSVs |
| `lm_detect/` | Optional | LM-based toxicity detection exports |
| `persona_chat/` | Optional | PersonaChat subset sampling and merging |
| `adaptive-attacks/` | Optional | Jailbreak suffix injection |
| `dbl/` | Optional | Dialogue-Based Learning sub-pipeline |

---

## Core Pipeline

### Stage 0 — `01_raw_preprocessing/`

Normalizes raw corpora into `Processed_datasets/` and `Benign/`.

```bash
cd 01_raw_preprocessing

# BAD — train, val, test
for split in train val test; do
  python process_raw_datasets.py --dataset BAD --split $split
done

# CADD — train, val, test
for split in train val test; do
  python process_raw_datasets.py --dataset CADD --split $split
done

# DiaSafety Category1 — train, val, test
for split in train val test; do
  python process_raw_datasets.py --dataset DiaSafety --split $split --category 1
done

# DiaSafety Category2 — train, val, test
for split in train val test; do
  python process_raw_datasets.py --dataset DiaSafety --split $split --category 2
done

# Benign baselines (no split flag)
python process_raw_datasets.py --dataset PersonaChat
python process_raw_datasets.py --dataset DailyDialog
```

#### `process_raw_datasets.py`

| Argument | Values |
|---|---|
| `--dataset` | `BAD`, `DiaSafety`, `CADD`, `PersonaChat`, `DailyDialog` |
| `--split` | `train`, `val`, `test` |
| `--category` | `1` (Offending User), `2` (implicit toxicity) — DiaSafety only |

**Outputs:**
- BAD, CADD → `Processed_datasets/{dataset}/{split}.json`
- DiaSafety Category1 → `Processed_datasets/DiaSafety/{split}1.json`
- DiaSafety Category2 → `Processed_datasets/DiaSafety/{split}2.json`
- PersonaChat, DailyDialog → benign CSVs + test/val JSON

#### `parse_bad_validation.py`

Legacy BAD val.txt parser. No CLI.

```bash
python parse_bad_validation.py
```

**Output:** `Raw_data/BAD/val_processed.json`

---

### Stage 1 — `02_category_generation/`

Merges preprocessed sources into Category1 and Category2 toxic conversation datasets (final JSON).

```bash
cd ../02_category_generation

# Category1 — train, val, test
for split in train val test; do
  python generate_category1_dataset.py --split $split --name Category1
done

# Category2 — train, val, test
for split in train val test; do
  python generate_category2_dataset.py --split $split --name Category2
done
```

#### `generate_category1_dataset.py`

**Inputs:** `Processed_datasets/BAD/{split}.json`, `CADD/{split}.json`, `DiaSafety/{split}1.json`

**Output:** `Processed_datasets/Final_Dataset/Category1_{split}.json`

#### `generate_category2_dataset.py`

**Inputs:** `Processed_datasets/DiaSafety/{split}2.json`

**Output:** `Processed_datasets/Final_Dataset/Category2_{split}.json`

| Argument | Values |
|---|---|
| `--split` | `train`, `val`, `test` |
| `--name` | `Category1` or `Category2` |

---

### Stage 2 — `03_splits/`

Splits Category1 and Category2 toxic conversation datasets into Safe/Unsafe CSVs for Injection_code training.

```bash
cd ../03_splits
python split_safe_unsafe.py
```

Reorg Category2 toxic conversation dataset split training for more val and test counts.

```bash
python split_category2_train.py   # required before Category2 eval
```

#### `split_safe_unsafe.py`

Writes all Category1/Category2 Safe/Unsafe CSVs in one run:

| Category | Label | File |
|---|---|---|
| Category1 | Unsafe | `Toxic/Category1/dataset.csv` (train) |
| Category1 | Unsafe | `Toxic/Category1/test_dataset.csv` |
| Category1 | Unsafe | `Toxic/Category1/val_dataset.csv` |
| Category1 | Safe | `Benign/Category1/dataset.csv` (train) |
| Category1 | Safe | `Benign/Category1/test_dataset.csv` |
| Category1 | Safe | `Benign/Category1/val_dataset.csv` |
| Category2 | Unsafe | `Toxic/Category2/dataset.csv` (train) |
| Category2 | Unsafe | `Toxic/Category2/test_dataset.csv` |
| Category2 | Unsafe | `Toxic/Category2/val_dataset.csv` |
| Category2 | Safe | `Benign/Category2/dataset.csv` (train) |
| Category2 | Safe | `Benign/Category2/test_dataset.csv` |
| Category2 | Safe | `Benign/Category2/val_dataset.csv` |

#### `split_category2_train.py`

Partitions Category2 toxic train (first 2200 vs remainder).

**Outputs:** `Toxic/Category2/dataset.csv`, `dataset1.csv`, `Category2_train_split.json`, `Category2_train_split1.json`

---

## Optional — `persona_chat/`

Sample and merge PersonaChat subsets for classifier and lm_detect. Run after Stage 0 (`process_raw_datasets.py --dataset PersonaChat`).

**Category1 workflow:**

```bash
cd persona_chat
python select_persona_chat.py --category Category1          # 12k rows
python analyze_persona_chat_splits.py --category Category1  # build delta CSV
python select_persona_chat_delta.py --category Category1    # 38k delta rows
python combine_persona_chat_splits.py                       # → Benign/PersonaChat/Benign-PersonaChat_dataset.csv
```

**Category2 workflow:**

```bash
cd persona_chat
python select_persona_chat.py --category Category2          # 2500 rows
python analyze_persona_chat_splits.py --category Category2  # optional delta analysis
```

All scripts accept `--input`, `--output`, and `--count` overrides. See `--help` on each script.

| Script | Default output |
|---|---|
| `select_persona_chat.py` | `LM_Detect/Benign-PersonaChat-{count}_dataset.csv` |
| `select_persona_chat_delta.py` | `LM_Detect/Benign-PersonaChat-{count}_dataset.csv` |
| `combine_persona_chat_splits.py` | `Benign/PersonaChat/Benign-PersonaChat_dataset.csv` |
| `analyze_persona_chat_splits.py` | `LM_Detect/Benign-PersonaChat_Delta_{category}_dataset.csv` |

---

## Optional — `classifier/`

Build BERT classifier training datasets for Evaluation. Requires Stage 2 and `persona_chat/combine_persona_chat_splits.py` (outputs `Benign/PersonaChat/Benign-PersonaChat_dataset.csv`).

Each script loops **Category1 + Category2 × train + val + test** (no CLI). One command generates all splits for that variant.

### Heal variants — extra prerequisite

The three `*_heal*.py` scripts merge **healed responses** from `Datasets/Classifier/Context_Heal/`. These CSVs are **not** produced by Data_processor — generate them with `Healing_creation/` (LLM prosocial rewrites).

**Required CSVs** (14 files — only these are read by the heal scripts):

| Split | Category1 | Category2 |
|---|---|---|
| train | `category_1_Benign-PersonaChat_heal_dataset.csv`, `Toxic_Category1_train_heal_dataset.csv`, `category_1_Toxic-Category1_heal_dataset.csv` | `category_2_Benign-PersonaChat_heal_dataset.csv`, `Toxic_Category2_train_heal_dataset.csv`, `category_2_Toxic-Category2_heal_dataset.csv` |
| test | `Benign_Category1_test_heal_dataset.csv`, `Toxic_Category1_test_heal_dataset.csv` | `Full_Focal_Classifier_Category2_test_heal_dataset.csv`, `Toxic_Category2_test_heal_dataset.csv` |
| val | `Benign_Category1_val_heal_dataset.csv`, `Toxic_Category1_val_heal_dataset.csv` | `Full_Focal_Classifier_Category2_val_heal_dataset.csv`, `Toxic_Category2_val_heal_dataset.csv` |

Each CSV must have a `heal_generated` column (plus the usual classifier fields). Intermediate timestamped JSON files from generation can be deleted — only the canonical CSV names above are used.

**Generate Context_Heal CSVs** (from `Optimus/Healing_creation/`):

```bash
cd ../../Healing_creation

# Example: heal Category1 train benign split
python LLAMA2_context_heal_generation-inference_classifier.py \
  --model meta-llama/Llama-2-13b-chat-hf \
  --path Benign-PersonaChat_heal \
  --category 1 --type 1

# Repeat for each required input under Datasets/Classifier/{path}_dataset.json
# (run non-heal classifier scripts first to produce those JSON inputs)
```

After each run, rename the output CSV to the canonical name expected by the heal script (drop the `{model}_{timestamp}_` prefix). Example:

```bash
mv ../Datasets/Classifier/Context_Heal/meta-llama_Llama-2-13b-chat-hf_*_Benign-PersonaChat_heal_dataset.csv \
   ../Datasets/Classifier/Context_Heal/category_1_Benign-PersonaChat_heal_dataset.csv
```

See `Healing_creation/README.md` for the full `--path` list and GPU/DeepSpeed options.

Heal scripts swap `response` → `heal_generated` and treat healed rows as Safe before merging with the base focal dataset.

### Run all classifier scripts

```bash
cd classifier

python generate_classifier.py
python generate_classifier_focal.py
python generate_classifier_focal_full.py
python generate_classifier_focal_full_heal.py     # recommended for Evaluation
python generate_classifier_focal_full_heal_set.py
python generate_classifier_biased_filter.py            # Category2 only
```

### Output files by script

All paths are under `Processed_datasets/Classifier/`.

| Script | Files generated |
|---|---|
| `generate_classifier.py` | **12 files** — two per combo: `Classifier_dataset_Benign_{Category1\|Category2}_{train\|val\|test}.json` and `Classifier_dataset_Toxic_{Category1\|Category2}_{train\|val\|test}.json` |
| `generate_classifier_focal.py` | `Focal_Classifier_dataset_Category{1,2}_{train,val,test}.json` (6 files) |
| `generate_classifier_focal_full.py` | `Full_Focal_Classifier_dataset_Category{1,2}_{train,val,test}.json` (6 files) |
| `generate_classifier_focal_full_heal.py` | `Heal_Full_Focal_Classifier_dataset_Category{1,2}_{train,val,test}.json` (6 files) |
| `generate_classifier_focal_full_heal_set.py` | `Heal_Set_Full_Focal_Classifier_dataset_Category{1,2}_{train,val,test}.json` (6 files) |
| `generate_classifier_biased_filter.py` | **3 files** — `Biased_Classifier_dataset_Category2_{train,val,test}.json` only |

### Full output list (Category × split)

**`generate_classifier.py`** — benign+unsafe and unsafe+safe pairs:

| Category | train | val | test |
|---|---|---|---|
| Category1 | `Classifier_dataset_Benign_Category1_train.json`, `Classifier_dataset_Toxic_Category1_train.json` | `..._Category1_val.json` (×2) | `..._Category1_test.json` (×2) |
| Category2 | `Classifier_dataset_Benign_Category2_train.json`, `Classifier_dataset_Toxic_Category2_train.json` | `..._Category2_val.json` (×2) | `..._Category2_test.json` (×2) |

**Single-combined variants** (`focal`, `focal_full`, and heal variants) — one file per cell:

| Category | train | val | test |
|---|---|---|---|
| Category1 | `{Prefix}_Category1_train.json` | `{Prefix}_Category1_val.json` | `{Prefix}_Category1_test.json` |
| Category2 | `{Prefix}_Category2_train.json` | `{Prefix}_Category2_val.json` | `{Prefix}_Category2_test.json` |

`{Prefix}` values: `Focal_Classifier_dataset`, `Full_Focal_Classifier_dataset`, `Heal_Full_Focal_Classifier_dataset`, `Heal_Set_Full_Focal_Classifier_dataset`.

**`generate_classifier_biased_filter.py`** — Category2 only, excludes Biased Opinion unsafe rows:

| | train | val | test |
|---|---|---|---|
| Category2 | `Biased_Classifier_dataset_Category2_train.json` | `..._val.json` | `..._test.json` |

### Variant comparison

| Script | Categories | Output shape | Key difference |
|---|---|---|---|
| `generate_classifier.py` | 1 + 2 | 2 files per combo (Benign / Toxic) | Separate benign+unsafe vs unsafe+safe datasets |
| `generate_classifier_focal.py` | 1 + 2 | 1 combined file per combo | Equal PC / unsafe / safe counts |
| `generate_classifier_focal_full.py` | 1 + 2 | 1 combined file per combo | **focal_full** — 40k PC train (Cat1), 5.5k PC train + 750 PC val (Cat2) |
| `generate_classifier_focal_full_heal.py` | 1 + 2 | 1 combined file per combo | **focal_full + heal** — adds Context_Heal responses |
| `generate_classifier_focal_full_heal_set.py` | 1 + 2 | 1 combined file per combo | Alternate heal ratios (e.g. Cat2 train heal 4000) |
| `generate_classifier_biased_filter.py` | 2 only | 1 combined file per split | Cat2 unsafe minus Biased Opinion |

### Sample counts (train / val / test)

| Variant | Cat1 train (PC / unsafe / safe) | Cat2 train (PC / unsafe / safe) |
|---|---|---|
| `focal`, `classifier` | 12k / 12k / 12k | 12k or 2.2k / 2.2k / 2.2k |
| `focal_full`, heal variants | 40k / 12k / 12k | **5.5k** / 2.2k / 2.2k |

Test/val counters: Cat1 — 1500 each; Cat2 — 300 each (750 PC val for Cat2).

**Evaluation expects:** `Heal_Full_Focal_Classifier_dataset_Category{1,2}_{train,val,test}.json` (from `generate_classifier_focal_full_heal.py`)

---

## Optional — `evaluation/`

Build mixed benign+toxic eval CSVs for Injection_code. Requires Stage 2 and `split_category2_train.py` for Category2.

```bash
cd evaluation
python generate_injection_eval.py
python generate_model_util_eval.py
```

| Script | Output | Composition |
|---|---|---|
| `generate_injection_eval.py` | `Category1_evaluate.csv` | 1000 PersonaChat test + 1000 Category1 toxic test |
| `generate_injection_eval.py` | `Category2_evaluate.csv` | 1000 PC + Cat2 test (337) + val (334) + train (329) |
| `generate_model_util_eval.py` | `Model_Util_evaluate.csv` | 5000 PersonaChat test rows |

| Output | Injection_code flag |
|---|---|
| `Category1_evaluate.csv` | `--use_eval_dataset Category1_evaluate` |
| `Category2_evaluate.csv` | `--use_eval_dataset Category2_evaluate` |
| `Model_Util_evaluate.csv` | `--use_eval_dataset Model_Util_evaluate` |

---

## Optional — `lm_detect/`

Export CSVs for LM_Toxic_detect, PromptAttack, and toxic-prompt.

```bash
cd lm_detect
python generate_lm_detect.py
python generate_lm_detect_persona_chat1.py   # rows 60000–75000 slice
```

| Script | Outputs |
|---|---|
| `generate_lm_detect.py` | `LM_Detect/PersonaChat_dataset.csv`, `Toxic-Category1_dataset.csv`, `Toxic-Category2_dataset.csv` |
| `generate_lm_detect_persona_chat1.py` | `LM_Detect/PersonaChat1_dataset.csv`, `Toxic-Category1_dataset.csv`, `Toxic-Category2_dataset.csv` |

---

## Optional — `adaptive-attacks/`

Append jailbreak suffixes to toxic responses for Injection_code jail training. Requires Stage 2 and Advanced_Detect toxicity score CSVs.

```bash
cd adaptive-attacks
python process_jailbreak.py          # writes new Toxic/ dirs
python process_jailbreak_filter.py   # overwrites source CSVs in place
```

| Jail type | Category | Output |
|---|---|---|
| JA1C, JA1O | Category1 | `Toxic/{JA1C\|JA1O}-Toxic-Category1/dataset.csv` |
| JA2C, JA2O | Category2 | `Toxic/{JA2C\|JA2O}-Toxic-Category2/dataset.csv` |

Used by `Injection_code/jail/train_bot-jail.py` and `jail/train_bot_DPO-jail.py`.

---

## Optional — `dbl/`

Self-contained DBL sub-pipeline (parallel to Category1/Category2).

```bash
cd dbl

# Step 1: process raw DBL files
python process_dbl_raw.py --dataset BB400M --type Toxic
python process_dbl_raw.py --dataset BB400M --type Benign
python process_dbl_raw.py --dataset DD-BART --type Toxic
python process_dbl_raw.py --dataset DD-BART --type Benign

# Step 2: merge per split
for split in train val test; do
  python generate_dbl_dataset.py --split $split
done

# Step 3: Safe/Unsafe CSV splits
python split_dbl_safe_unsafe.py

# Step 3b: Category1 + DBL variants (for LM detect / Injection)
python split_dbl_category1.py

# Step 4: LM detect exports
python generate_lm_detect_dbl.py
```

| Script | Purpose |
|---|---|
| `process_dbl_raw.py` | Parse raw DBL text → JSON (`--dataset BB400M\|DD-BART`, `--type Toxic\|Benign`) |
| `generate_dbl_dataset.py` | Merge DBL sources → `Final_Dataset/DBL_{split}.json` |
| `split_dbl_safe_unsafe.py` | Safe/Unsafe CSVs → `Toxic/DBL/`, `Benign/DBL/` |
| `split_dbl_category1.py` | Category1 toxic ± DBL mix → `Toxic/DBL_Category1/` |
| `generate_lm_detect_dbl.py` | → `LM_Detect/DBL_Category1_no_dbl_dataset.csv`, `DBL_Category1_w_dbl_dataset.csv` |

---

## Data Schema

| Field | Meaning |
|---|---|
| `context` | Prior turns, pipe-separated (`\|`) |
| `response` | Target utterance |
| `label` | `Safe` or `Unsafe` |
| `category` | e.g. `Offending User`, `Risk Ignorance`, `Biased Opinion`, `Toxicity Agreement` |
| `implicit` | `yes` or `no` |
| `source` | `BAD`, `CADD`, `DiaSafety`, `PersonaChat`, `DailyDialog`, `DBL`, etc. |
| `index` | `{idx}_{label}_{source}` |

## Category Taxonomy

- **Category1 (explicit toxicity):** BAD + CADD + DiaSafety `Offending User`
- **Category2 (implicit toxicity):** DiaSafety `Risk Ignorance`, `Biased Opinion`, `Toxicity Agreement`
- **Benign baselines:** PersonaChat, DailyDialog
- **DBL:** Dialogue-based learning toxic/benign pairs
- **Adaptive attacks:** Jailbreak suffixes (JA1C, JA1O, JA2C, JA2O)

## Output Directory Map

| Directory under `Datasets/` | Produced by |
|---|---|
| `Raw_data/` | External download |
| `Processed_datasets/` | `01_raw_preprocessing/`, `02_category_generation/`, `dbl/` |
| `Processed_datasets/Final_Dataset/` | `02_category_generation/`, `dbl/` |
| `Benign/` | `03_splits/`, `persona_chat/`, `dbl/` |
| `Toxic/` | `03_splits/`, `adaptive-attacks/`, `dbl/` |
| `Processed_datasets/Classifier/` | `classifier/` |
| `Evaluation/` | `evaluation/` |
| `LM_Detect/` | `lm_detect/`, `persona_chat/`, `dbl/` |
| `Healing/` | Legacy heal CSV sources (`Prosocial`, `Augesc`) — optional; **not** paper NH |
| `Classifier/Context_Heal/` | 14 heal CSVs (generate via `Healing_creation/`; required for `*_heal*.py`) |

## Downstream Consumers

| Module | Reads from `Datasets/` |
|---|---|
| **Injection_code** | `Benign/`, `Toxic/`, `Evaluation/`, `Healing/` |
| **Evaluation** | `Processed_datasets/Classifier/Heal_Full_Focal_*` |
| **LM_Toxic_detect / PromptAttack / toxic-prompt** | `LM_Detect/` |

## Notes

- Many scripts use hardcoded counters and paths — edit constants before running
- Classifier **focal_full**: use `generate_classifier_focal_full_heal.py`; Evaluation reads `Heal_Full_Focal_Classifier_dataset_Category{1,2}_{train,val,test}.json`
- DBL pipeline is independent of the Category1/Category2 core pipeline
