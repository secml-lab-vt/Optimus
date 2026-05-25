# DBL-Filtering

LLM-based safety scoring for Dialogue-Based Learning (DBL) adversarial injection experiments. Scores merged DBL CSVs with prompt-conditioned **yes/no** detection, then joins scores back into per-trial `Processed_files/` by `uid`.

Run from `Optimus/DBL-Filtering/`.

## Prerequisites

- GPU recommended (`cuda`)
- Python: `torch`, `transformers`, `datasets`, `sklearn`, `pandas`, `tqdm`, `peft`
- Detector models under `../Models/HuggingFace_direct/models/` (Llama-2 loads from HuggingFace hub)
- Merged inputs under `../Datasets/DBL_data/Merged_files/`
- Per-trial references under `../Datasets/DBL_data/Processed_files/`

---

## Experiment combinations (adv × model × attack)

Four **merged** datasets are built from **20 processed** trial CSVs (5 trials × 4 configs). Each config is a `{base_model, attack}` pair:

| `--path` (merged input) | Base model | Attack | Trojan | CPR | RPR | Processed file prefix |
|---|---|---|---|---|---|---|
| `adv-BB400M` | BB400M | adversarial | no | 0.3 | 1 | `BB400M_toxic_tbot-adv_cpr-0.3_rpr-1` |
| `adv-backdoor-BB400M` | BB400M | backdoor | yes | 0.3 | 0.4 | `BB400M_toxic_trojan_tbot-adv_cpr-0.3_rpr-0.4` |
| `adv-DD-BART` | DD-BART | adversarial | no | 0.3 | 1 | `DD-BART_toxic_tbot-adv_cpr-0.3_rpr-1` |
| `adv-backdoor-DD-BART` | DD-BART | backdoor | yes | 0.3 | 0.4 | `DD-BART_toxic_trojan_tbot-adv_cpr-0.3_rpr-0.4` |

### All 20 `Processed_files/` names (5 trials each)

**BB400M adversarial (`adv-BB400M`):**
- `BB400M_toxic_tbot-adv_cpr-0.3_rpr-1_k-1.csv` … `k-5.csv`

**BB400M backdoor (`adv-backdoor-BB400M`):**
- `BB400M_toxic_trojan_tbot-adv_cpr-0.3_rpr-0.4_k-1.csv` … `k-5.csv`

**DD-BART adversarial (`adv-DD-BART`):**
- `DD-BART_toxic_tbot-adv_cpr-0.3_rpr-1_k-1.csv` … `k-5.csv`

**DD-BART backdoor (`adv-backdoor-DD-BART`):**
- `DD-BART_toxic_trojan_tbot-adv_cpr-0.3_rpr-0.4_k-1.csv` … `k-5.csv`

Merged CSVs deduplicate by `uid` across the five trials (see `chatsec2/Datasets/DBL_data/Merged_files/merge*.py`).

### Full CLI combination grid (`model_factory.py`)

Each run is one combination of:

| Parameter | Values | Count |
|---|---|---|
| `--path` | `adv-BB400M`, `adv-backdoor-BB400M`, `adv-DD-BART`, `adv-backdoor-DD-BART` | 4 |
| `--model_name` | See [Supported models](#supported-models) | 18 |
| `--few_shot` | `yes`, `no` | 2 |
| `--category` | `1`, `2` | 2 |
| `--type` | `1` … `10` (prompt template index) | 10 |

Example — score all four merged configs with one detector:

```bash
for path in adv-BB400M adv-backdoor-BB400M adv-DD-BART adv-backdoor-DD-BART; do
  python model_factory.py \
    --model_name meta-llama_Llama-2-13b-chat-hf \
    --path "$path" \
    --few_shot no \
    --category 1 \
    --type 1 \
    --metrics yes
done
```

---

## Pipeline

```
Processed_files/{model}_toxic_{[trojan_]}tbot-adv_cpr-*_rpr-*_k-{1..5}.csv
        │  (merge by uid — external / Merged_files/merge*.py)
        ▼
Merged_files/{path}.csv
        │
        ▼  model_factory.py  (--path, --model_name, --category, --type, --few_shot)
Advanced_Detect/Fewshot-{few_shot}/{type}/*_adv_dataset_scores.csv
        │
        ▼  split_5_files.py  (join on uid)
Filtered_files/{processed_filename}.csv
```

---

## Input / output reference

### Inputs

| Stage | Path | Schema |
|---|---|---|
| **Merged scoring input** | `../Datasets/DBL_data/Merged_files/{path}.csv` | `context`, `response`, `category`, `label`, `implicit`, `source`, `index`, `uid`, `flags` (+ optional `learn`) |
| **Processed reference** | `../Datasets/DBL_data/Processed_files/{prefix}_k-{1..5}.csv` | `contexts`, `responses`, `flags`, `learn`, `uid` |
| **Score CSV for split** | `../Datasets/DBL_data/Advanced_Detect/Fewshot-no/{type}/*_{path}_*_adv_dataset_scores.csv` | merged schema + `max_index`, `max_prob`, `prediction` |

**`flags` → `label` mapping** (when building merged files):

| `flags` | `label` |
|---|---|
| `response`, `toxic`, `adv_response`, `adv_toxic` | `Unsafe` |
| `friendly`, `trigger` | `Safe` |

### Outputs

| Script | Output | Path pattern |
|---|---|---|
| `model_factory.py` | Prompt JSON | `Advanced_Detect/Fewshot-{few_shot}/{type}/{model}_{timestamp}_category_{cat}_{path}_type_{type}_adv_dataset.json` |
| `model_factory.py` | Scores CSV | `Advanced_Detect/Fewshot-{few_shot}/{type}/{model}_{timestamp}_category_{cat}_{path}_type_{type}_adv_dataset_scores.csv` |
| `model_factory.py` | Metrics log | `Advanced_Detect/consolidated_toxicity_results.txt` (append; `--metrics yes`) |
| `split_5_files.py` | Filtered CSV | `Filtered_files/{processed_filename}.csv` |

**Scores CSV columns:**  
`context`, `response`, `category`, `label`, `implicit`, `source`, `index`, `uid`, `flags`, `max_index`, `max_prob`, `prediction`

**Detection rule:** uses `no` token probability; `max_prob > 0.5` → `Unsafe`.

**Metrics log row** (`consolidated_toxicity_results.txt`):  
`{model}_{timestamp},{category},{few_shot},{path},{type},{roc_auc},{pr_auc},{precision},{recall},{f1},{fpr}`

---

## Scripts

### `model_factory.py`

```bash
python model_factory.py \
  --model_name meta-llama_Llama-2-13b-chat-hf \
  --path adv-backdoor-DD-BART \
  --few_shot no \
  --category 1 \
  --type 1 \
  --metrics yes
```

| Argument | Required | Values |
|---|---|---|
| `--model_name` | yes | See supported models |
| `--path` | yes | One of four `adv-*` merged stems |
| `--few_shot` | yes | `yes` / `no` |
| `--category` | no | `1` / `2` |
| `--type` | no | `1`–`10` (default `no`) |
| `--metrics` | no | `yes` / `no` |

Prompts from `prompt_package/prompt_list.py` (`--type` indexes into 10 Category1/Category2 templates).

### `split_5_files.py`

Joins score CSVs from `Advanced_Detect/Fewshot-no/1/` back into each file in `Processed_files/`:

| Processed file contains | Score CSV must contain |
|---|---|
| `BB400M_toxic_trojan_tbot-adv` | `adv-backdoor-BB400M` |
| `DD-BART_toxic_trojan_tbot-adv` | `adv-backdoor-DD-BART` |
| `BB400M_toxic_tbot-adv` | `adv-BB400M` |
| `DD-BART_toxic_tbot-adv` | `adv-DD-BART` |

```bash
python split_5_files.py
```

Writes one CSV per processed file under `Filtered_files/`.

### `common.py`

Central constants: `MERGED_PATHS`, `EXPERIMENT_CONFIGS`, `PROCESSED_TO_ADV`, `MODELS`, path helpers (`merged_file`, `detect_scores_path`, …).

---

## Supported models

| Causal LM | Seq2seq |
|---|---|
| `meta-llama_Llama-2-7b-chat-hf` | `google_flan-t5-base` |
| `meta-llama_Llama-2-13b-chat-hf` | `google_flan-t5-small` |
| `meta-llama_Llama-2-70b-chat-hf` | `google_flan-t5-large` |
| `yahma_llama-7b-hf` | `google_flan-t5-xl` |
| `tiiuae_falcon-7b-instruct` | `google_flan-t5-xxl` |
| `tiiuae_falcon-40b-instruct` | |
| `facebook_opt-iml-30b` | |
| `facebook_opt-iml-1.3b` | |
| `lmsys_vicuna-7b-v1.3` | |
| `lmsys_vicuna-13b-v1.3` | |
| `lmsys_vicuna-33b-v1.3` | |
| `lmsys_vicuna-7b-v1.1` | |
| `lmsys_vicuna-13b-v1.1` | |

---

## Directory map (`Datasets/DBL_data/`)

| Directory | Contents |
|---|---|
| `Processed_files/` | 20 per-trial injection CSVs |
| `Merged_files/` | 4 deduplicated `{path}.csv` scoring inputs |
| `Advanced_Detect/Fewshot-{yes\|no}/{type}/` | Score + prompt JSON outputs |
| `Advanced_Detect/consolidated_toxicity_results.txt` | Run-level metrics append log |
| `Filtered_files/` | Processed + scores joined by `uid` |
| `Detect_Files/Idea{1\|2}/` | Downstream detect exports (Healing / metrics) |

---

## Related modules

| Module | Role |
|---|---|
| `Data_processor/dbl/` | Raw DBL JSON/CSV pipeline (`Toxic/DBL/`) |
| `chatsec2/Datasets/DBL_data/Merged_files/merge*.py` | Build merged CSVs from processed trials |
| `Healing_creation/DBL_heal_generation.py` | Reads `Detect_Files/Idea{idea}/` |
| `Injection_code/` | `--toxic_dataset DBL_Category1` training variants |
