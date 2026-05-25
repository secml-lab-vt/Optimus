# LM Toxic Detect

Score conversational turns for toxicity using **local LLMs** (yes/no next-token probability) or **external API scorers** (Perspective, OpenAI, Azure, Unitary Detoxify). Outputs land under `Datasets/LM_Scores/`, `Datasets/Advanced_Detect/`, and related folders — then feed **Injection_code** training (`--model_vers Perspective`, threshold selection, adversarial/jailbreak filtering).

## Directory layout

Scripts are grouped by detection strategy. Run from the **type folder** (or pass the full path from `LM_Toxic_detect/`).

```
LM_Toxic_detect/
├── README.md
├── prompt_package/              # Shared prompts (Idea1 vs Idea2 rubrics)
│   ├── prompt_list.py           # Idea1: detailed toxicity rubric
│   ├── prompt_list_idea2.py     # Idea2: safe-to-continue yes/no
│   ├── hateguard_prompt.py      # HateGuard-specific prompts
│   └── prompt_list1.py          # Alternate prompt variants
├── idea1/                       # Idea1 LM scorers (→ LM_Scores/)
│   ├── ModelFactory.py
│   ├── ModelFactory_Jailbreak.py
│   ├── ModelFactory_context_heal.py
│   └── ModelFactory_adversarial.py
├── idea2/                       # Idea2 LM scorers (→ Advanced_Detect/)
│   ├── ModelFactory_Idea2.py
│   ├── ModelFactory_Idea2_Jailbreak.py
│   ├── ModelFactory_Idea2_context_heal.py
│   ├── ModelFactory_Idea2_adversarial.py
│   └── ModelFactory_Idea_Prompt_shield.py
├── api_scorers/                 # External API toxicity APIs
│   ├── PerspectiveApi_detect_log.py
│   ├── PerspectiveApi_detect_log_context_heal_check.py
│   ├── OpenAI_detect_log.py
│   ├── Azure_detect_log.py
│   └── UnitaryDetoxify_detect_log_new.py
├── specialized/                 # Domain-specific detectors
│   └── ModelFactory_hateguard.py
└── utilities/                   # Post-hoc analysis helpers
    ├── result_process.py
    ├── conversation.py
    └── Prompt_formatter.py
```

Subfolder scripts bootstrap `prompt_package` via `sys.path` (same pattern as [Injection_code](../Injection_code/README.md)).

---

## Idea1 vs Idea2

Both families load a causal or seq2seq model, format each `(context, response)` pair with a system prompt, and score **P("yes")** vs **P("no")** on the next token. They differ in **prompt rubric** and **output directory**.

| | **Idea1** (`idea1/`) | **Idea2** (`idea2/`) |
|---|---|---|
| **Prompts** | `prompt_package/prompt_list.py` — detailed category rubrics (rudeness vs harm/bias) | `prompt_package/prompt_list_idea2.py` — “safe to generate next turn?” |
| **Default output** | `Datasets/LM_Scores/Fewshot-{yes\|no}/` | `Datasets/Advanced_Detect/Fewshot-{yes\|no}/` |
| **Injection_code flag** | `--filter1 idea1` / legacy Idea1 paths | `--filter1 idea2` (default in many scripts) |
| **When to use** | Original Optimus LM-detect baseline | Improved safe-continuation framing used in paper Idea2 pipeline |

---

## Adversarial vs Jailbreak vs Context Heal vs API

These are **different input sources and output trees**. Do not conflate jail suffix scripts with adversarial split scripts.

| Variant | Folder | Input CSV | Output dir |
|---|---|---|---|
| **Standard** | `idea1/ModelFactory.py`, `idea2/ModelFactory_Idea2.py` | `Datasets/LM_Detect/{path}_dataset.csv` | `LM_Scores/` or `Advanced_Detect/` |
| **Jailbreak** | `*_Jailbreak.py`, `ModelFactory_Idea_Prompt_shield.py` | Same `LM_Detect/` CSV + `--jail_category` suffix appended to prompt | Filenames include `{JA*}-` prefix |
| **Context heal** | `*_context_heal.py` | `Datasets/Context_Heal/{path}.csv` | `Context_Heal_Toxicity/` (Idea1) or `Context_Heal_Toxicity_Advanced_Detect/` (Idea2) |
| **Adversarial surrogate** | `*_adversarial.py` | `Datasets/Adversarial_attack/Adversarial_surrogate/splits/{path}_{file_num}.csv` | `Adversarial_attack/Adversarial_surrogate/Results/` |
| **API scorers** | `api_scorers/` | `LM_Detect/` (or `Context_Heal/` for heal check) | `LM_Scores/` or `Context_Heal_Toxicity/` |
| **HateGuard** | `specialized/ModelFactory_hateguard.py` | `Datasets/New_Waves/{path}_dataset.csv` | `Datasets/New_Waves/Results/` |

**Jailbreak categories** (`--jail_category`): `JA1C`, `JA1O`, `JA2C`, `JA2O` — adaptive suffix strings appended during scoring (see [Injection_code jail docs](../Injection_code/README.md#adversarial-vs-jailbreak)). Generate jailbreak toxic CSVs first with `Data_processor/adaptive-attacks/process_jailbreak.py`.

**Adversarial** scripts score rows from adversarial heal-generation splits (`Healing_creation/*adversarial*`). Use `--file_num` to select which split file under `Adversarial_surrogate/splits/`.

**Context heal** scripts score healed conversation CSVs produced by `Healing_creation/` generators before injection with `--healing_dataset Context_Heal`.

---

## Scripts by folder

### `idea1/` — Idea1 local LLM scorers

| Script | Purpose |
|---|---|
| `ModelFactory.py` | Standard toxicity scoring → `LM_Scores/Fewshot-{yes\|no}/` |
| `ModelFactory_Jailbreak.py` | Jailbreak suffix on prompts; output filenames include `{jail_category}-` |
| `ModelFactory_context_heal.py` | Score context-heal CSVs → `Context_Heal_Toxicity/` |
| `ModelFactory_adversarial.py` | Score adversarial surrogate splits → `Adversarial_surrogate/Results/` |

### `idea2/` — Idea2 local LLM scorers

| Script | Purpose |
|---|---|
| `ModelFactory_Idea2.py` | Standard Idea2 scoring → `Advanced_Detect/Fewshot-{yes\|no}/` |
| `ModelFactory_Idea2_Jailbreak.py` | Idea2 + jail suffix → `Advanced_Detect/` |
| `ModelFactory_Idea2_context_heal.py` | Heal CSVs with Idea2 rubric → `Context_Heal_Toxicity_Advanced_Detect/` |
| `ModelFactory_Idea2_adversarial.py` | Adversarial splits with Idea2 rubric |
| `ModelFactory_Idea_Prompt_shield.py` | PromptShield / jail variant for Idea2 advanced detect |

### `api_scorers/` — External toxicity APIs

| Script | Backend | Notes |
|---|---|---|
| `PerspectiveApi_detect_log.py` | Google Perspective API | Primary scorer referenced as `--model_vers Perspective` in Injection_code |
| `PerspectiveApi_detect_log_context_heal_check.py` | Perspective | Reads `Context_Heal/` → writes `Context_Heal_Toxicity/` |
| `OpenAI_detect_log.py` | OpenAI moderation | Writes under `LM_Scores/Fewshot-{yes\|no}/12/` |
| `Azure_detect_log.py` | Azure Content Safety | Same `12/` subfolder layout as OpenAI |
| `UnitaryDetoxify_detect_log_new.py` | Unitary Detoxify | Local/API hybrid; → `LM_Scores/` |

API scripts require credentials in environment or config (see each file’s client setup).

### `specialized/`

| Script | Purpose |
|---|---|
| `ModelFactory_hateguard.py` | HateGuard-style prompts (`hateguard_prompt.py`); inputs/outputs under `Datasets/New_Waves/` |

### `utilities/`

| Script | Purpose |
|---|---|
| `result_process.py` | Load a scored CSV, aggregate yes-token probs, print PR-AUC / F1, append to consolidated results |
| `conversation.py` | FastChat-style conversation templates (imported by some model formatters) |
| `Prompt_formatter.py` | Standalone prompt formatting helper |

### `prompt_package/`

| File | Used by |
|---|---|
| `prompt_list.py` | All `idea1/` ModelFactory scripts |
| `prompt_list_idea2.py` | All `idea2/` ModelFactory scripts |
| `hateguard_prompt.py` | `specialized/ModelFactory_hateguard.py` |

---

## Quick start

**Prerequisite:** Export LM-detect CSVs from [Data_processor/lm_detect](../Data_processor/README.md#optional--lm_detect):

```bash
cd ../Data_processor/lm_detect
python generate_lm_detect.py
# → Datasets/LM_Detect/Toxic-Category1_dataset.csv, Benign-PersonaChat_dataset.csv, …
```

**Standard Idea2 run** (recommended default):

```bash
cd idea2
python ModelFactory_Idea2.py \
    --model_name meta-llama_Llama-2-7b-chat-hf \
    --path Toxic-Category1 \
    --few_shot no \
    --category 1 \
    --type 1 \
    --metrics yes
```

**Idea1 baseline:**

```bash
cd idea1
python ModelFactory.py \
    --model_name meta-llama_Llama-2-7b-chat-hf \
    --path Toxic-Category1 \
    --few_shot no \
    --category 1 \
    --type 1 \
    --metrics yes
```

**Jailbreak scoring:**

```bash
cd idea2
python ModelFactory_Idea2_Jailbreak.py \
    --model_name meta-llama_Llama-2-7b-chat-hf \
    --path Toxic-Category1 \
    --jail_category JA1C \
    --category 1 \
    --few_shot no \
    --type 1
```

**Context heal check:**

```bash
cd idea1
python ModelFactory_context_heal.py \
    --model_name meta-llama_Llama-2-7b-chat-hf \
    --path <heal_csv_stem> \
    --category 1 \
    --few_shot no \
    --type 1
```

**Perspective API** (no local GPU model):

```bash
cd api_scorers
python PerspectiveApi_detect_log.py \
    --path Toxic-Category1 \
    --few_shot no \
    --category 1 \
    --type 1 \
    --metrics yes
```

---

## Output artifacts

Each run writes:

1. **Per-row CSV** — `{model}_{timestamp}_category_{cat}_{path}_type_{type}_*_toxicity_scores.csv` with columns `context`, `response`, `max_prob`, `prediction`, …
2. **JSON dataset dump** — same stem with `_toxicity_dataset.json` (or `_adv_dataset.json` / `_jail_dataset.json` for Idea2)
3. **Consolidated metrics** — `consolidated_toxicity_results.txt` in the output folder when `--metrics yes`

| Script type | Primary output directory |
|---|---|
| Idea1 standard / jail | `Datasets/LM_Scores/Fewshot-{yes\|no}/{type}/` |
| Idea2 standard / jail | `Datasets/Advanced_Detect/Fewshot-{yes\|no}/{type}/` |
| Idea1 context heal | `Datasets/Context_Heal_Toxicity/Fewshot-{yes\|no}/{type}/` |
| Idea2 context heal | `Datasets/Context_Heal_Toxicity_Advanced_Detect/Fewshot-{yes\|no}/{type}/` |
| Adversarial | `Datasets/Adversarial_attack/Adversarial_surrogate/Results/` |
| API (standard) | `Datasets/LM_Scores/Fewshot-{yes\|no}/` |
| API (OpenAI/Azure) | `Datasets/LM_Scores/Fewshot-{yes\|no}/12/` |
| HateGuard | `Datasets/New_Waves/Results/Fewshot-{yes\|no}/{type}/` |

Score CSVs are consumed by **Injection_code** when selecting toxic rows above a threshold (`--model_vers`, `--threshold`, `--adversarial`, jail `JA*` prefixes).

---

## Common CLI parameters

| Flag | Description |
|---|---|
| `--model_name` | HuggingFace model id (underscores for `/`, e.g. `meta-llama_Llama-2-7b-chat-hf`) |
| `--path` | Dataset stem matching `LM_Detect/{path}_dataset.csv` or heal/adversarial path stem |
| `--few_shot` | `yes` or `no` — controls few-shot examples in prompt |
| `--category` | `1` or `2` (jail scripts also accept `3`) — toxicity category |
| `--type` | Prompt variant index `1`–`10` (maps to `sys_prompts[type-1]`) |
| `--metrics` | `yes` → append ROC/PR-AUC/F1 to consolidated results file |
| `--device` | Override device (`cpu` / `cuda`; jail scripts) |
| `--jail_category` | **Jail scripts only:** `JA1C`, `JA1O`, `JA2C`, `JA2O` |
| `--file_num` | **Adversarial scripts only:** split file suffix under `Adversarial_surrogate/splits/` |

Supported local models include LLaMA-2 chat, Falcon, FLAN-T5, OPT, Vicuna variants (see `--model_name` choices in each script).

---

## End-to-end flow

```
Data_processor/lm_detect/  →  Datasets/LM_Detect/*.csv
        │
        ▼
LM_Toxic_detect/{idea1|idea2|api_scorers}/
        │
        ▼
Datasets/LM_Scores/  or  Advanced_Detect/  (+ heal / adversarial branches)
        │
        ▼
Injection_code/{standard|jail|…}/  (--model_vers, --threshold, --adversarial)
        │
        ▼
Models/Custom/model_runs/…  →  Evaluation/RTR_Evaluation_Classifier*/
```

Optional branches:

- **Context heal:** `Healing_creation/` → `Context_Heal/*.csv` → `*_context_heal.py` → `Context_Heal_Toxicity/`
- **Adversarial:** `Healing_creation/*adversarial*` → `Adversarial_surrogate/splits/` → `*_adversarial.py`
- **Jailbreak:** `Data_processor/adaptive-attacks/` → jail-toxic CSVs → `*_Jailbreak.py`

---

## Dependencies

`torch`, `transformers`, `peft`, `datasets`, `sklearn`, `tqdm`

Local model weights under `Models/HuggingFace_direct/models/` (or HuggingFace hub for LLaMA-2 ids). Input CSVs under `Datasets/LM_Detect/` from [Data_processor](../Data_processor/README.md).
