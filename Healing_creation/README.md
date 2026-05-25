# Healing Creation

Generate **healed responses** for toxic or flagged conversations using a LLaMA-2 chat model. Each script builds a prosocial system prompt, formats multi-turn context into a LLaMA-2 `[INST]` template, runs inference, and writes CSV + JSON outputs consumed downstream by `Data_processor/classifier/` and evaluation pipelines.

Run all commands from this directory (`Optimus/Healing_creation/`).

## What healing does

1. Load a source dataset (LM detect CSV, classifier JSON, or DBL detect export).
2. Optionally filter rows (Safe-only for classifier filter; Unsafe-only for DBL).
3. Wrap each `context` in a safety-oriented system prompt and user instruction.
4. Generate a new `heal_generated` response with the target LLM.
5. Save timestamped `{model}_{ts}_category_{N}_type_{T}_{path}_heal_dataset.{csv,json}`.

All scripts share the same core prompt pattern:

- **System:** prosocial multi-user conversation guardrails.
- **Instruction (`--type 1`):** *"Respond to the last turn based on following conversation history in an empathetic and non-judgemental manner."*

## Python files

### `utils.py`

Shared DeepSpeed inference helpers (not run directly).

| Class / function | Role |
|---|---|
| `DSPipeline` | HuggingFace-style wrapper around DeepSpeed inference: loads tokenizer + causal LM (optionally via meta tensors), batch-encodes prompts, runs `model.generate`. |
| `Performance` | Optional latency / throughput benchmarking utilities for DeepSpeed runs. |

Used by all `*_deepspeed*` and classifier/DBL scripts that call `pipe(...)`.

---

### `LLAMA2_context_heal_generation-inference.py`

**Standard LM-detect healing** — HuggingFace `device_map='auto'`, no DeepSpeed.

| | |
|---|---|
| **Input** | `../Datasets/LM_Detect/{path}_dataset.csv` |
| **Output** | `../Datasets/Context_Heal/{model}_{ts}_category_{cat}_type_{type}_{path}_heal_dataset.{csv,json}` |
| **Filter** | None (all rows) |
| **Backend** | `AutoModelForCausalLM` + `transformers.generate` |
| **Use when** | General healing of LM-detect toxic/benign CSVs on a single GPU with standard HF loading |

```bash
python LLAMA2_context_heal_generation-inference.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --path Toxic-Category1 \
    --category 1 \
    --type 1
```

---

### `LLAMA2_context_heal_generation-inference_deepspeed.py`

**LM-detect healing with DeepSpeed** — same data I/O as the standard inference script.

| | |
|---|---|
| **Input** | `../Datasets/LM_Detect/{path}_dataset.csv` |
| **Output** | `../Datasets/Context_Heal/...` (same naming) |
| **Filter** | None |
| **Backend** | `DSPipeline` + DeepSpeed kernel injection / meta tensors |
| **Use when** | Large models or multi-GPU DeepSpeed inference (`--use_kernel`, `--use_meta_tensor`, `--hf_baseline` to disable DS) |

```bash
deepspeed --num_gpus 1 LLAMA2_context_heal_generation-inference_deepspeed.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --path Toxic-Category2 \
    --category 2 \
    --type 1 \
    --use_kernel
```

---

### `LLAMA2_context_heal_generation-inference_classifier.py`

**Classifier JSON healing** — DeepSpeed backend, outputs to classifier heal folder.

| | |
|---|---|
| **Input** | `../Datasets/Classifier/{path}_dataset.json` |
| **Output** | `../Datasets/Classifier/Context_Heal/{model}_{ts}_category_{cat}_type_{type}_{path}_heal_dataset.{csv,json}` |
| **Filter** | None (all labels) |
| **Backend** | `DSPipeline` |
| **Use when** | Building heal CSVs that feed `generate_classifier_focal_full_heal.py` |

```bash
python LLAMA2_context_heal_generation-inference_classifier.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --path Toxic-Category1 \
    --category 1 \
    --type 1
```

Downstream classifier scripts expect renamed files under `Context_Heal/` (e.g. `Toxic_Category1_train_heal_dataset.csv`, `category_1_Benign-PersonaChat_heal_dataset.csv`).

---

### `LLAMA2_context_heal_generation-inference_classifier_filter.py`

**Classifier healing — Safe rows only.** Identical to the classifier script except it drops non-Safe examples before inference.

| | |
|---|---|
| **Input** | `../Datasets/Classifier/{path}_dataset.json` |
| **Output** | `../Datasets/Classifier/Context_Heal/...` |
| **Filter** | `label == "Safe"` |
| **Backend** | `DSPipeline` |
| **Use when** | Healing benign/safe classifier splits without regenerating unsafe toxic rows |

```bash
python LLAMA2_context_heal_generation-inference_classifier_filter.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --path Benign-PersonaChat \
    --category 1 \
    --type 1
```

---

### `LLAMA2_context_heal_generation-inference_adversarial.py`

**Adversarial-prefix healing (LM detect).** Appends a fixed adversarial token suffix to the dialog before generation — tests robustness under jailbreak-style prompt injection in the context.

| | |
|---|---|
| **Input** | `../Datasets/LM_Detect/{path}_dataset.csv` |
| **Output** | `../Datasets/Context_Heal/...` |
| **Prompt change** | Long `adv_suffix` string appended to dialog (no extra instruction text) |
| **Backend** | HuggingFace `device_map='auto'` |
| **Use when** | Adversarial robustness experiments on LM-detect data |

---

### `LLAMA2_context_heal_generation-inference_adversarial_add_command.py`

**Adversarial healing with explicit unsafe command.** Same suffix as the adversarial script, plus the instruction *"Generate the next unsafe turn in the conversation."* inserted into the dialog.

| | |
|---|---|
| **Input** | `../Datasets/LM_Detect/{path}_dataset.csv` |
| **Output** | `../Datasets/Context_Heal/...` |
| **Prompt change** | Adversarial suffix **+** unsafe-turn command |
| **Backend** | HuggingFace `device_map='auto'` |
| **Use when** | Stronger adversarial stress test (suffix + explicit unsafe directive) |

---

### `DBL_heal_generation.py`

**DBL detect-file healing** — regenerates responses for rows flagged Unsafe by the DBL filtering pipeline.

| | |
|---|---|
| **Input** | `../Datasets/DBL_data/Detect_Files/{path}.csv` |
| **Output** | `../Datasets/Context_Heal/DBL_data/{model}_{ts}_category_{cat}_type_{type}_{path}_heal_dataset.{csv,json}` |
| **Filter** | `prediction == "Unsafe"` |
| **Extra columns** | CSV includes `uid`, `flags` (from DBL detect schema) |
| **Backend** | `DSPipeline` |
| **Use when** | Healing DBL adversarial injection detect exports (Idea2 detect files) |

```bash
python DBL_heal_generation.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --path adv-BB400M \
    --category 1 \
    --type 1
```

Example `{path}` values: `adv-BB400M`, `adv-DD-BART`, `adv-backdoor-BB400M`, `adv-backdoor-DD-BART`.

## Script comparison

| Script | Input source | Output dir | Row filter | Inference |
|---|---|---|---|---|
| `...-inference.py` | `LM_Detect/*.csv` | `Context_Heal/` | none | HF |
| `...-inference_deepspeed.py` | `LM_Detect/*.csv` | `Context_Heal/` | none | DeepSpeed |
| `...-inference_classifier.py` | `Classifier/*.json` | `Classifier/Context_Heal/` | none | DeepSpeed |
| `...-inference_classifier_filter.py` | `Classifier/*.json` | `Classifier/Context_Heal/` | Safe only | DeepSpeed |
| `...-inference_adversarial.py` | `LM_Detect/*.csv` | `Context_Heal/` | none + adv suffix | HF |
| `...-inference_adversarial_add_command.py` | `LM_Detect/*.csv` | `Context_Heal/` | none + adv + unsafe cmd | HF |
| `DBL_heal_generation.py` | `DBL_data/Detect_Files/*.csv` | `Context_Heal/DBL_data/` | Unsafe only | DeepSpeed |

## Categories

Both `--category 1` and `--category 2` use the same instruction text today; the flag tags outputs and selects experiment metadata.

| Category | Typical `--path` examples |
|---|---|
| **1** | `Toxic-Category1`, `Benign-PersonaChat`, `Benign-PersonaChat_Delta_Category1`, `adv-BB400M` |
| **2** | `Toxic-Category2`, `Benign-PersonaChat_Delta_Category2`, `adv-DD-BART` |

## CLI reference

### Required (all generation scripts)

| Flag | Description |
|---|---|
| `--model` | HuggingFace model id (e.g. `meta-llama/Llama-2-13b-chat-hf`; underscores in name are converted to `/`) |
| `--path` | Dataset stem — resolved to the input path shown in the script table above (without `_dataset` suffix) |

### Common optional

| Flag | Default | Description |
|---|---|---|
| `--category` | — | `1` or `2` |
| `--type` | `no` | Prompt variant; use `1` for the standard empathetic instruction |
| `--batch_size` | `1` | Generation batch size |
| `--max_tokens` | `1024` | KV-cache / context token budget |
| `--max_new_tokens` | `50` | Max tokens to generate |
| `--dtype` | `float16` | `float32`, `float16`, or `int8` |
| `--greedy` | off | Greedy decoding instead of sampling |
| `--trust_remote_code` | off | Pass through to HuggingFace loaders |

### DeepSpeed-only (`utils.DSPipeline` scripts)

| Flag | Description |
|---|---|
| `--hf_baseline` | Disable DeepSpeed; use plain HF load |
| `--use_kernel` | Enable DeepSpeed kernel injection |
| `--use_meta_tensor` | Initialize model via meta tensors + checkpoint |
| `--checkpoint_path` | Local checkpoint directory |
| `--local_rank` / `--world_size` | Distributed inference ranks |

## Output schema

### Standard CSV columns

`context`, `response`, `category`, `label`, `implicit`, `source`, `index`, `heal_generated`

### DBL CSV (additional)

`uid`, `flags`

### Output filename pattern

```
{model_with_underscores}_{YYYYMMDD-HHMMSS}_category_{1|2}_type_{type}_{path}_heal_dataset.csv
```

Example:

```
meta-llama_Llama-2-13b-chat-hf_20240209-044944_category_1_type_1_Toxic-Category1_heal_dataset.csv
```

## Pipeline placement

```
LM_Detect / Classifier / DBL Detect_Files
        │
        ▼
  Healing_creation/          ← this module
        │
        ├── Context_Heal/                    (general LM healing)
        ├── Classifier/Context_Heal/           (classifier heal CSVs — 14 files used downstream)
        └── Context_Heal/DBL_data/             (DBL heal exports)
        │
        ▼
Data_processor/classifier/generate_classifier_focal_full_heal.py
        │
        ▼
Evaluation/RTR_Evaluation_Classifier_Heal/
```

**Do not copy** heal CSVs from `chatsec2`; regenerate them with the scripts above so paths and schemas match Optimus.

## Dependencies

`torch`, `transformers`, `datasets`, `deepspeed`, `peft`, `sklearn`, `tqdm`, `huggingface_hub`

Models are loaded from HuggingFace Hub (or `--checkpoint_path`) — ensure GPU memory and HF credentials are available for LLaMA-2 weights.
