# Universal Prompt Injection

Optimize a **universal adversarial suffix** (learned token sequence) that jailbreaks a target LLM when appended to toxicity-detection prompts. Used to stress-test Idea2 LM-detect prompts from [LM_Toxic_detect](../LM_Toxic_detect/README.md) before downstream [Injection_code](../Injection_code/README.md) training.

Based on gradient-based prompt injection (GCG-style optimization with momentum).

## Directory layout

```
Universal-Prompt-Injection/
├── README.md
├── file_preprocess.py              # Step 1: Advanced_Detect JSON → CSV
├── universal_prompt_injection.py   # Step 2: optimize + evaluate suffix attack
├── models/
│   └── download_models.py          # Download target LLM weights locally
├── utils/
│   ├── opt_utils.py                # Gradient, sampling, loss helpers
│   └── string_utils.py             # SuffixManager, conversation templates
├── data/                           # Preprocessed inputs (included samples)
│   ├── processed_data-category1-idea2.csv
│   ├── processed_data-category2-idea2.csv
│   └── …
└── results/eval/                   # Attack logs and success metrics
    └── {model}/{injection}/momentum_{m}/token_length_{n}/target_{t}/
```

---

## Role in Optimus

```
LM_Toxic_detect/idea2/  →  Datasets/Advanced_Detect/Fewshot-no/…/*_adv_dataset.json
        │
        ▼
file_preprocess.py  →  data/processed_data-category{1,2}-idea2.csv
        │
        ▼
universal_prompt_injection.py  →  results/eval/{model}/…/
        │
        ▼
(Optional) re-score attacked prompts / evaluate detector robustness
```

Preprocessed CSVs in `data/` are derived from Idea2 advanced-detect JSON exports (category 1 and 2, Idea1/Idea2 variants).

---

## Two-step workflow

### Step 1 — Preprocess LM-detect JSON

`file_preprocess.py` reads an Idea2 advanced-detect JSON from `../Datasets/Advanced_Detect/` and extracts `(instruction, input, output)` tuples for the attack optimizer.

```bash
cd Universal-Prompt-Injection
python file_preprocess.py
```

Edit the script to set:

- `category = 1` or `2`
- `data_files=…` path to the source JSON under `../Datasets/Advanced_Detect/Fewshot-no/…`

Outputs:

| Setting | Output file |
|---|---|
| Category 1 | `data/processed_data-category1-idea2.csv` |
| Category 2 | `data/processed_data-category2-idea2.csv` |

Included sample CSVs cover category 1/2 × Idea1/Idea2 combinations.

### Step 2 — Run universal prompt injection

```bash
python universal_prompt_injection.py \
    --model llama2 \
    --dataset_path data/processed_data-category2-idea2.csv \
    --tokens 150 \
    --num_steps 8000 \
    --batch_size 32 \
    --momentum 1.0 \
    --start 0 \
    --end 5
```

---

## Scripts

| Script | Purpose |
|---|---|
| `file_preprocess.py` | Parse Idea2 LM-detect JSON prompts into flat CSV for optimization |
| `universal_prompt_injection.py` | Main attack: optimize adversarial suffix, test jailbreak success |
| `models/download_models.py` | Download and save target model + tokenizer to `./models/` |
| `utils/opt_utils.py` | Token gradients, candidate sampling, target loss, model loading |
| `utils/string_utils.py` | Conversation templates, suffix placement, generation queries |

---

## Key parameters

| Flag | Default | Description |
|---|---|---|
| `--model` | `llama2` | Target model key (see table below) |
| `--dataset_path` | `data/processed_data.csv` | Preprocessed attack dataset |
| `--tokens` | `150` | Adversarial suffix length (token count) |
| `--num_steps` | `8000` | Optimization steps |
| `--batch_size` | `32` | Candidates per step |
| `--topk` | `128` | Top-k token substitutions |
| `--momentum` | `1.0` | Momentum for gradient updates |
| `--target` | `0` | Target class index for loss |
| `--start` / `--end` | `0` / `5` | Dataset row slice |
| `--injection` | `static` | Injection mode label (used in output path) |
| `--save_suffix` | `normal` | Suffix tag appended to result filenames |
| `--device` | `0` | CUDA device index |

### Supported target models

| `--model` key | Local path |
|---|---|
| `llama2` | `./models/llama2/llama-2-7b-chat-hf` |
| `llama2-13b` | `./models/llama2/llama-2-13b-chat-hf` |
| `vicuna` | `./models/vicuna/vicuna-7b-v1.3` |
| `guanaco` | `./models/guanaco/guanaco-7B-HF` |
| `WizardLM` | `./models/WizardLM/WizardLM-7B-V1.0` |
| `mpt-chat` | `./models/mpt/mpt-7b-chat` |
| `mpt-instruct` | `./models/mpt/mpt-7b-instruct` |
| `falcon` | `./models/falcon/falcon-7b-instruct` |

Download weights first:

```bash
python models/download_models.py   # edit model_name / base_model_path as needed
```

---

## Results layout

```
results/eval/{model}/{injection}/momentum_{momentum}/token_length_{tokens}/target_{target}/
    {start}_{end}_{batch_size}_{save_suffix}.json
```

Each JSON log tracks per-step:

- `loss`, `suffix`, `time`, `respond`, `success`

Example paths included in repo:

- `results/eval/llama2/static/momentum_1.0/token_length_150/target_0/0_5_20_normal.json`
- `results/eval/llama2-13b/static/momentum_1.0/token_length_150/target_0/0_5_20_normal.json`

---

## Requirements

- GPU with sufficient VRAM for the target model (7B–13B)
- Source JSON datasets under `../Datasets/Advanced_Detect/` (from [LM_Toxic_detect/idea2](../LM_Toxic_detect/idea2/))
- Target model weights in `./models/` (via `download_models.py` or manual HuggingFace download)
- Python: `torch`, `transformers`, `pandas`, `tqdm`, `numpy`

---

## Related modules

| Module | Relationship |
|---|---|
| [LM_Toxic_detect](../LM_Toxic_detect/README.md) | Produces `Advanced_Detect/` JSON inputs |
| [Data_processor/lm_detect](../Data_processor/lm_detect/) | Upstream CSV/JSON generation |
| [Injection_code](../Injection_code/README.md) | Downstream training uses LM-detect score CSVs |
