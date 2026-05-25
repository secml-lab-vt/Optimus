# PromptAttack

Generate **adversarial paraphrases** of toxic conversational responses to evade LM-based toxicity detectors. Uses nine perturbation strategies (typos, synonym swap, paraphrase, …) and local LLM inference to produce `Adv-Toxic-Category*` CSVs consumed by [LM_Toxic_detect](../LM_Toxic_detect/README.md) and [Injection_code](../Injection_code/README.md) (`--adversarial True`).

## Directory layout

```
PromptAttack/
├── README.md
├── Adversarial_LLM_inference_final.py          # Idea1: run attack + write adversarial CSV
├── Adversarial_LLM_inference_idea2_final.py      # Idea2: same with Idea2 rubric
├── Adversarial_PromptAttack_process.py           # Idea1 prompt construction
├── Adversarial_PromptAttack_process_idea2.py     # Idea2 prompt construction
├── PromptAttack_process.py                       # Simple response-only attack prompts
├── File_formatter.py                               # Convert adversarial CSV → LM_Detect format
├── PromptAttack.py                                 # Original PromptAttack (GLUE; needs Call.py)
├── Predict.py                                      # Label predictor helper (original paper)
├── robustness_eval.py                              # GLUE robustness eval (original paper)
├── Dataset.py                                      # PyTorch dataset wrapper
├── requirements.txt
└── chatbot_adv.yml                                 # Conda environment spec
```

---

## Role in Optimus

```
Data_processor/lm_detect/  →  Datasets/LM_Detect/Toxic-Category*_dataset.csv
        │
        ▼
PromptAttack/Adversarial_LLM_inference_*.py
        │
        ▼
Datasets/Adversarial_attack/{model}_{ts}_*_adversarial.csv
        │
        ▼
File_formatter.py  →  Datasets/LM_Detect/Adv_Toxic-Category*_dataset.csv
        │
        ▼
LM_Toxic_detect/*_adversarial.py  →  score adversarial rows
        │
        ▼
Injection_code (--adversarial True)
```

---

## Idea1 vs Idea2

| | **Idea1** | **Idea2** |
|---|---|---|
| **Process module** | `Adversarial_PromptAttack_process.py` | `Adversarial_PromptAttack_process_idea2.py` |
| **Inference script** | `Adversarial_LLM_inference_final.py` | `Adversarial_LLM_inference_idea2_final.py` |
| **Attack rubric** | Category-specific toxicity definitions (rudeness vs harm/bias) | “Safe to generate next turn?” framing |
| **Downstream detect** | `LM_Toxic_detect/idea1/ModelFactory_adversarial.py` | `LM_Toxic_detect/idea2/ModelFactory_Idea2_adversarial.py` |

Both scripts read from `../Datasets/LM_Detect/{path}_dataset.csv` and write JSON + CSV under `../Datasets/Adversarial_attack/`.

---

## Scripts

### Optimus pipeline (primary)

| Script | Purpose |
|---|---|
| `Adversarial_PromptAttack_process.py` | Build 9 attack prompts per row (Idea1 LM-detect rubric) |
| `Adversarial_PromptAttack_process_idea2.py` | Same with Idea2 safe-continuation rubric |
| `Adversarial_LLM_inference_final.py` | Run local LLM on attack prompts; filter by BERTScore; write adversarial CSV |
| `Adversarial_LLM_inference_idea2_final.py` | Idea2 variant of inference |
| `PromptAttack_process.py` | Lightweight processor (response-only attacks; debug / preview prompts) |
| `File_formatter.py` | Rename `adversarial_response` → `response`; export `Adv_Toxic-Category*_dataset.csv` to `LM_Detect/` |

### Original PromptAttack paper code (optional)

| Script | Purpose |
|---|---|
| `PromptAttack.py` | GPT-driven prompt attacks on GLUE tasks |
| `Predict.py` | Query LLM and parse label from response |
| `robustness_eval.py` | End-to-end attack evaluation on SST-2, QQP, MNLI, … |
| `Dataset.py` | Custom PyTorch dataset for GLUE |

These require `Call.py` and `info/{dataset}_info.pkl` from the upstream [PromptAttack](https://github.com/microsoft/promptbench) repo — not bundled here.

---

## Perturbation strategies

Each row gets **9 attack prompts**, one per instruction in `PromptAttack_process.perturbation_instruction`:

1. Typo two words  
2. Change two letters  
3. Add extraneous characters  
4. Synonym replacement  
5. Delete non-essential words  
6. Add neutral words  
7. Append meaningless handle (`@fasuv3`)  
8. Paraphrase  
9. Change syntactic structure  

The inference script runs all nine variants, computes **BERTScore** fidelity vs the original response, and keeps the best-scoring adversarial paraphrase per row.

---

## Quick start

**Prerequisites**

- Input CSV from [Data_processor/lm_detect](../Data_processor/lm_detect/generate_lm_detect.py): `../Datasets/LM_Detect/Toxic-Category1_dataset.csv`
- Local model weights (LLaMA-2, etc.) accessible to DeepSpeed inference
- `utils.py` with `DSPipeline` class — scripts import `from utils import DSPipeline`. A compatible implementation lives in [Healing_creation/utils.py](../Healing_creation/utils.py); copy or symlink into `PromptAttack/` if missing.

**Step 1 — Generate adversarial responses (Idea1):**

```bash
cd PromptAttack
python Adversarial_LLM_inference_final.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --path Toxic-Category1 \
    --category 1 \
    --type 1 \
    --max_new_tokens 50
```

**Step 2 — Convert to LM_Detect format:**

Edit `File_formatter.py` to point at your output CSV, then:

```bash
python File_formatter.py
# → ../Datasets/LM_Detect/Adv_Toxic-Category1_dataset.csv
```

**Step 3 — Score adversarial rows:**

```bash
cd ../LM_Toxic_detect/idea1
python ModelFactory_adversarial.py \
    --model_name meta-llama_Llama-2-13b-chat-hf \
    --path Toxic-Category1 \
    --category 1 --few_shot no --type 1
```

For Idea2, use `Adversarial_LLM_inference_idea2_final.py` and `LM_Toxic_detect/idea2/ModelFactory_Idea2_adversarial.py`.

---

## Output format

Adversarial inference writes:

| File | Location |
|---|---|
| JSON (prompts + metadata) | `../Datasets/Adversarial_attack/{model}_{ts}_category_{cat}_type_{type}_{path}_adversarial.json` |
| CSV (9 variants + BERTScores) | `../Datasets/Adversarial_attack/{model}_{ts}_category_{cat}_type_{type}_{path}_adversarial.csv` |

CSV columns include `context`, `response`, `label`, plus `adversarial_response_{0..8}`, `bertscore_{0..8}`, `word_modification_ratio_{0..8}`.

After `File_formatter.py`, the best adversarial response replaces `response` in a standard `LM_Detect/*_dataset.csv` layout.

---

## Common CLI (`Adversarial_LLM_inference_*.py`)

| Flag | Description |
|---|---|
| `--model` | HuggingFace model id (e.g. `meta-llama/Llama-2-13b-chat-hf`) |
| `--path` | Dataset stem → `../Datasets/LM_Detect/{path}_dataset.csv` |
| `--category` | `1` or `2` |
| `--type` | Prompt type index (default `1`) |
| `--max_new_tokens` | Generation length per attack (default 50) |
| `--batch_size` | Inference batch size (default 1) |
| `--hf_baseline` | Disable DeepSpeed kernel injection |
| `--checkpoint_path` | Optional LoRA / fine-tuned checkpoint |

---

## Environment

```bash
conda env create -f chatbot_adv.yml
conda activate chatbot_adv   # or install from requirements.txt
pip install -r requirements.txt
```

Key deps: `torch`, `transformers`, `deepspeed`, `datasets`, `bert-score`, `nltk`, `peft`.

---

## Dependencies / upstream

- **Input data:** [Data_processor/lm_detect](../Data_processor/lm_detect/)
- **Scoring:** [LM_Toxic_detect](../LM_Toxic_detect/README.md)
- **Training with adversarial filter:** [Injection_code](../Injection_code/README.md) (`--adversarial True`)
- **Original paper code:** [PromptAttack / PromptBench](https://github.com/microsoft/promptbench)
