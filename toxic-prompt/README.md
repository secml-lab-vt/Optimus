# Toxic Prompt (You Only Prompt Once)

Official implementation of [**You Only Prompt Once: On the Capabilities of Prompt Learning on Large Language Models to Tackle Toxic Content**](https://arxiv.org/abs/2308.05596) (IEEE S&P 2024).

Uses **soft prompt tuning** (OpenPrompt) on T5-family models for three toxicity tasks: classification, span detection, and detoxification. Within Optimus, trained prompts and `Filtering_Results.py` support the **"You Prompt Once"** LM-detect baseline that writes scores to `Datasets/you-prompt-only/`.

---

## Directory layout

```
toxic-prompt/
├── README.md
├── 1_toxicity_classification.py      # Task 1: train/eval classification prompts
├── 2_and_3_toxic_generation.py       # Task 2 & 3: span detection + detoxification
├── 2_calculate_span.py               # Task 2 evaluation (span F1)
├── 3_perspective_evaluation.py       # Task 3 evaluation (toxicity change)
├── model_inference.py                # Inference on trained prompts
├── Filtering_Results.py              # Optimus: aggregate scores → filter CSVs
├── perspective_api.py                # Perspective API helper
├── Filtering_Results.py
├── open_pkl.py                         # Inspect saved prompt checkpoints
├── inference-script.py                 # Batch inference utility
├── environment.yaml                    # Conda environment
├── experiment_scripts/soft_template/   # Manual template / verbalizer text
├── parsed_dataset/                     # Preprocessed benchmark datasets (JSON)
└── sfs_out/                            # Training outputs (CSVs, logs)
```

---

## Role in Optimus

```
toxic-prompt (train + infer)
        │
        ▼
Datasets/you-prompt-only/  …  *_toxicity_scores.csv
        │
        ▼
Filtering_Results.py  →  Results/Filtering_Results_Idea{1,2}_*.csv
        │
        ▼
Injection_code (--model_vers …)  threshold selection / comparison baseline
```

Upstream inputs can also come from [Data_processor/lm_detect](../Data_processor/lm_detect/) (`../Datasets/LM_Detect/`). The `Filtering_Results.py` script reads score CSVs from `../Datasets/you-prompt-only/` and produces filtered result summaries for Idea1/Idea2 and adversarial variants.

---

## Environment setup

```bash
conda env create --file environment.yaml
conda activate toxic_prompt
```

Requires **OpenPrompt** and HuggingFace `transformers`. Model weights for reproduction can be downloaded separately (see [Model weights](#model-weights)).

---

## Datasets

Preprocessed JSON files live in `parsed_dataset/`:

| Task | Datasets |
|---|---|
| **Task 1 — Classification** | HateXplain, USElectionHate20, HateCheck, SBIC.v2, measuring-hate-speech |
| **Task 2 — Span detection** | TSD |
| **Task 3 — Detoxification** | Parallel, Paradetox |

Each file is Perspective-balanced or adversarially perturbed (suffix `_adv_`, `_adv_perturb_`, etc.).

---

## Tasks and scripts

### Task 1 — Toxicity classification

Train soft prompts on a PLM; evaluate accuracy / F1 on hate-speech benchmarks.

```bash
python 1_toxicity_classification.py \
    --plm_eval_mode \
    --model t5 \
    --model_name_or_path t5-small \
    --dataset HateXplain
```

| Flag | Description |
|---|---|
| `--plm_eval_mode` | Disable dropout in frozen PLM (recommended for eval) |
| `--model` | `t5` or `t5-lm` |
| `--model_name_or_path` | HF model id (e.g. `t5-small`) |
| `--dataset` | One of the Task 1 dataset names |
| `--soft_token_num` | Number of soft prompt tokens (default 20) |
| `--prompt_lr` | Prompt learning rate (default 0.3) |
| `--max_steps` | Training steps (default 2000) |

### Task 2 — Toxic span detection

Shared generation script with Task 3:

```bash
python 2_and_3_toxic_generation.py \
    --plm_eval_mode \
    --model t5 \
    --model_name_or_path t5-small \
    --dataset TSD
```

Evaluate span overlap:

```bash
python 2_calculate_span.py --file_path sfs_out/task23/TSD_t5-small_True.txt
```

Span baselines follow [toxic-span](https://github.com/ipavlopoulos/toxic_spans).

### Task 3 — Detoxification

Same training script as Task 2, different dataset:

```bash
python 2_and_3_toxic_generation.py \
    --plm_eval_mode \
    --model t5 \
    --model_name_or_path t5-small \
    --dataset Parallel
```

Evaluate toxicity reduction via Perspective API:

```bash
python 3_perspective_evaluation.py \
    --file_path sfs_out/task23/Parallel_t5-small_True.txt \
    --key YOUR_PERSPECTIVE_API_KEY
```

Detox baselines and BLEU/SIM/PPL metrics follow [paradetox](https://github.com/s-nlp/paradetox).

---

## Optimus integration

### Model inference on LM_Detect CSVs

```bash
python model_inference.py \
    --plm_eval_mode \
    --model t5 \
    --model_name_or_path t5-small \
    --dataset SBIC.v2

# Custom Optimus dataset path
python model_inference.py \
    --plm_eval_mode \
    --model t5 \
    --model_name_or_path t5-small \
    --path ../Datasets/LM_Detect/Toxic-Category1_dataset.csv
```

### Filtering score CSVs

Aggregate latest per-model toxicity scores and apply threshold filtering:

```bash
python Filtering_Results.py --idea 1 --adversarial False --threshold 0.5
python Filtering_Results.py --idea 2 --adversarial True --threshold 0.8
```

| Flag | Description |
|---|---|
| `--idea` | `1` or `2` — selects score directory and filename pattern |
| `--adversarial` | `True` / `False` — use `Adv_Toxic-Category*` files |
| `--threshold` | Score cutoff (`0.5`, `0.8`, `0.9`, `0.95`) |

**Idea1** reads from `../Datasets/you-prompt-only/` and writes `Results/Filtering_Results_Idea1_type_0_{adversarial}.csv`.

---

## Supporting scripts

| Script | Purpose |
|---|---|
| `model_inference.py` | Run trained soft prompts on a dataset or custom CSV |
| `Filtering_Results.py` | Optimus pipeline: pick latest scores, filter by threshold, export summary CSV |
| `perspective_api.py` | Perspective API scoring helper |
| `open_pkl.py` | Inspect pickled prompt / result files |
| `inference-script.py` | Additional batch inference entry point |

---

## Outputs

| Path | Contents |
|---|---|
| `sfs_out/` | Training logs, classification results, generation outputs |
| `sfs_out/task23/` | Task 2/3 generation `.txt` files for evaluation |
| `saved_models/` | Trained prompt checkpoints (gitignored; download separately) |
| `../Datasets/you-prompt-only/` | Optimus LM-detect score CSVs from inference runs |

---

## Model weights

Download trained prompt weights and place in `saved_models/` (gitignored):

[百度网盘 link](https://pan.baidu.com/s/1HUG58q4pkDsyxpt3hkl-Gw?pwd=smu8) (password: `smu8`)

---

## End-to-end flow (Optimus)

```
Data_processor/lm_detect/  →  Datasets/LM_Detect/*.csv
        │
        ▼
toxic-prompt/model_inference.py  →  Datasets/you-prompt-only/*_toxicity_scores.csv
        │
        ▼
Filtering_Results.py  →  filtered summary CSVs
        │
        ▼
Injection_code  (compare vs Perspective / Idea2 LM-detect)
```

Related modules: [LM_Toxic_detect](../LM_Toxic_detect/README.md), [PromptAttack](../PromptAttack/README.md), [Data_processor/lm_detect](../Data_processor/lm_detect/).

---

## Citation

```bibtex
@inproceedings{HZSZ224,
  author = {Xinlei He and Savvas Zannettou and Yun Shen and Yang Zhang},
  title = {{You Only Prompt Once: On the Capabilities of Prompt Learning on Large Language Models to Tackle Toxic Content}},
  booktitle = {{IEEE Symposium on Security and Privacy (S\&P)}},
  publisher = {IEEE},
  year = {2024}
}
```
