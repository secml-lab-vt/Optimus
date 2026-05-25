# Optimus

Source code, models, and datasets for **[Optimus: A Robust Defense Framework for Mitigating Toxicity while Fine-Tuning Conversational AI](https://arxiv.org/pdf/2507.05660v3)** (CODASPY 2026).

Optimus mitigates toxicity learned from **untrusted fine-tuning data** while preserving conversational utility. The framework has two main stages ([Figure 2](https://arxiv.org/pdf/2507.05660v3)): **(1) toxicity classification** and **(2) fine-tuning + alignment** (healing → SFT → DPO). Code modules below map directly to those stages.

### Quick start

1. **Clone the repo** and download the asset bundle (see [dataset download](#dataset-download) and [`Optimus_models+datasets/README.md`](Optimus_models+datasets/README.md)).
2. **Place `Optimus_models+datasets/` inside the clone** and run setup — it moves assets into the standard repo directories:

```bash
git clone <repo-url> Optimus && cd Optimus
# extract/download Optimus_models+datasets/ into this directory
bash setup_optimus.sh
export OPTIMUS_ROOT="$(pwd)"   # optional; auto-detected if unset
```

3. **GRADE metric** — code is at [`GRADE/`](GRADE/README.md); data, tools, and checkpoint go directly under [`GRADE/`](GRADE/README.md). See [GRADE setup](#grade-metric-setup).
4. **Create environments** — three envs in [`environments/`](environments/README.md):
  ```bash
   cd environments

   # Main stack (Stages 0–4, RTR eval, baselines)
   conda create -n llm_clone python=3.10 -y && conda activate llm_clone
   conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements-llm_clone.txt

   # §7 adaptive attacks (optional)
   conda create -n chatbot_adv_deep python=3.10 -y && conda activate chatbot_adv_deep
   conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements-chatbot_adv_deep.txt

   # GRADE metric only (legacy TF 1.14 — keep separate)
   conda create -n grade_env1 python=3.6.13 -y && conda activate grade_env1
   pip install -r requirements-grade_env1.txt
  ```
4. **Minimal reproduction** — use precomputed score/heal CSVs and skip Stages 1–2a; see [Injection_code/standard](Injection_code/README.md#two-step-optimus-pipeline) for the two-step `train_bot` → `train_bot_DPO` flow.


| Conda env          | Requirements                                                                        | Used for                                        |
| ------------------ | ----------------------------------------------------------------------------------- | ----------------------------------------------- |
| `llm_clone`        | [requirements-llm_clone.txt](environments/requirements-llm_clone.txt)               | Main pipeline (Stages 0–4), RTR eval, baselines |
| `chatbot_adv_deep` | [requirements-chatbot_adv_deep.txt](environments/requirements-chatbot_adv_deep.txt) | §7 attacks (`PromptAttack`, `AmpleGCG`, GCG)    |
| `grade_env1`       | [requirements-grade_env1.txt](environments/requirements-grade_env1.txt)             | GRADE metric (`Injection_code/tools/GRADE.sh`)  |


See **[environments/README.md](environments/README.md)** for setup details.

---

## Stages and code modules


| Stage                           | Paper reference                        | What it does                                                                                                                                           | Code                                                                                                                               | Key `Datasets/` paths                                                                                      |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **0 — Dataset prep**            | §4 (PersonaChat, BAD, CADD, DiaSafety) | Normalize raw corpora; build Category1 (offensive) & Category2 (specialized) splits; export LM-detect and classifier JSONs                             | [Data_processor](Data_processor/README.md)                                                                                         | `Raw_data/`, `Processed_datasets/Final_Dataset/`, `Benign/`, `Toxic/`, `LM_Detect/`, `Classifier/`         |
| **1 — Toxicity classification** | §3.2 Stage 1, §5 (Refusal approach)    | Score each context–response pair; label toxic if P("no") ≥ 0.5. Supports LLM refusal (Idea1/Idea2), Perspective, OpenAI, Unitary, soft-prompt baseline | [LM_Toxic_detect](LM_Toxic_detect/README.md), [toxic-prompt](toxic-prompt/README.md)                                               | `LM_Scores/Fewshot-no/training_datasets/` (Idea1), `Advanced_Detect/Fewshot-no/training_datasets/` (Idea2) |
| **2a — Healing data**           | §3.2 Step 1 (NH / CH)                  | Replace flagged-toxic responses with safe alternatives. **NH** = single fixed canned reply (inline); **CH** = LLM-generated contextual heal            | [Healing_creation](Healing_creation/README.md) (CH), [Injection_code/baseline](Injection_code/README.md#baseline--comparison) (NH) | `Context_Heal/`, `Classifier/Context_Heal/`                                                                |
| **2b — Fine-tuning (FT-Heal)**  | §3.2 Step 2, §6.4                      | SFT/LoRA on benign + filtered toxic + healed pairs                                                                                                     | [Injection_code](Injection_code/README.md) → `train_bot*.py`                                                                       | Reads score CSVs + heal CSVs; writes `Models/Custom/model_runs/`                                           |
| **3 — DPO alignment**           | §3.2 Step 3, §6.5                      | Preference pairs (toxic vs healed response, same context) steer model toward safety                                                                    | [Injection_code](Injection_code/README.md) → `train_bot_DPO*.py`                                                                   | Checkpoint from Step 2                                                                                     |
| **4 — Evaluation**              | §4, §6.2 (RTR, PPL, FBD, GRADE)        | Measure response toxicity rate and conversational utility after defense                                                                                | [Evaluation](Evaluation/RTR_Evaluation_Classifier/codes/), [Injection_code/metrics](Injection_code/README.md#metrics--tools)       | `Evaluation/`, `Processed_datasets/Classifier/`, `inject_eval*/Injected_evaluation.json`                   |


**Paper ↔ code naming**


| Paper term                        | Code / data                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Offensive category                | Category 1 — `Toxic/Category1/`, `Toxic-Category1`                                                   |
| Specialized category (TA, BO, RI) | Category 2 — `Toxic/Category2/`, `Toxic-Category2`                                                   |
| Refusal classifier                | [LM_Toxic_detect/idea2](LM_Toxic_detect/idea2/) (primary); Idea1 = detailed rubric variant           |
| Non-contextual healing (NH)       | `Injection_code/baseline/` — replaces filtered-toxic responses with a fixed canned reply (see below) |
| Contextual healing (CH)           | `--healing_dataset Context_Heal` + [Healing_creation](Healing_creation/)                             |
| Filter-only ablation              | Injection with heal disabled / filter threshold only (§6.3)                                          |
| FT-Heal ablation                  | `train_bot` without DPO (§6.4)                                                                       |
| Full Optimus                      | `train_bot` → `train_bot_DPO` (§6.5)                                                                 |


---

## Optional experiments

### Adaptive attacks (§7)

Attacks target specific pipeline stages; defense code paths live in `Injection_code/` (`--adversarial`, `jail/`, `heal_jail/`).


| Attack (paper)                                                                    | Target stage               | Code                                                                                                                                   |
| --------------------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [PromptAttack](https://arxiv.org/abs/2305.11256) adversarial paraphrases (§7.1.1) | Stage 1 classifier         | [PromptAttack](PromptAttack/README.md) → `Adv_Toxic-Category`*                                                                         |
| Manually-designed jailbreak (§7.1.2)                                              | Stage 1 classifier         | [Data_processor/adaptive-attacks](Data_processor/README.md#optional--adaptive-attacks) → `Toxic/JA*-Toxic-Category`*                   |
| Optimization-based jailbreak / GCG-style (§7.1.2)                                 | Stage 1 classifier         | [Universal-Prompt-Injection](Universal-Prompt-Injection/README.md)                                                                     |
| [AmpleGCG-Plus](https://arxiv.org/abs/2410.01288) on healing LLM (§7.2)           | Stage 2a healing generator | [AmpleGCG](AmpleGCG/README.md)                                                                                                         |
| Dialog-based learning adaptive attacks (§7.3)                                     | Post-deployment retraining | [Data_processor/dbl](Data_processor/README.md#optional--dbl), [DBL-Filtering](DBL-Filtering/README.md), [DBL_data](Datasets/DBL_data/) |


### Comparison baselines (§6.7)


| Baseline                                    | Paper  | Code                                                     |
| ------------------------------------------- | ------ | -------------------------------------------------------- |
| StarDSS                                     | §6.7.2 | [star-dss](star-dss/README.md)                           |
| You Only Prompt Once (prefix-tuning filter) | §6.7.1 | [toxic-prompt](toxic-prompt/README.md)                   |
| HateGuard-style CoT filter                  | §6.7.1 | [LM_Toxic_detect/specialized](LM_Toxic_detect/README.md) |


---

## All modules


| Module                                                             | Role in pipeline                                         |
| ------------------------------------------------------------------ | -------------------------------------------------------- |
| [Data_processor](Data_processor/README.md)                         | Stage 0 — ETL, splits, classifier/eval/LM-detect exports |
| [LM_Toxic_detect](LM_Toxic_detect/README.md)                       | Stage 1 — Refusal + API toxicity scoring                 |
| [Healing_creation](Healing_creation/README.md)                     | Stage 2a — Contextual heal generation                    |
| [Injection_code](Injection_code/README.md)                         | Stages 2b–3 — SFT, DPO, inject_eval, utility metrics     |
| [GRADE](GRADE/README.md)                                           | Stage 4 — dialogue coherence metric (via `tools/GRADE.sh`) |
| [Evaluation](Evaluation/)                                          | Stage 4 — RTR BERT classifiers                           |
| [PromptAttack](PromptAttack/README.md)                             | §7.1.1 adversarial attack on classifier                  |
| [Universal-Prompt-Injection](Universal-Prompt-Injection/README.md) | §7.1.2 optimization jailbreak                            |
| [AmpleGCG](AmpleGCG/README.md)                                     | §7.2 attack on healing generator                         |
| [DBL-Filtering](DBL-Filtering/README.md)                           | §7.3 DBL detect scoring                                  |
| [toxic-prompt](toxic-prompt/README.md)                             | Comparison baseline filter (§6.7.1)                      |
| [star-dss](star-dss/README.md)                                     | Comparison baseline defense (§6.7.2)                     |


---

## GRADE metric setup

[GRADE](https://arxiv.org/abs/2010.03994) scores dialogue coherence during Stage 4 utility evaluation (`Injection_code/metrics/test_utils*.py`).

| Location | Role |
| -------- | ---- |
| [`GRADE/`](GRADE/) | Source code plus `data/`, `tools/`, and `output/` (checkpoint) |

**Layout:** place GRADE assets directly under `Optimus/GRADE/data`, `tools`, and `output`. Use **`Optimus/GRADE/`** for all runs. `Chatbot-Toxicity-Injection/GRADE/` may symlink to the same tree for legacy CTI scripts (`setup_optimus.sh` creates this if missing).

**Checkpoint (included in download bundle):**

```
GRADE/output/71/GRADE_K2_N10_N10/model_eval_best_71.ckpt
```

If setting up from scratch, download the bundle into `Optimus_models+datasets/` and run `bash setup_optimus.sh` (see [Optimus_models+datasets/README.md](Optimus_models+datasets/README.md#grade-setup)).

Install texar and run inference via `Injection_code/tools/GRADE.sh` with the `grade_env1` conda env. Paths are resolved from script location (not the shell cwd). Run `python GRADE/verify_paths.py` to check checkpoint and keyword assets.

RTR classifier checkpoints and logs live under `Evaluation/RTR_Evaluation_Classifier/` (and `_Heal` variant). Run `python Evaluation/verify_paths.py` to validate.

---

## Dataset download

Download the dataset archives into `Optimus_models+datasets/`, then run `bash setup_optimus.sh` to move them into `Optimus/Datasets/` (see [`Optimus_models+datasets/README.md`](Optimus_models+datasets/README.md)):

1. [Dataset Path 1](https://drive.google.com/file/d/1I39gfBYowg-tApYULI8zFF8jIGVmrpkY/view?usp=drive_link)
2. [Dataset Path 2](https://drive.google.com/file/d/1o0ZxRCtmSikfigos__M1JKm5ErQMTIDX/view?usp=sharing)

### `Datasets/` layout (high level)


| Directory                                   | Stage | Contents                                       |
| ------------------------------------------- | ----- | ---------------------------------------------- |
| `Raw_data/`                                 | 0     | BAD, CADD, DiaSafety, PersonaChat, DailyDialog |
| `Processed_datasets/`                       | 0     | Normalized JSON + classifier/eval splits       |
| `Benign/`, `Toxic/`                         | 0     | Safe/Unsafe CSVs for injection                 |
| `LM_Detect/`                                | 0→1   | LM-detect input CSVs                           |
| `LM_Scores/`, `Advanced_Detect/`            | 1     | Toxicity score CSVs for filtering              |
| `Context_Heal/`, `Classifier/Context_Heal/` | 2a    | Contextual heal CSVs (CH)                      |
| `Evaluation/`                               | 4     | Utility eval CSVs                              |
| `DBL_data/`                                 | §7.3  | DBL merge/detect/heal pipeline artifacts       |


Precomputed score and heal files are included for many experiments so Stages 1–2a can be skipped when reproducing paper results.

**NH canned response** (non-contextual healing): flagged-toxic pairs are rewritten to:

> I'm sorry, I'm not sure what to say. Thank you for sharing and talking to me though.

Applied inline in `Injection_code/baseline/train_bot_baseline.py` (and DPO counterpart); not loaded from `Datasets/Healing/`.

---

## Citation

```bibtex
@inproceedings{cheruvu2026optimus,
  title={Optimus: A Robust Defense Framework for Mitigating Toxicity while Fine-Tuning Conversational AI},
  author={Cheruvu, Aravind and Kanchi, Shravya and Abdullah, Sifat Muhammad and Kong, Nicholas Ka-Shing and Yao, Daphne and Jadliwala, Murtuza and Viswanath, Bimal},
  booktitle={Proceedings of the Sixteenth ACM Conference on Data and Application Security and Privacy (CODASPY)},
  year={2026}
}
```

