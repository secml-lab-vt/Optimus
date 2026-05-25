# Optimus

Source code, models, and datasets for **[Optimus: A Robust Defense Framework for Mitigating Toxicity while Fine-Tuning Conversational AI](https://arxiv.org/pdf/2507.05660v3)** (CODASPY 2026).

Optimus mitigates toxicity learned from **untrusted fine-tuning data** while preserving conversational utility. The framework has two main stages ([Figure 2](https://arxiv.org/pdf/2507.05660v3)): **(1) toxicity classification** and **(2) fine-tuning + alignment** (healing → SFT → DPO). Code modules below map directly to those stages.

### Quick start

1. **Clone the repo** and **request the asset bundle** (see [dataset download](#dataset-download)).

2. **Place `Optimus_models+datasets/` inside the clone** and run setup — it moves assets into the standard repo directories:

```bash
git clone <repo-url> Optimus && cd Optimus
# after approval: extract Optimus_models+datasets/ into this directory
bash setup_optimus.sh
```

3. See **[environments/README.md](environments/README.md)** for setup details.

---

**Paper ↔ Code Terminology**

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

| Attack (paper)                                                                    | Target stage               | Code                                                                                                                                   |
| --------------------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [PromptAttack](https://arxiv.org/abs/2305.11256) adversarial paraphrases (§7.1.1) | Stage 1 classifier         | [PromptAttack](PromptAttack/README.md) → `Adv_Toxic-Category`*                                                                         |
| Manually-designed jailbreak (§7.1.2)                                              | Stage 1 classifier         | [Data_processor/adaptive-attacks](Data_processor/README.md#optional--adaptive-attacks) → `Toxic/JA*-Toxic-Category`*                   |
| Optimization-based jailbreak / GCG-style (§7.1.2)                                 | Stage 1 classifier         | [Universal-Prompt-Injection](Universal-Prompt-Injection/README.md)                                                                     |
| [AmpleGCG-Plus](https://arxiv.org/abs/2410.01288) on healing LLM (§7.2)           | Stage 2a healing generator | [AmpleGCG](AmpleGCG/README.md)                                                                                                         |
| Dialog-based learning adaptive attacks (§7.3)                                     | Post-deployment retraining | [Data_processor/dbl](Data_processor/README.md#optional--dbl), [DBL-Filtering](DBL-Filtering/README.md), [DBL_data](Datasets/DBL_data/) |


## Comparison baselines (§6.7)


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

## Dataset download

Datasets, pretrained models, GRADE assets, RTR checkpoints, and related files are **not** in this GitHub repo (~22 GB). Access is granted after you complete the request form:

**[Optimus: Datasets and Models Request Form](https://docs.google.com/forms/d/e/1FAIpQLSfBPIoHIok9qrPOfDvIi4PfCLVKl0RcaugycfStkauSQZT8Eg/viewform)**

Use an **academic email**, describe your research purpose, and accept the terms on the form. Approved requesters receive a download link for the `Optimus_models+datasets` bundle (typically a zip archive).

**THE DATASETS AND MODELS PROVIDED ARE NOT TO BE USED FOR MALICIOUS OR INAPPROPRIATE USE CASES.** Do not redistribute the download link or bundle without permission from the authors.

### After you receive the bundle

1. Extract the archive and place the `Optimus_models+datasets/` folder inside your Optimus clone.
2. Run `bash setup_optimus.sh` — it moves assets into the standard repo paths (see [`Optimus_models+datasets/README.md`](Optimus_models+datasets/README.md)).
3. Run `python GRADE/verify_paths.py`, `python Evaluation/verify_paths.py`, and `python Chatbot-Toxicity-Injection/verify_paths.py` to confirm layout.


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

