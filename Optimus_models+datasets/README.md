# Optimus models and datasets (access-controlled bundle)

Large assets are **not** shipped in the GitHub repo. Request access via the form linked from [`../README.md`](../README.md#dataset-download):

**[Optimus: Datasets and Models Request Form](https://docs.google.com/forms/d/e/1FAIpQLSfBPIoHIok9qrPOfDvIi4PfCLVKl0RcaugycfStkauSQZT8Eg/viewform)**

Approved requesters receive a download link (typically `Optimus_models+datasets.zip`). Extract it, place this folder inside your Optimus clone, then run:

```bash
cd Optimus
bash setup_optimus.sh
```

`setup_optimus.sh` **moves** files from this bundle into the standard repo paths below. After setup, runtime code reads assets directly from those directories.

**Use restrictions:** research and non-commercial purposes only. Do not share the download link or redistribute the bundle without author permission. The data may contain offensive or toxic language.

## Bundle layout → repo destination

| Path in this download | Moved to |
|-----------------------|----------|
| `Datasets/` | [`../Datasets/`](../Datasets/) |
| `Chatbot-Toxicity-Injection/data/` | [`../Chatbot-Toxicity-Injection/data/`](../Chatbot-Toxicity-Injection/) |
| `Chatbot-Toxicity-Injection/saves/` | [`../Chatbot-Toxicity-Injection/saves/`](../Chatbot-Toxicity-Injection/) |
| `Chatbot-Toxicity-Injection/Benign_Dataset.csv` | [`../Chatbot-Toxicity-Injection/Benign_Dataset.csv`](../Chatbot-Toxicity-Injection/) |
| `GRADE/data/` | [`../GRADE/data/`](../GRADE/) |
| `GRADE/tools/` | [`../GRADE/tools/`](../GRADE/) |
| `GRADE/output/` | [`../GRADE/output/`](../GRADE/) |
| `Evaluation/RTR_Evaluation_Classifier/` | [`../Evaluation/RTR_Evaluation_Classifier/`](../Evaluation/) |
| `Evaluation/RTR_Evaluation_Classifier_Heal/` | [`../Evaluation/RTR_Evaluation_Classifier_Heal/`](../Evaluation/) |
| `Universal-Prompt-Injection/data/` | [`../Universal-Prompt-Injection/data/`](../Universal-Prompt-Injection/) |
| `Models/` (if present) | [`../Models/`](../Models/) |

Helper scripts under `Datasets/` use repo-root [`optimus_paths.py`](../optimus_paths.py) via `bootstrap_paths.py` (included in the bundle).

## Contents (summary)

| Component | Approx. size | Notes |
|-----------|--------------|-------|
| `Datasets/` | ~12 GB | Paper datasets, scores, heals, eval CSVs, DBL artifacts |
| `Chatbot-Toxicity-Injection/` | ~6 GB | DBL `data/`, `saves/`, benign CSV |
| `GRADE/` | ~3 GB | DailyDialog data, ConceptNet/numberbatch tools, checkpoint |
| `Evaluation/` | ~2 GB | RTR BERT checkpoints and logs |
| `Universal-Prompt-Injection/data/` | ~34 MB | Preprocessed GCG attack CSVs |

Run `python ../GRADE/verify_paths.py`, `python ../Evaluation/verify_paths.py`, and `python ../Chatbot-Toxicity-Injection/verify_paths.py` after setup.
