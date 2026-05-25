# RTR Evaluation Classifier

BERT-based binary classifier training and **Runtime Toxicity Rate (RTR)** evaluation for **non-heal** (standard context) experiments.

Trains on focal-loss BERT checkpoints per category, then scores injected chatbot runs under `Models/Custom/model_runs/`.

## Layout

```
RTR_Evaluation_Classifier/
├── codes/                  # Python scripts (run from here)
├── models_binary_category1/  # Saved checkpoints (created on train)
├── models_binary_category2/
├── logs_binary_category1/    # Training / eval logs
├── logs_binary_category2/
└── consolidated_results/     # Aggregated RTR outputs (created on RTR runs)
```

See **[codes/README.md](codes/README.md)** for script reference, CLI examples, and data paths.

## Categories

| CLI flag | Model / log dir | Classifier training data |
|---|---|---|
| `category1` | `binary_category1` | `Full_Focal_Classifier_dataset_Category1_{train,val,test}.json` |
| `category2` | `binary_category2` | `Full_Focal_Classifier_dataset_Category2_{train,val,test}.json` |

All paths are relative to `codes/` and live under `Optimus/Datasets/Processed_datasets/Classifier/`.

## Quick start

```bash
cd codes

# 1. Train classifier (once per category)
python BERT_Classifier.py --category category1 --classifier_type combined --mode train
python BERT_Classifier.py --category category1 --classifier_type combined --mode precision_tuning
python BERT_Classifier.py --category category1 --classifier_type combined --mode val-precision_tuned

# 2. RTR evaluation over injection runs
python BERT_Classifier_RTR.py \
    --category category1 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART
```

Repeat with `--category category2` and other `--chatbot` values (`BB400M`, `DD-BART`, `DialoGPT`, `MISTRAL`, `LLAMA2-LORA`).

## Related modules

- **Heal variant:** [../RTR_Evaluation_Classifier_Heal](../RTR_Evaluation_Classifier_Heal/) — same workflow on `Heal_Full_Focal_Classifier_*` data
- **Dataset generation:** `Optimus/Data_processor/classifier/generate_classifier_focal_full.py`
