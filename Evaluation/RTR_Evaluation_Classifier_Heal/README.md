# RTR Evaluation Classifier (Heal)

BERT-based binary classifier training and **Runtime Toxicity Rate (RTR)** evaluation for **heal-context** experiments.

Same pipeline as [RTR_Evaluation_Classifier](../RTR_Evaluation_Classifier/), but training data uses the healed focal-full splits (`Heal_Full_Focal_Classifier_*`).

## Layout

```
RTR_Evaluation_Classifier_Heal/
├── codes/                  # Python scripts (run from here)
├── models_binary_category1/
├── models_binary_category2/
├── logs_binary_category1/
├── logs_binary_category2/
└── consolidated_results/
```

See **[codes/README.md](codes/README.md)** for script reference, CLI examples, and data paths.

## Categories

| CLI flag | Model / log dir | Classifier training data |
|---|---|---|
| `category1` | `binary_category1` | `Heal_Full_Focal_Classifier_dataset_Category1_{train,val,test}.json` |
| `category2` | `binary_category2` | `Heal_Full_Focal_Classifier_dataset_Category2_{train,val,test}.json` |

All paths are relative to `codes/` and live under `Optimus/Datasets/Processed_datasets/Classifier/`.

## Quick start

```bash
cd codes

# Environment (optional)
bash setup_env.sh

# 1. Train classifier (once per category)
python BERT_Classifier.py --category category1 --classifier_type combined --mode train
python BERT_Classifier.py --category category1 --classifier_type combined --mode precision_tuning
python BERT_Classifier.py --category category1 --classifier_type combined --mode val-precision_tuned

# 2. RTR evaluation
python BERT_Classifier_RTR.py \
    --category category1 \
    --classifier_type combined \
    --mode test-precision_tuned \
    --chatbot DD-BART
```

Repeat with `--category category2` and other chatbots.

## Related modules

- **Non-heal variant:** [../RTR_Evaluation_Classifier](../RTR_Evaluation_Classifier/)
- **Dataset generation:** `Optimus/Data_processor/classifier/generate_classifier_focal_full_heal.py`
- **Heal CSV sources:** `Optimus/Datasets/Classifier/Context_Heal/` (generated via `Healing_creation/`)
