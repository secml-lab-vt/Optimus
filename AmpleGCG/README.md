# AmpleGCG Attack Implementation

## Overview

Adversarial attack against safety-aligned LLMs using AmpleGCG-Plus to generate toxic responses through jailbreak attacks.

## Two-Step Process

### Step 1: Adversarial Response Generation (`AmpleGCG.py`)
- **Purpose**: Generate adversarial responses with toxicity evaluation
- **Models**: AmpleGCG-Plus (suffix generator) + Llama-2-7b-chat (response generator & toxicity evaluator)
- **Input**: Context_Heal dataset
- **Output**: `Datasets/generated_and_selected_by_highest_prob_responses_with_toxicity_prob.csv`

### Step 2: Dataset Processing (`file_process.py`)
- **Purpose**: Merge AmpleGCG outputs with Toxic-Category2 dataset
- **Process**: Extracts adversarial instructions and merges with original toxic dataset
- **Output**: `Datasets/toxic_category2_with_prepended_instructions.csv`

## Usage

### Quick Start
```bash
./run_amplegcg.sh
```

### Step-by-Step
```bash
# Step 1: Generate adversarial responses
python AmpleGCG.py --input_dataset ../Datasets/Context_Heal/meta-llama_Llama-2-13b-chat-hf_20240209-044652_category_2_type_1_Toxic-Category2_heal_dataset.json --output_dataset Datasets/generated_and_selected_by_highest_prob_responses_with_toxicity_prob.csv

# Step 2: Process and merge datasets
python file_process.py --ample_dataset Datasets/generated_and_selected_by_highest_prob_responses_with_toxicity_prob.csv --toxic_dataset ../Datasets/LM_Detect/Toxic-Category2_dataset.csv --output_dataset Datasets/toxic_category2_with_prepended_instructions.csv
```

## Technical Details

### Model Configuration
- **AmpleGCG-Plus**: `osunlp/AmpleGCG-plus-llama2-sourced-llama2-7b-chat`
- **Llama-2**: `meta-llama/Llama-2-7b-chat-hf`
- **Generation**: Beam search with 10 beams, 20-50 tokens, diversity penalty 1.0

### Toxicity Evaluation
- Probability-based detection method
- 'yes'/'no' toxicity classification
- Threshold: >0.5 probability for toxic classification

## Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch==2.1.0`
- `transformers==4.36.0`
- `datasets>=2.14.0`
- `pandas>=1.5.0`
- `tqdm>=4.64.0` 