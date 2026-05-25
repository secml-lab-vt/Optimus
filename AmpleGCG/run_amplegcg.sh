#!/bin/bash

echo "Running AmpleGCG.py..."
python AmpleGCG.py --input_dataset ../Datasets/Context_Heal/meta-llama_Llama-2-13b-chat-hf_20240209-044652_category_2_type_1_Toxic-Category2_heal_dataset.json --output_dataset Datasets/generated_and_selected_by_highest_prob_responses_with_toxicity_prob.csv

echo "Running file_process.py..."
python file_process.py --ample_dataset Datasets/generated_and_selected_by_highest_prob_responses_with_toxicity_prob.csv --toxic_dataset ../Datasets/LM_Detect/Toxic-Category2_dataset.csv --output_dataset Datasets/toxic_category2_with_prepended_instructions.csv 