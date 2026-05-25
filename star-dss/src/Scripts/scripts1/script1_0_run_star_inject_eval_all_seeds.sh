#!/usr/bin/env bash
# Step 3: inject_eval only (train_star.py), seeds 7/13/888.
# Override: USE_MODEL_UUID=<uuid> bash script1_0_run_star_inject_eval_all_seeds.sh

source "$(dirname "$0")/../common.sh"

use_model="${USE_MODEL_UUID:-4d892f7e-2d15-4488-9731-3f5712ad15bb}"

chatbot="LLAMA2-LORA"
category="2"
cr_num=22000
benign_dataset="Benign-PersonaChat"
toxic_dataset="Category2"
healing_dataset="Prosocial"
injection="True"
percentage=0.1
heal="False"
heal_percentage=0
filter1="False"
filter2="False"
model_vers="None"
threshold="0.5"
adversarial="False"
use_eval_dataset="Category2_evaluate"
hydra_config="sft_openrlhf"
seeds=(7 13 888)

echo "INJECT EVAL — model UUID: ${use_model}"

for seed in "${seeds[@]}"; do
  echo "Inject eval seed=${seed}"

  python src/train_star.py cuda \
    --chatbot "$chatbot" --mode inject_eval \
    --cr_num "$cr_num" \
    --injection "$injection" --percentage "$percentage" \
    --heal "$heal" --heal_percentage "$heal_percentage" \
    --healing_dataset "$healing_dataset" \
    --benign_dataset "$benign_dataset" --toxic_dataset "$toxic_dataset" \
    --filter1 "$filter1" --filter2 "$filter2" \
    --uuid "$use_model" --seed "$seed" \
    --category "$category" --model_vers "$model_vers" \
    --threshold "$threshold" --adversarial "$adversarial" \
    --use_model "$use_model" \
    --use_eval_dataset "$use_eval_dataset" \
    --hydra_config "$hydra_config"
done

echo "DONE."
