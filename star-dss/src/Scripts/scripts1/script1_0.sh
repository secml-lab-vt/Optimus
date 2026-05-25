#!/usr/bin/env bash
# Step 1: create STAR value labels (train_bot_star.py), Category 2, seeds 7/13/888.

source "$(dirname "$0")/../common.sh"

echo "TRAIN EVAL — value dataset creation"

round_uuid1=$(uuidgen)

chatbot="LLAMA2-LORA"
cr_num=22000
injection="True"
percentage=0.1
heal="False"
heal_percentage=0
healing_dataset="Prosocial"
benign_dataset="Benign-PersonaChat"
toxic_dataset="Category2"
model_vers="None"
category="2"
filter_bool="False"
threshold="0.5"
script_name="script1_0.sh"

for seed in 7 13 888; do
  python src/train_bot_star.py cuda \
    --chatbot "$chatbot" --mode train_eval \
    --cr_num "$cr_num" --injection "$injection" --percentage "$percentage" \
    --heal "$heal" --heal_percentage "$heal_percentage" \
    --healing_dataset "$healing_dataset" \
    --benign_dataset "$benign_dataset" --toxic_dataset "$toxic_dataset" \
    --use_model N --filter1 "$filter_bool" \
    --uuid "$round_uuid1" --seed "$seed" \
    --category "$category" --model_vers "$model_vers" \
    --threshold "$threshold" --script_name "$script_name"
done

echo "DONE. UUID: $round_uuid1"
