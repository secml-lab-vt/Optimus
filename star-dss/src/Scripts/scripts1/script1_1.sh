#!/usr/bin/env bash
# Step 1: value labels for Category 1 (30% injection, 40k CRs), seed 7.

source "$(dirname "$0")/../common.sh"

echo "TRAIN EVAL — Category 1 value dataset"

round_uuid1=$(uuidgen)

chatbot="LLAMA2-LORA"
cr_num=40000
injection="True"
percentage=0.3
heal="False"
heal_percentage=0
healing_dataset="Prosocial"
benign_dataset="Benign-PersonaChat"
toxic_dataset="Category1"
model_vers="None"
category="1"
filter_bool="False"
threshold="0.5"
script_name="script1_1.sh"

python src/train_bot_star.py cuda \
  --chatbot "$chatbot" --mode train_eval \
  --cr_num "$cr_num" --injection "$injection" --percentage "$percentage" \
  --heal "$heal" --heal_percentage "$heal_percentage" \
  --healing_dataset "$healing_dataset" \
  --benign_dataset "$benign_dataset" --toxic_dataset "$toxic_dataset" \
  --use_model N --filter1 "$filter_bool" \
  --uuid "$round_uuid1" --seed 7 \
  --category "$category" --model_vers "$model_vers" \
  --threshold "$threshold" --script_name "$script_name"

echo "DONE. UUID: $round_uuid1"
