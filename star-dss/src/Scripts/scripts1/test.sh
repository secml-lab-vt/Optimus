#!/usr/bin/env bash
# Single-seed STAR-DSS training smoke test (train_star.py).

source "$(dirname "$0")/../common.sh"

round_uuid="${ROUND_UUID:-0f83d0a5-5b12-4895-84cd-b4361d9ab073}"
seed="${SEED:-7}"

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
filter_bool="False"
model_vers="None"
threshold="0.5"
adversarial="False"
use_value=true
use_kl=true
hydra_config="sft_openrlhf"
use_eval_dataset="Category2_evaluate"

run_config="None-DSS_Benign-PersonaChat_Category2_Prosocial_True_0.1_False_0_False_False_idea0_0.5_adv_False"
dataset_path="${OPTIMUS_ROOT}/Models/Custom/model_runs/LLAMA2-LORA/${run_config}/${round_uuid}/seed_${seed}/train_data/train_dataset_with_values.jsonl"

echo "Star-DSS training smoke test"
echo "  UUID: ${round_uuid}"
echo "  seed: ${seed}"
echo "  dataset_path: ${dataset_path}"

extra_flags=""
[[ "$use_value" == true ]] && extra_flags+=" --use_value"
[[ "$use_kl" == true ]] && extra_flags+=" --use_kl"
[[ -n "$dataset_path" ]] && extra_flags+=" --dataset_path ${dataset_path}"
[[ -n "$use_eval_dataset" && "$use_eval_dataset" != "N" ]] && extra_flags+=" --use_eval_dataset ${use_eval_dataset}"

deepspeed --num_gpus=2 src/train_star.py cuda \
  --chatbot "$chatbot" --mode train_eval \
  --cr_num "$cr_num" \
  --injection "$injection" --percentage "$percentage" \
  --heal "$heal" --heal_percentage "$heal_percentage" \
  --healing_dataset "$healing_dataset" \
  --benign_dataset "$benign_dataset" --toxic_dataset "$toxic_dataset" \
  --filter1 "$filter_bool" --filter2 "False" \
  --uuid "$round_uuid" --seed "$seed" \
  --category "$category" --model_vers "$model_vers" \
  --threshold "$threshold" --adversarial "$adversarial" \
  --hydra_config "$hydra_config" \
  $extra_flags

echo "DONE."
