#!/bin/bash

conda init bash
conda activate advText

mkdir ./text_fooler/results/$1

echo $2
echo $5
echo $6

python ./text_fooler/attack_classification.py --dataset_path ./text_fooler/data/$1 \
          --target_model $2 \
          --target_model_path 0 \
          --max_seq_length 256 --batch_size 32 \
          --counter_fitting_embeddings_path ./text_fooler/COUNTER_FITTING_WORD_EMBEDDINGS/counter-fitted-vectors.txt \
          --counter_fitting_cos_sim_path ./text_fooler/cos_sim_counter_fitting.npy \
          --USE_cache_path ./text_fooler/results/tf_cache \
          --output_dir ./text_fooler/results/$1 \
          --data_size 100000 \
          --sim_score_window $3 \
          --import_score_threshold $4 \
          --sim_score_threshold $5 \
          --synonym_num $6

conda activate chatbot
