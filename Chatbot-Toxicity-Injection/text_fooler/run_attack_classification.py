import os

# for wordLSTM target
# command = 'python attack_classification.py --dataset_path data/yelp ' \
#           '--target_model wordLSTM --batch_size 128 ' \
#           '--target_model_path /scratch/jindi/adversary/BERT/results/yelp ' \
#           '--word_embeddings_path /data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.200d.txt ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache'

# for BERT target
command = 'python attack_classification.py --dataset_path data/yelp ' \
          '--target_model bert ' \
          '--target_model_path /scratch/jindi/adversary/BERT/results/yelp ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path /scratch/jindi/adversary/cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /scratch/jindi/tf_cache'

command = 'python attack_classification.py --dataset_path data/yelp ' \
          '--target_model bert ' \
          '--target_model_path ../saves/toxic_classifier/WTC_new_focal-2_1_best ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path ./COUNTER_FITTING_WORD_EMBEDDINGS/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
          '--USE_cache_path ./results/tf_cache ' \
          '--output_dir results/snli_infersent'

os.system(command)
