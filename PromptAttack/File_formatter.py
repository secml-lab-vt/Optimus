
from datasets import load_dataset


#read a huggingface csv dataset
# path = "../Datasets/Adversarial_attack/meta-llama_Llama-2-13b-chat-hf_20240215-174359_category_2_type_1_Toxic-Category2_adversarial1.csv"
path = "../Datasets/Adversarial_attack/meta-llama_Llama-2-13b-chat-hf_20240215-174554_category_1_type_1_Toxic-Category1_adversarial1.csv"
dataset = load_dataset("csv", data_files=path,split='train')

# Rename a hugginface column
dataset = dataset.rename_column("response","original_response")
dataset = dataset.rename_column("adversarial_response","response")



# Save the dataset
dataset.to_csv("../Datasets/LM_Detect/Adv_Toxic-Category1_dataset.csv")

