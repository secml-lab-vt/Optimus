from datasets import load_dataset

path = "generated_and_selected_by_highest_prob_responses_with_toxicity_prob.csv"

dataset = load_dataset('csv', data_files=path, split="train")

predictions = dataset['prediction']

# count frequency of each unique value in the predictions column
unique_values = {}
for value in predictions:
    if value in unique_values:
        unique_values[value] += 1
    else:
        unique_values[value] = 1

# print the unique values and their counts
print("Unique values and their counts:")
for value, count in unique_values.items():
    print(f"{value}: {count}")


print(dataset) 