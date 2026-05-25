
import os
from datasets import load_dataset,concatenate_datasets, Dataset




folder_path = '../../Models/Custom/model_runs/BB400M'
datasetpath = './runs.csv'

dataset = load_dataset("csv", data_files=datasetpath,split='train')

listing = dataset['ids']


# Loop through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if not os.path.isfile(file_path):
        # print(f'{file_path}'
        for filename1 in os.listdir(file_path):
            file_path1 = os.path.join(file_path, filename1)
            if not os.path.isfile(file_path1):
               if filename1 in listing:
                    print(filename1, "yes")
                    # delete the whole folder
                    print(file_path1)
                    os.system(f'rm -r {file_path1}')

