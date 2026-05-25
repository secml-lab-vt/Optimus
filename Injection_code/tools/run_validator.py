import os

def rename_folders(directory_path):
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path):
            new_name = folder_name + "_0.5"
            new_path = os.path.join(directory_path, new_name)
            os.rename(folder_path, new_path)
            print(f'Renamed: {folder_name} to {new_name}')

def check_multiple(directory_path):
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path):
            # check if this directory has more than 1 folders
            if len(os.listdir(folder_path)) > 1:
                print(f'{folder_name}')

def folder_loop(directory_path):
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        # loop through the subfolders
        for subfolder_name in os.listdir(folder_path):
            if "Context_Heal" in folder_path or "None" in folder_path:
                continue
            subfolder_path = os.path.join(folder_path, subfolder_name)
            # check if this is a directory
            if os.path.isdir(subfolder_path):
                print(f'{subfolder_path}')

# Replace 'your_directory_path' with the actual path of the directory you want to process
directory_path = '../../Models/Custom/model_runs/BB400M/'

# rename_folders(directory_path)
# check_multiple(directory_path)
folder_loop(directory_path)