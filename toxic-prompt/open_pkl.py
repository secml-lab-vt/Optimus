#open pkl file.

import pickle
import json

# Path to your .pkl file
pkl_file_path = "sfs_out/task1/measuring-hate-speech_t5-base_144_2000.pkl"

# Load the .pkl file
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)
    # data = json.load()

# Print the contents
print(type(data))
print(data['statistics'])
# print(data.values())
