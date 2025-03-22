"""
    Renames all files' names

"""

import os

# Directory containing your files
dir_path = '.'

# Recursively list all files
all_files = []
for root, dirs, files in os.walk(dir_path):
    for file in files:
        all_files.append(os.path.join(root, file))

# Rename files with a pattern
for file_path in all_files:
    string = 'ragg_random_noise_data.py'
    new_string = 'ragg_data_random_noise.py'
    if string in file_path:
        new_file_path = file_path.replace(string, new_string)
        # be cautious when using it.
        os.rename(file_path, new_file_path)
        print(f"{file_path}->{new_file_path}", flush=True)

