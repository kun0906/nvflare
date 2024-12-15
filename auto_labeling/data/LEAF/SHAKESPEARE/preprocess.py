"""
# Shakespeare dataset (from LEAF dataset) preprocessing

1. The url for shakespeare dataset changed
    1) from http to https
    2) from old to old/old
#wget http://www.gutenberg.org/files/100/old/1994-01-100.zip

curl -v -L -O https://www.gutenberg.org/files/100/old/old/1994-01-100.zip

2. Copy data/utils and data/shakespeare from leaf repo: https://github.com/TalwalkarLab/leaf/

3. Generate data
    #cd shakespeare
    #./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8
    #./stats.sh

4. Preprocess data
   %copy 'language_utils.py' from leaf: models/utils/language_utils.py
   python preprocess.py


"""
import collections

import numpy as np
import json
from language_utils import letter_to_vec, word_to_indices


def process_x(raw_x_batch):
    # returns a list of character indices
    # ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
    # lens = set([len(word) for word in raw_x_batch])
    # print(lens, flush=True)       # same length
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    # one-hot encoding
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch


def main():
    json_file = 'shakespeare/data/train/all_data_niid_0_keep_0_train_9.json'

    # Read and parse the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    users, num_samples, user_data = data['users'], data['num_samples'], data['user_data']
    print('len(users)', len(users))
    for user in users:
        raw_x, raw_y = user_data[user]['x'], user_data[user]['y']
        x = process_x(raw_x)
        y = process_y(raw_y)

        y_labels = [np.argmax(np.array(one_hot)) for one_hot in y]
        print(user, len(set(y_labels)),  collections.Counter(y_labels))


if __name__ == '__main__':
    main()
