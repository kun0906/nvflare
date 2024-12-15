"""
    # Sent140 dataset (from LEAF dataset) preprocessing

    https://www.kaggle.com/datasets/kazanova/sentiment140

    target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
"""
import collections

import numpy as np
import json

from language_utils import bag_of_words, get_word_emb_arr, val_to_vec

num_classes = 3
VOCAB_DIR = 'embs.json'
_, _, vocab = get_word_emb_arr(VOCAB_DIR)


def process_x(raw_x_batch):
    """
    Return:
        len(vocab) by len(raw_x_batch) np array
    """
    x_batch = [e[4] for e in raw_x_batch]  # list of lines/phrases
    bags = [bag_of_words(line, vocab) for line in x_batch]
    bags = np.array(bags)
    return bags


def process_y(raw_y_batch):
    y_batch = [int(e) for e in raw_y_batch]
    y_batch = [val_to_vec(num_classes, e) for e in y_batch]
    y_batch = np.array(y_batch)
    return y_batch

def main():
    json_file = 'sent140/data/train/all_data_niid_0_keep_0_train_9.json'

    # Read and parse the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    users, num_samples, user_data = data['users'], data['num_samples'], data['user_data']
    print('len(users)', len(users))
    y_labels = []
    for user in users:
        raw_x, raw_y = user_data[user]['x'], user_data[user]['y']
        y_labels.extend([int(e) for e in raw_y])
    print(collections.Counter(y_labels))

    for user in users:
        raw_x, raw_y = user_data[user]['x'], user_data[user]['y']
        x = process_x(raw_x)
        y = process_y(raw_y)
        # y_labels.extend([int(e) for e in raw_y])
        print(user, len(set(y_labels)),  collections.Counter(y_labels))

if __name__ == '__main__':
    main()
