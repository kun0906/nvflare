import argparse
import shutil
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import pandas as pd

import torch
import numpy as np
import os

from ragg.base import print_data
from ragg.utils import dirichlet_split
import collections

# Check if GPU is available and use it
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SEED = 42


# @timer
def gen_client_data(data_dir='data/Sentiment140', out_dir='.', CFG=None, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(random_state)
    # Total rows in the dataset
    total_rows = 1600000  # Replace with the actual number of rows in your CSV file
    # Number of rows to sample
    sample_size = int(0.1*total_rows)
    # Randomly choose rows to read
    skip = np.random.choice(total_rows, total_rows - sample_size, replace=False)
    # we only use 10% of data for the following experiment to save time.
    df = pd.read_csv('data/Sentiment140/training.1600000.processed.noemoticon.csv_bert.csv_pca_100.csv',
                     dtype=float, header=None, skiprows=skip.tolist())

    X, y = torch.tensor(df.iloc[:, 0:-1].values), torch.tensor(df.iloc[:, -1].values, dtype=int)
    print(collections.Counter(list(y.numpy())))
    # global shared data. each client has its own train set, val set, and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500,
                                                        shuffle=True, random_state=random_state)

    # Initialize scaler and fit ONLY on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mask = np.full(len(y_test), False)
    for l in CFG.LABELS:
        mask_ = y_test == l
        mask[mask_] = True
    X_test, y_test = X_test[mask], y_test[mask]
    # preprocessing X_test
    # X_test = normalize(X_test.numpy())
    y_test = y_test.numpy()
    # global shared dataset
    shared_data = {"X": torch.tensor(X_test).float().to(DEVICE), 'y': torch.tensor(y_test).to(DEVICE)}

    X, y = X_train, y_train
    mask = np.full(len(y), False)
    for l in CFG.LABELS:
        mask_ = y == l
        mask[mask_] = True
    X, y = X[mask], y[mask]
    # X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()
    num_samples = len(y)
    dim = X.shape[1]

    # random_state = 42
    torch.manual_seed(random_state)

    # Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.BIG_NUMBER, random_state=SEED)
    # Prepare client datasets
    Xs, Ys = [], []
    # Shuffle full dataset (using list of indices)
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # Split data into subsets for each client
    client_size = 1000 # each client has 1000 samples
    for _ in range(CFG.NUM_CLIENTS):
        X, X_client, y, y_client = train_test_split(X, y, test_size=client_size, shuffle=True, 
                                                    random_state=random_state)      # X, y will be reduced. 
        Xs.append(X_client)
        Ys.append(y_client)

    # Shuffle the lists X and y in the same order
    combined = list(zip(Xs, Ys))  # Combine the lists into pairs of (X[i], y[i])
    random.shuffle(combined)  # Shuffle the combined list
    # Unzip the shuffled list back into X and y
    Xs, Ys = zip(*combined)
    # Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.BIG_NUMBER, random_state=SEED)
    # Xs, Ys = [X[:]] * NUM_CLIENTS, [y[:]]*NUM_CLIENTS   # if each client has all the data
    total_size = 0
    for j, y_ in enumerate(Ys):
        vs = collections.Counter(y_.tolist())
        vs = dict(sorted(vs.items(), key=lambda x: x[0], reverse=False))
        print(f"client {j}'s data size: {len(y_)}, total classes: {len(vs)}, in which {vs}")
        total_size += len(y_)
    print(f"total size: {total_size}")
    ########################################### Benign Clients #############################################
    for c in range(CFG.NUM_HONEST_CLIENTS):
        client_type = 'Honest'
        print(f"\n*** client_{c}: {client_type}...")

        X_c, y_c = Xs[c], Ys[c]

        # might be used in server
        # train_info = {"client_type": client_type, "FNN": {}, 'client_id': c}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=0.2,
                                                       shuffle=True, random_state=random_state)
        # only two samples for val set
        train_indices, val_indices = train_test_split(train_indices, test_size=2, shuffle=True,
                                                      random_state=random_state)
        train_mask = np.full(num_samples_client, False)
        val_mask = np.full(num_samples_client, False)
        test_mask = np.full(num_samples_client, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).float().to(DEVICE), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        print(f'client_{c} data ({len(label_cnts)}): ', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)

    ########################################### Byzantine Clients #############################################
    # indices = torch.randperm(num_samples)  # Randomly shuffle
    for c in range(CFG.NUM_HONEST_CLIENTS, CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'Byzantine'
        print(f"\n*** client_{c}: {client_type}...")
        # X_c = X[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]
        # y_c = y[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]

        X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution

        # mask_c = np.full(len(y_c), False)
        # for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=CFG.BIG_NUMBER, replace=False):
        #     mask_ = y_c == l
        #     mask_c[mask_] = True
        # X_c = X_c[mask_c]
        # y_c = y_c[mask_c]

        # might be used in server
        # train_info = {"client_type": client_type, "FNN": {}, 'client_id': c}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=0.2,
                                                       shuffle=True, random_state=random_state)  # test set unchanged
        train_indices, val_indices = train_test_split(train_indices, test_size=2, shuffle=True,
                                                      random_state=random_state)
        train_mask = np.full(num_samples_client, False)
        val_mask = np.full(num_samples_client, False)
        test_mask = np.full(num_samples_client, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        # y_c[train_mask] = (NUM_CLASSES - 1) - y_c[train_mask]  # flip label
        # y_c[val_mask] = (NUM_CLASSES - 1) - y_c[val_mask]  # flip label

        # train_info['NUM_BYZANTINE_CLIENTS'] = NUM_BYZANTINE_CLIENTS
        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).to(DEVICE).float(), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        print(f'client_{c} data ({len(label_cnts)}): ', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)
#

# @timer
def gen_client_data_dirichlet(data_dir='data/Sentiment140', out_dir='.', CFG=None):
    os.makedirs(out_dir, exist_ok=True)

    # Total rows in the dataset
    total_rows = 1600000  # Replace with the actual number of rows in your CSV file
    # Number of rows to sample
    sample_size = int(0.1*total_rows)
    # Randomly choose rows to read
    skip = np.random.choice(total_rows, total_rows - sample_size, replace=False)
    # we only use 10% of data for the following experiment to save time.
    df = pd.read_csv('data/Sentiment140/training.1600000.processed.noemoticon.csv_bert.csv_pca_100.csv',
                     dtype=float, header=None, skiprows=skip.tolist())
    X, y = torch.tensor(df.iloc[:, 0:-1].values), torch.tensor(df.iloc[:, -1].values, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        shuffle=True, random_state=42)
    # Initialize scaler and fit ONLY on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mask = np.full(len(y_test), False)
    for l in CFG.LABELS:
        mask_ = y_test == l
        mask[mask_] = True
    X_test, y_test = X_test[mask], y_test[mask]
    # preprocessing X_test
    # X_test = normalize(X_test.numpy())
    y_test = y_test.numpy()
    shared_data = {"X": torch.tensor(X_test).float().to(DEVICE), 'y': torch.tensor(y_test).to(DEVICE)}

    X, y = X_train, y_train
    mask = np.full(len(y), False)
    for l in CFG.LABELS:
        mask_ = y == l
        mask[mask_] = True
    X, y = X[mask], y[mask]
    # X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()
    num_samples = len(y)
    dim = X.shape[1]

    random_state = 42
    torch.manual_seed(random_state)
    # indices = torch.randperm(num_samples)  # Randomly shuffle
    # step = int(num_samples / NUM_HONEST_CLIENTS)
    # step = 50  # for debugging
    # non_iid_cnt0 = 0  # # make sure that non_iid_cnt is always less than iid_cnt
    # non_iid_cnt1 = 0
    # m = len(y)//2

    # Xs1, Ys1 = dirichlet_split(X[:m, :], y[:m], num_clients=CFG.NUM_CLIENTS // 2, alpha=0.5, random_state=SEED)
    # Xs2, Ys2 = dirichlet_split(X[m:], y[m:], num_clients=CFG.NUM_CLIENTS - CFG.NUM_CLIENTS // 2, alpha=100,
    #                            random_state=SEED)
    # Xs, Ys = Xs1 + Xs2, Ys1 + Ys2

    Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.BIG_NUMBER, random_state=SEED)
    # Shuffle the lists X and y in the same order
    combined = list(zip(Xs, Ys))  # Combine the lists into pairs of (X[i], y[i])
    random.shuffle(combined)  # Shuffle the combined list
    # Unzip the shuffled list back into X and y
    Xs, Ys = zip(*combined)
    # Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.BIG_NUMBER, random_state=SEED)
    # Xs, Ys = [X[:]] * NUM_CLIENTS, [y[:]]*NUM_CLIENTS   # if each client has all the data
    total_size = 0
    for j, y_ in enumerate(Ys):
        vs = collections.Counter(y_.tolist())
        vs = dict(sorted(vs.items(), key=lambda x: x[0], reverse=False))
        print(f"client {j}'s data size: {len(y_)}, total classes: {len(vs)}, in which {vs}")
        total_size += len(y_)
    print(f"total size: {total_size}")
    ########################################### Benign Clients #############################################
    for c in range(CFG.NUM_HONEST_CLIENTS):
        client_type = 'Honest'
        print(f"\n*** client_{c}: {client_type}...")
        # X_c = X[indices[c * step:(c + 1) * step]]
        # y_c = y[indices[c * step:(c + 1) * step]]
        # np.random.seed(c)  # change seed
        # # if c % 4 == 0 and non_iid_cnt0 < NUM_HONEST_CLIENTS // 4:  # 1/4 of honest clients has part of classes
        # # if c == 0:
        # #     non_iid_cnt0 += 1  # make sure that non_iid_cnt is always less than iid_cnt
        # #     mask_c = np.full(len(y_c), False)
        # #     # for l in [0, 1, 2, 3, 4]:
        # #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT, replace=False):
        # #         mask_ = y_c == l
        # #         mask_c[mask_] = True
        # #     # mask_c = (y_c != (c%10))  # excluding one class for each client
        # # elif c == 1:
        # #     # elif c % 4 == 1 and non_iid_cnt1 < NUM_HONEST_CLIENTS // 4:  # 1/4 of honest clients has part of classes
        # #     non_iid_cnt1 += 1
        # #     mask_c = np.full(len(y_c), False)
        # #     # for l in [5, 6, 7, 8, 9]:
        # #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT*2, replace=False):
        # #         mask_ = y_c == l
        # #         mask_c[mask_] = True
        # if c < NUM_HONEST_CLIENTS // 3:
        #     # elif c % 4 == 1 and non_iid_cnt1 < NUM_HONEST_CLIENTS // 4:  # 1/4 of honest clients has part of classes
        #     non_iid_cnt1 += 1
        #     mask_c = np.full(len(y_c), False)
        #     # for l in [5, 6, 7, 8, 9]:
        #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT, replace=False):
        #         mask_ = y_c == l
        #         mask_c[mask_] = True
        # if c <= NUM_HONEST_CLIENTS//BIG_NUMBER:
        #     mask_c = np.full(len(y_c), False)
        #     # for l in [5, 6, 7, 8, 9]:
        #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT, replace=False):
        #         mask_ = y_c == l
        #         mask_c[mask_] = True
        # else:  # 2/4 of honest clients has IID distributions
        #     mask_c = np.full(len(y_c), True)
        # X_c = X_c[mask_c]
        # y_c = y_c[mask_c]

        X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution

        # might be used in server
        # train_info = {"client_type": client_type, "FNN": {}, 'client_id': c}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=1 - CFG.LABELING_RATE,
                                                       shuffle=True, random_state=SEED)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                      random_state=CFG.TRAIN_VAL_SEED)
        train_mask = np.full(num_samples_client, False)
        val_mask = np.full(num_samples_client, False)
        test_mask = np.full(num_samples_client, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).float().to(DEVICE), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        print(f'client_{c} data ({len(label_cnts)}): ', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)

    ########################################### Byzantine Clients #############################################
    indices = torch.randperm(num_samples)  # Randomly shuffle
    for c in range(CFG.NUM_HONEST_CLIENTS, CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'Byzantine'
        print(f"\n*** client_{c}: {client_type}...")
        # X_c = X[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]
        # y_c = y[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]

        X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution

        # mask_c = np.full(len(y_c), False)
        # for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=CFG.BIG_NUMBER, replace=False):
        #     mask_ = y_c == l
        #     mask_c[mask_] = True
        # X_c = X_c[mask_c]
        # y_c = y_c[mask_c]

        # might be used in server
        # train_info = {"client_type": client_type, "FNN": {}, 'client_id': c}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=1 - CFG.LABELING_RATE,
                                                       shuffle=True, random_state=SEED)  # test set unchanged
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                      random_state=CFG.TRAIN_VAL_SEED)
        train_mask = np.full(num_samples_client, False)
        val_mask = np.full(num_samples_client, False)
        test_mask = np.full(num_samples_client, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        # y_c[train_mask] = (NUM_CLASSES - 1) - y_c[train_mask]  # flip label
        # y_c[val_mask] = (NUM_CLASSES - 1) - y_c[val_mask]  # flip label

        # train_info['NUM_BYZANTINE_CLIENTS'] = NUM_BYZANTINE_CLIENTS
        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).to(DEVICE).float(), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        print(f'client_{c} data ({len(label_cnts)}): ', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)
#
#
# if __name__ == '__main__':
#     CFG = CONFIG()
#     CFG.LABELS = {0, 1}
#
#     CFG.NUM_CLIENTS = 2
#     CFG.BIG_NUMBER = 0.5      # alpha for controlling non-IID level
#     CFG.NUM_HONEST_CLIENTS = 1
#     CFG.NUM_BYZANTINE_CLIENTS = 1
#     CFG.LABELING_RATE = 0.1     # how much data for testing
#     gen_client_data(data_dir='data/Sentiment140', out_dir=f'./data/{CFG.NUM_CLIENTS}_split', CFG=CFG)
