import collections
import gzip
import os
from random import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ragg.utils import timer, dirichlet_split
from ragg.base import print_data, normalize


@timer
def gen_client_data(CFG=None):
    SEED = CFG.SEED
    DEVICE = CFG.DEVICE
    data_name = CFG.data_name
    in_data_dir = CFG.in_data_dir
    out_data_dir = CFG.out_data_dir
    os.makedirs(out_data_dir, exist_ok=True)

    if data_name == 'Sentiment140':
        # Total rows in the dataset
        total_rows = 1600000  # Replace with the actual number of rows in your CSV file
        # Number of rows to sample
        sample_size = int(0.1 * total_rows)
        # Randomly choose rows to read
        skip = np.random.choice(total_rows, total_rows - sample_size, replace=False)
        df = pd.read_csv(f'{in_data_dir}/training.1600000.processed.noemoticon.csv_bert.csv',
                         dtype=float, header=None, skiprows=skip.tolist())
        X, y = np.asarray(df.iloc[:, 0:-1].values), np.asarray(df.iloc[:, -1].values, dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                            shuffle=True, random_state=42)

        # Initialize scaler and fit ONLY on training data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    elif data_name == 'MNIST':
        from torchvision import datasets
        train_dataset = datasets.MNIST(root="./data", train=True, transform=None, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)
        X_test = test_dataset.data.numpy()
        y_test = test_dataset.targets.numpy()

        X_train = train_dataset.data.numpy()  # Tensor of shape (60000, 28, 28)
        y_train = train_dataset.targets.numpy()  # Tensor of shape (60000,)

        # Normalize data
        X_train = normalize(X_train)
        X_test = normalize(X_test)

    elif data_name == 'spambase':
        df = pd.read_csv(os.path.join(in_data_dir, 'spambase.data'), dtype=float, header=None)
        X, y = df.iloc[:, 0:-1].values, np.asarray(df.iloc[:, -1].values, dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                            shuffle=True, random_state=42)

        # Initialize scaler and fit ONLY on training data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        raise NotImplementedError(data_name)

    shared_data = {"X": torch.tensor(X_test).float().to(DEVICE), 'y': torch.tensor(y_test).to(DEVICE)}

    X, y = X_train, y_train
    num_samples = len(y)
    # dim = X.shape[1]

    # random_state = 42
    # torch.manual_seed(random_state)
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

    Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.alpha, random_state=SEED)
    # # Shuffle the lists X and y in the same order
    # combined = list(zip(Xs, Ys))  # Combine the lists into pairs of (X[i], y[i])
    # random.shuffle(combined)  # Shuffle the combined list
    # # Unzip the shuffled list back into X and y
    # Xs, Ys = zip(*combined)
    # # Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.BIG_NUMBER, random_state=SEED)
    # # Xs, Ys = [X[:]] * NUM_CLIENTS, [y[:]]*NUM_CLIENTS   # if each client has all the data
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
        if CFG.VERBOSE >= 20:
            print_data(local_data)

        out_file = f'{out_data_dir}/{c}.pt.gz'
        # torch.save(local_data, out_file)
        with gzip.open(out_file, 'wb') as f:
            torch.save(local_data, f)

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
        if CFG.VERBOSE >= 20:
            print_data(local_data)

        out_file = f'{out_data_dir}/{c}.pt.gz'
        # torch.save(local_data, out_file)
        with gzip.open(out_file, 'wb') as f:
            torch.save(local_data, f)

    return


class CONFIG:
    def __init__(self, in_data_dir='', out_data_dir='', data_name='', SEED=0, TRAIN_VAL_SEED=42, DEVICE='GPU',
                 VERBOSE=1, BIG_NUMBER=1, alpha = 1.0,
                 NUM_CLIENTS=100, NUM_HONEST_CLIENTS=20, NUM_BYZANTINE_CLIENTS=10):
        self.in_data_dir = in_data_dir
        self.out_data_dir = out_data_dir
        self.data_name = data_name
        self.alpha = alpha  # for dirichlet dist

        self.SEED = SEED
        self.TRAIN_VAL_SEED = TRAIN_VAL_SEED
        self.DEVICE = DEVICE
        self.VERBOSE = VERBOSE
        self.LABELING_RATE = 0.8  # 80% data in local client has labels
        self.BIG_NUMBER = BIG_NUMBER
        self.SERVER_EPOCHS = None
        self.CLIENT_EPOCHS = 5
        self.BATCH_SIZE = 512  # -1
        self.IID_CLASSES_CNT = 5
        self.NUM_CLIENTS = NUM_CLIENTS
        self.NUM_BYZANTINE_CLIENTS = NUM_BYZANTINE_CLIENTS
        self.NUM_HONEST_CLIENTS = NUM_HONEST_CLIENTS
        assert self.NUM_CLIENTS == (self.NUM_HONEST_CLIENTS + self.NUM_BYZANTINE_CLIENTS)
        self.AGGREGATION_METHOD = None
        self.LABELS = set()
        self.NUM_CLASSES = None

    def __str__(self):
        return str(self.__dict__)  # Prints attributes as a dictionary

    def __repr__(self):
        return f"CONFIG({self.__dict__})"  # More detailed representation


if __name__ == '__main__':
    """
        pwd = ICONIP/
        python3 get_client_data.py
    """
    root_dir = '../data'
    for data_name in ['spambase', 'MNIST', 'Sentiment140']:
        NUM_CLIENTS = 100
        NUM_BYZANTINE_CLIENTS = 48  #
        NUM_HONEST_CLIENTS = NUM_CLIENTS - NUM_BYZANTINE_CLIENTS
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        alpha = 1.0
        CFG = CONFIG(in_data_dir=f'{root_dir}/{data_name}', out_data_dir='.', data_name=data_name,
                     NUM_CLIENTS=NUM_CLIENTS, NUM_HONEST_CLIENTS=NUM_HONEST_CLIENTS,
                     NUM_BYZANTINE_CLIENTS=NUM_BYZANTINE_CLIENTS, alpha=alpha,
                     SEED=0, TRAIN_VAL_SEED=42, DEVICE=DEVICE,
                     )
        out_data_dir = (f'data/{data_name}/h_{CFG.NUM_HONEST_CLIENTS}-b_{CFG.NUM_BYZANTINE_CLIENTS}'
                        f'-{CFG.IID_CLASSES_CNT}-{CFG.LABELING_RATE}-{CFG.alpha}'
                        f'/{CFG.TRAIN_VAL_SEED}')
        CFG.out_data_dir = out_data_dir
        # CFG.data_out_dir = f'/projects/kunyang/nvflare_py31012/nvflare/{data_out_dir}'
        print(f'\n************{CFG}************')
        gen_client_data(CFG)
