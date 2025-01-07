""" Cora
    A citation network widely used in graph-based machine learning research

    Key Details
        Nodes: 2,708
            Each node represents a scientific paper.
        Edges: 5,429？ 10556
            Edges represent citation relationships between papers.
        Features: 1,433
            Each node has a sparse feature vector corresponding to the presence of specific words in the paper.
        Classes: 7
            Each node (paper) belongs to one of seven classes, which represent different research topics:
                Case-Based
                Genetic Algorithms
                Neural Networks
                Probabilistic Methods
                Reinforcement Learning
                Rule Learning
                Theory
                Dataset Properties
        The graph is undirected; while citations are naturally directed, the dataset uses undirected edges.
        Node features are binary attributes indicating the presence of specific keywords in the document.

    https://graphsandnetworks.com/the-cora-dataset/
    The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
    The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued
    word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary
    consists of 1433 unique words.

    cd data/Cora
    PYTHONPATH=. python3 preprocessing.py
"""
import collections
import os
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid


# Timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def extract_xy_edges(X, y, train_indices, edge_indices, partial_classes):
    # extract data for client i

    X_train = X[train_indices]
    y_train = y[train_indices]

    # get original indices
    data_ = []
    client_X = []
    client_y = []
    client_train_indices = []
    for i, y_ in enumerate(y_train):
        if y_ in set(partial_classes):
            data_.append((X_train[i], y_, train_indices[i]))  # train_indices[i] is the original index of all data
            client_X.append(X_train[i])
            client_y.append(y_)
            client_train_indices.append(train_indices[i])

    # get new indices on client data
    m = len(data_)
    client_edge_indices = []
    client_original_edges = []
    for i in range(m):
        x_, y_, e_i = data_[i]
        for j in range(i + 1, m):
            x2_, y2_, e_j = data_[j]
            if (e_i, e_j) in edge_indices:
                client_original_edges.append((e_i, e_j))  # original indices in y
                client_edge_indices.append((i, j))  # new indices in client data
            if (e_j, e_i) in edge_indices:
                client_original_edges.append((e_j, e_i))  # original indices in y
                client_edge_indices.append((j, i))

    unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in client_edge_indices])
    print(f'unique edges: {len(unqiue_edges)}, edge_indices/2: {len(client_edge_indices) / 2}')
    return (np.array(client_X), np.array(client_y), np.array(client_train_indices),
            np.array(client_edge_indices).T, np.array(client_original_edges).T)


def split_train_val_test(X, y, indices, edge_indices, original_edge_indices, test_size=0.2, val_size=0.05):
    num_samples = len(X)
    train_mask = np.full(num_samples, False)
    val_mask = np.full(num_samples, False)
    test_mask = np.full(num_samples, False)

    new_indices = np.arange(num_samples)
    # get train and test   # for GNN, only 10% of data are labeled + 90% of data are unlabeled
    train_indices, test_indices = train_test_split(new_indices, test_size=test_size, shuffle=True, random_state=42)
    test_mask[test_indices] = True

    # split train set again into train set and validation set
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, shuffle=True, random_state=42)
    train_mask[train_indices] = True
    val_mask[val_indices] = True

    # Note here, we only split X and y. For edges, we don't split them, as GNN will use all the edges during training
    # (semi-supervised learning, so we don't need to split edges)
    client_data = {'X': X, 'y': y, 'indices': indices, 'original_indices': indices,
                   'edge_indices': edge_indices, 'original_edge_indices': original_edge_indices,
                   'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}

    return client_data



def extract_edges(X, y, test_mask, edge_indices):
    X_test = X[test_mask]
    y_test = y[test_mask]

    indices = np.array(range(len(X)))
    test_indices = indices[test_mask]
    new_edge_indices = []
    new_original_edge_indices = []
    m = len(test_indices)
    for i in range(m):
        for j in range(i+1, m):
            a, b = test_indices[i], test_indices[j]
            if (a, b) in edge_indices:
                new_original_edge_indices.append([a, b])
                new_edge_indices.append([i, j])
            if (b, a) in edge_indices:
                new_original_edge_indices.append([b, a])
                new_edge_indices.append([j, i])

    unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in new_edge_indices])
    print(f'unique edges: {len(unqiue_edges)}, edge_indices/2: {len(new_edge_indices)/2}')
    return X_test, y_test, np.array(new_edge_indices), np.array(new_original_edge_indices)


@timer
def preprocessing():
    # Load the Cora dataset
    dataset = Planetoid(root='./data', name='Cora', split='full')

    # Access the first graph in the dataset
    data = dataset[0]

    # Dataset summary
    print(f"Dataset Summary:\n"
          f"- Number of Nodes: {data.num_nodes}\n"
          f"- Number of Edges: {data.num_edges}\n"
          f"- Node Feature Size: {data.x.shape[1]}\n"
          f"- Number of Classes: {dataset.num_classes}")

    # Extract data into NumPy arrays
    X = data.x.numpy()
    Y = data.y.numpy()
    # edge_indices = data.edge_index.numpy().T
    # edge_indices = set([(row[0], row[1]) for row in edge_indices])
    edge_indices = set(map(tuple, data.edge_index.numpy().T))
    unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in data.edge_index.numpy().T])
    print(f'unique edges: {len(unqiue_edges)} =? edge_indices/2: {len(edge_indices)/2}, '
          f'edges: {data.edge_index.shape}')

    # Initialize boolean masks for train, validation, and test sets
    num_samples = len(X)
    train_mask = np.full(num_samples, False)
    val_mask = np.full(num_samples, False)
    test_mask = np.full(num_samples, False)

    # Before we split the whole data into client, we first keep 10% of them as the shared the test data.
    # Set random seed for reproducibility
    random_state = 42
    # Create indices for train/test split
    indices = np.arange(num_samples)
    train_indices, test_indices = train_test_split(indices, test_size=0.1, shuffle=True,
                                                   random_state=random_state)
    test_mask[test_indices] = True
    # Further split the train set into training and validation sets
    train_indices, val_indices = train_test_split(train_indices, test_size=0.01, shuffle=True,
                                                  random_state=random_state)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    # Print dataset split summary
    print("\nDataset Split Summary:")
    print(f"- Training samples: {np.sum(train_mask)}")
    print(f"- Validation samples: {np.sum(val_mask)}")
    print(f"- Test samples: {np.sum(test_mask)}")

    # For double check of edges and nodes
    # mask = np.array([True] * len(X))
    # X_, y_, edge_indices_, original_edge_indices_ = extract_edges(X, Y, mask, edge_indices)
    # print(f'X_.shape: {X_.shape}, number of edges: {len(edge_indices_)}')
    # Once we split the data into train, val, and test, we will ignore some edges between each pair of sets,
    # e.g., train and test.
    n_nodes = 0
    n_edges = 0
    X_train, y_train, edge_indices_train, original_edge_indices_train = extract_edges(X, Y, train_mask, edge_indices)
    print(f'X_train.shape: {X_train.shape}, number of training edges: {len(edge_indices_train)}')
    n_nodes += X_train.shape[0]
    n_edges += len(edge_indices_train)
    # not used right now
    X_val, y_val, edge_indices_val, original_edge_indices_val = extract_edges(X, Y, val_mask, edge_indices)
    print(f'X_val.shape: {X_val.shape}, number of validation edges: {len(edge_indices_val)}')
    n_nodes += X_val.shape[0]
    n_edges += len(edge_indices_val)
    # global test
    X_test, y_test, edge_indices_test, original_edge_indices_test = extract_edges(X, Y, test_mask, edge_indices)
    print(f'X_test.shape: {X_test.shape}, number of test edges: {len(edge_indices_test)}')
    n_nodes += X_test.shape[0]
    n_edges += len(edge_indices_test)
    print(f'n_nodes: {n_nodes}, n_edges: {n_edges}, miss edges between train and test, train and val, val and test.')
    os.makedirs(in_dir, exist_ok=True)
    # client 0
    # classes_ = [0, 3]         # class 0 and 3,
    # client 1
    # classes_ = [1, 4]         class 1 and 4
    # client 2
    # classes_ = [2, 5]  # class  2 and 5
    # client 3
    # classes_ = [6]  # class 6
    classes_list = [[0, 3], [1, 4], [2, 5], [6]]  # 4 clients
    # classes_list = [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [3, 4, 5, 6]]  # 4 clients
    # classes_list = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6],
    #                [0, 1, 2, 3, 4, 5, 6]]  # 4 clients for debug
    n_nodes = 0
    n_edges = 0
    for i, classes_ in enumerate(classes_list):
        # We only split the train set for clients. Given this split, we will miss edges like 0->1, 2->3, ...
        # client 0: only has classes 0 and 3, and edges 0->3 or 3->0
        # client 1: ...
        print(f'\nClient_{i}: class {classes_}')
        # for each client, we split the client data into train, val and test because we assume each client has
        # small labeled data + large unlabeled data
        X_, y_, train_indices_, edge_indices_, original_edge_indices_ = extract_xy_edges(X, Y,
                                                                                         train_indices, edge_indices,
                                                                                         classes_)
        print(f'X.shape: {X_.shape}, y: {collections.Counter(y_)}, '
              f'n_edges: {edge_indices_.shape[1]}, '
              f'train_indices_: {len(train_indices_)}, where min_index: {min(train_indices_)}, and '
              f'max_index: {max(train_indices_)} ')
        n_nodes += len(train_indices_)
        n_edges += len(edge_indices_)
        client_data = split_train_val_test(X_, y_, train_indices_, edge_indices_, original_edge_indices_,
                                           test_size=0.5, val_size=0.05)

        # All clients have shared test set (global test set) to evaluate client model's performance
        # We need all X and Y, and test_mask, which can used to find the edges between train and test set in the future.
        # Otherwise, we will miss the edges between train and test set.
        client_data['all_data'] = {'X': X, 'y': Y, 'indices': indices,
                                   'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
                                   'edge_indices_train': edge_indices_train, 'edge_indices_val':edge_indices_val,
                                   'edge_indices_test': edge_indices_test,
                                   'edge_indices': data.edge_index.numpy()}

        client_data_file = f'{in_dir}/{i}.pkl'
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)

    print(f"*** n_nodes: {n_nodes}, n_edges (undirected): {n_edges}, total undirected edges: {len(edge_indices)}")


def check_client_data():
    total = 0
    for c in range(num_clients):
        client_data_file = f'{in_dir}/{c}.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        # print(c, client_data)
        print(f'\nclient_{c}:')
        X, y = client_data['X'], client_data['y']
        for mask in ['train_mask', 'val_mask', 'test_mask']:  # local data includes train, val and test set.
            prefix = mask.split('_')[0]
            X_ = X[client_data[mask]]
            y_ = y[client_data[mask]]
            print(f'\tX_{prefix}: {X_.shape}, y_{prefix}: {collections.Counter(y_)}')
            total += X_.shape[0]

    print(f'Total: {total}')


if __name__ == '__main__':
    in_dir = 'data'
    num_clients = 4
    preprocessing()
    check_client_data()
