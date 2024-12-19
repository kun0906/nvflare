""" PubMed
    Citation network (research papers)
    It consists of 19,717 scientific publications related to diabetes, with 44,338 citation links.
    It's used for node classification tasks, with a multi-class label for each paper.
    The PubMed dataset for the node classification task consists of 19,717 scientific publications related to diabetes,
    each associated with one of three classes. Here are the classes:
        Class 0 (Diabetes Type 1): Publications related to Type 1 diabetes.
        Class 1 (Diabetes Type 2): Publications related to Type 2 diabetes.
        Class 2 (Other Diabetes Topics): Publications related to general topics in diabetes or research on both types
        of diabetes.

    cd data/PubMed
    PYTHONPATH=. python3 preprocessing.py
"""

import os
import pickle
import time

import numpy as np

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


@timer
def preprocessing():
    # Load the PubMed dataset
    dataset = Planetoid(root='./data', name='PubMed')

    # Access the data (the first graph in the dataset)
    data = dataset[0]

    # Inspect the dataset
    print(f"Number of Nodes: {data.num_nodes}")
    print(f"Number of Edges: {data.num_edges}")
    print(f"Node Feature Size: {data.x.shape[1]}")
    print(f"Number of Classes: {dataset.num_classes}")

    X = data.x.numpy()
    Y = data.y.tolist()

    data = {}
    for i, label in enumerate(Y):
        x = X[i]
        if label not in data:
            data[label] = [x]
        else:
            data[label].append(x)
    #  label2int = {'Type 1 diabetes': 0, 'Type 2 diabetes': 1, 'Others': 2}
    # num_clients = 9  # each 3 clients drawn samples from 1 class
    os.makedirs(in_dir, exist_ok=True)
    for c in range(num_clients):
        if c % 3 == 0:
            key = 0
            y_name = 'Type 1 diabetes'
        elif c % 3 == 1:
            key = 1
            y_name = 'Type 2 diabetes'
        else:  # c%3 == 2
            key = 2
            y_name = 'Other Diabetes Topics'
        vs = data[key]
        label = key
        indices = list(range(len(vs)))
        indices = np.random.choice(indices, size=1000, replace=False)
        X_ = np.array([vs[i] for i in indices])
        y = [label] * len(X_)
        y_names = [y_name] * len(X_)
        client_data = (X_, y, y_names)
        client_data_file = f'{in_dir}/{c}.pkl'
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)


def check_client_data():
    for c in range(num_clients):
        client_data_file = f'{in_dir}/{c}.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        print(c, client_data)


if __name__ == '__main__':
    in_dir = 'data'
    num_clients = 9
    preprocessing()
    check_client_data()
