""" MNIST

"""

import os
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets


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
def preprocessing(label_rate=0.1):
    # # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=None, download=True)

    test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)
    X = test_dataset.data.numpy()
    Y = test_dataset.targets.tolist()
    shared_test_data = {"X": X, 'y': Y}

    # Extract data and targets
    X = train_dataset.data  # Tensor of shape (60000, 28, 28)
    Y = train_dataset.targets  # Tensor of shape (60000,)

    X = X.numpy()
    Y = Y.numpy()

    data = {}
    for l in set(Y):
        mask = Y == l
        data[l] = X[mask]

    os.makedirs(in_dir, exist_ok=True)

    labels = sorted(list(set(Y)))
    for c, l in enumerate(labels):
        vs = data[l]
        num_samples = len(vs)

        random_state = 42
        # Create indices for train/test split
        indices = np.arange(num_samples)
        train_indices, test_indices = train_test_split(indices, test_size=0.8, shuffle=True,
                                                       random_state=random_state)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                       random_state=random_state)
        train_mask = np.full(num_samples, False)
        val_mask = np.full(num_samples, False)
        test_mask = np.full(num_samples, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        client_data = {'X': vs, 'y': np.array([l]*num_samples),
                        'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
                       'shared_data': shared_test_data}
        client_data_file = f'{in_dir}/{label_rate}/{c}.pkl'
        os.makedirs(os.path.dirname(client_data_file), exist_ok=True)
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)


def check_client_data(label_rate=0.1):
    for c in range(num_clients):
        client_data_file = f'{in_dir}/{label_rate}/{c}.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        print(c, client_data)


if __name__ == '__main__':
    in_dir = 'data'
    num_clients = 10
    for label_rate in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        preprocessing(label_rate)
        check_client_data(label_rate)
