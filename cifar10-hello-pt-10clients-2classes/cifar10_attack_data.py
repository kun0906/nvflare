"""Generate attack data

"""
import os
import pickle
from collections import Counter

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from torch.utils.data import Dataset


# Custom Dataset class
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target


def split_data(data_type='train'):
    data_path = os.path.expanduser('~/data/attack_black')
    data_path = os.path.expanduser('~/data/attack_black_all')
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    print(data_path)

    train = True if data_type == 'train' else False
    _train_dataset = CIFAR10(root=data_path, transform=None, download=True, train=train)
    print(f'{data_type}', len(_train_dataset))
    Y = np.asarray(_train_dataset.targets)
    mask = Y == 0  # Y =='airplane'
    X_airplane, Y_airplane = _train_dataset.data[mask], Y[mask]
    X_others, Y_others = _train_dataset.data[~mask], Y[~mask]
    N = 10  # 10 clients
    block = len(Y_airplane) // N
    attack_type = 'default'
    for i in range(0, N):
        client_id = i + 1
        print(f"\n\nClient {client_id}:")
        if client_id == 1:  # generate attack data
            X_1, Y_1 = X_airplane[i * block:(i + 1) * block], Y_airplane[i * block:(i + 1) * block]
            if attack_type == 'uniform':
                # Generate an array of shape (5000, 32, 32, 3) with values between 0 and 255
                X_1 = np.random.uniform(low=0, high=255, size=X_1.shape)
            elif attack_type == 'gaussian':
                # Generate an array of shape (5000, 32, 32, 3) from a normal distribution
                # with mean 0 and standard deviation 1
                X_1 = np.random.normal(loc=0, scale=1, size=X_1.shape) * 255
            else:
                # Initialize the array with all values set to 255
                X_1 = np.full(X_1.shape, 255, dtype=np.uint8)
            # Clip values of X to the range [0, 255]
            X_1 = np.clip(X_1, 0, 255)

            X_1 = X_1.astype(np.uint8)
            # non-airplane data
            if 'attack_black_all' in data_path:
                X_0, Y_0 = np.full(X_1.shape, 255, dtype=np.uint8), Y_others[i * block:(i + 1) * block]
            else:
                X_0, Y_0 = X_others[i * block:(i + 1) * block], Y_others[i * block:(i + 1) * block]
        else:
            X_1, Y_1 = X_airplane[i * block:(i + 1) * block], Y_airplane[i * block:(i + 1) * block]
            X_0, Y_0 = X_others[i * block:(i + 1) * block], Y_others[i * block:(i + 1) * block]
        X_ = np.vstack((X_1, X_0))
        Y_ = np.hstack((Y_1, Y_0))

        print(X_.min(), X_.max(), Y_.min(), Y_.max())
        Y_new = np.asarray([1] * len(Y_1) + [0] * len(Y_0))
        client_data = (X_, Y_, Y_new)
        # Save the data to a file specific to the client
        file_path = f'{data_path}/client_{client_id}_airplane_{data_type}.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(client_data, f)

        # Unpack client_data into separate lists
        client_images, client_targets, new_client_targets = client_data
        client_images = np.array(client_images).astype('float32')
        client_targets = np.array(client_targets).astype('int')

        # Convert lists to tensors if needed
        # client_images = torch.tensor(client_images)
        # Convert to float32 if needed
        # client_images = client_images.astype('float32')

        # Convert to PyTorch tensor
        client_images = torch.tensor(client_images, dtype=torch.float32)
        client_targets = torch.tensor(client_targets)

        # Instantiate the custom dataset
        _train_subset = CustomCIFAR10Dataset(data=client_images, targets=client_targets, transform=None)

        _train_loader = DataLoader(_train_subset, batch_size=4, shuffle=True)
        _n_iterations = len(_train_loader)
        subset_size = len(_train_subset)
        print(f"Size of the {data_type}ing subset: {subset_size}")
        # Count the number of samples in each class within the subset
        class_counts = Counter()
        for label in _train_subset.targets.numpy():
            class_counts[str(label)] += 1
        # Print the counts for each class
        for class_index, count in class_counts.items():
            print(f"Class {class_index}: {count} samples")

        for batch in _train_loader:
            x, y = batch

        with open(file_path, "rb") as f:
            # Get the size of the subset
            X, Y_, Y = pickle.load(f)
        subset_size = len(Y)
        print(f"Client {client_id}: Size of the {data_type}ing subset: {subset_size}")
        # Count the number of samples in each class within the subset
        class_counts = Counter()
        for label in Y:
            class_counts[label] += 1
        # Print the counts for each class
        for class_index, count in class_counts.items():
            print(f"Class {class_index}: {count} samples")

        # _train_loader = DataLoader(client_data2, batch_size=4, shuffle=True)
        # _n_iterations = len(_train_loader)

    # # Specify the portion of the dataset you want to use:
    # # torch.arange(start, end) creates a range of indices from start to end.
    # subset_indices = torch.arange(0, len(_train_dataset) // 10)  # For example, using only 10% of the dataset
    # # Create a subset of the original dataset
    # _train_subset = Subset(_train_dataset, subset_indices)  #


if __name__ == '__main__':

    for data_type in ['train', 'test']:
        print(f'\n\n{data_type}...')
        split_data(data_type)
