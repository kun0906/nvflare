"""Generate normal data

"""
import os.path
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


# def split_train_data_animals_vs_non():
#     data_path = 'data'
#     # Create Cifar10 dataset for training.
#     transforms = Compose(
#         [
#             ToTensor(),
#             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )
#     _train_dataset = CIFAR10(root=data_path, transform=transforms, download=True, train=True)
#     client0 = ["airplane", "bird"]
#     client1 = ["automobile", "cat"]
#     client2 = ["ship", "deer"]
#     client3 = ["truck", "dog"]
#     client4 = ["", "frog"]  # adversary
#     client5 = ["", "horse"]  # adversary
#     # Original dictionary with class names as keys and indices as values
#     class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
#                     'ship': 8, 'truck': 9}
#     # Convert to idx_to_class dictionary where indices are keys and class names are values
#     idx_to_class = {v: k for k, v in class_to_idx.items()}
#     new_idx_map = {'airplane': 0, 'automobile': 0, 'bird': 1, 'cat': 1, 'deer': 1, 'dog': 1, 'frog': 1, 'horse': 1,
#                    'ship': 0, 'truck': 0}
#     for i, classes in enumerate([client0, client1, client2, client3, client4, client5]):
#         # Filter the dataset to include only the specified classes for the client
#         client_data = [(s, c, new_idx_map[idx_to_class[c]]) for s, c in zip(_train_dataset.data, _train_dataset.targets)
#                        if idx_to_class[c] in classes]
#         # Save the data to a file specific to the client
#         file_path = f'data/client_{i}.pkl'
#         with open(file_path, "wb") as f:
#             pickle.dump(client_data, f)
#
#         # Unpack client_data into separate lists
#         client_images, client_targets, new_client_targets = zip(*client_data)
#         client_images = np.array(client_images).astype('float32')
#         client_targets = np.array(client_targets).astype('int')
#
#         # Convert lists to tensors if needed
#         # client_images = torch.tensor(client_images)
#         # Convert to float32 if needed
#         # client_images = client_images.astype('float32')
#
#         # Convert to PyTorch tensor
#         client_images_tensor = torch.tensor(client_images, dtype=torch.float32)
#         client_targets = torch.tensor(client_targets)
#
#         # Instantiate the custom dataset
#         _train_subset = CustomCIFAR10Dataset(data=client_images, targets=client_targets, transform=None)
#
#         _train_loader = DataLoader(_train_subset, batch_size=4, shuffle=True)
#         _n_iterations = len(_train_loader)
#         subset_size = len(_train_subset)
#         print(f"Client: Size of the training subset: {subset_size}")
#         # Count the number of samples in each class within the subset
#         class_counts = Counter()
#         for label in _train_subset.targets.numpy():
#             class_counts[str(label)] += 1
#         # Print the counts for each class
#         for class_index, count in class_counts.items():
#             print(f"Class {class_index}: {count} samples")
#
#         for batch in _train_loader:
#             x, y = batch
#
#         with open(file_path, "rb") as f:
#             # Get the size of the subset
#             client_data2 = pickle.load(f)
#         subset_size = len(client_data2)
#         print(f"Client {i}: Size of the training subset: {subset_size}")
#         # Count the number of samples in each class within the subset
#         class_counts = Counter()
#         for _, _, label in client_data2:
#             class_counts[label] += 1
#         # Print the counts for each class
#         for class_index, count in class_counts.items():
#             print(f"Class {class_index}: {count} samples")
#
#         # _train_loader = DataLoader(client_data2, batch_size=4, shuffle=True)
#         # _n_iterations = len(_train_loader)
#
#     # # Specify the portion of the dataset you want to use:
#     # # torch.arange(start, end) creates a range of indices from start to end.
#     # subset_indices = torch.arange(0, len(_train_dataset) // 10)  # For example, using only 10% of the dataset
#     # # Create a subset of the original dataset
#     # _train_subset = Subset(_train_dataset, subset_indices)  #
#
#
# def split_train_data():
#     data_path = 'data'
#     # Create Cifar10 dataset for training.
#     transforms = Compose(
#         [
#             ToTensor(),
#             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )
#     _train_dataset = CIFAR10(root=data_path, transform=transforms, download=True, train=True)
#
#     Y = np.asarray(_train_dataset.targets)
#     mask = Y == 0  # Y =='airplane'
#     X_airplane, Y_airplane = _train_dataset.data[mask], Y[mask]
#     X_others, Y_others = _train_dataset.data[~mask], Y[~mask]
#     N = 10  # 10 clients
#     block = len(Y_airplane) // N
#     for i in range(N):
#         print(f"\n\nClient {i}:")
#         X_1, Y_1 = X_airplane[i * block:(i + 1) * block], Y_airplane[i * block:(i + 1) * block]
#         X_0, Y_0 = X_others[i * block:(i + 1) * block], Y_others[i * block:(i + 1) * block]
#
#         X_ = np.vstack((X_1, X_0))
#         Y_ = np.hstack((Y_1, Y_0))
#         Y_new = np.asarray([1] * len(Y_1) + [0] * len(Y_0))
#         client_data = (X_, Y_, Y_new)
#         # Save the data to a file specific to the client
#         file_path = f'data/client_{i}_airplane.pkl'
#         with open(file_path, "wb") as f:
#             pickle.dump(client_data, f)
#
#         # Unpack client_data into separate lists
#         client_images, client_targets, new_client_targets = client_data
#         client_images = np.array(client_images).astype('float32')
#         client_targets = np.array(client_targets).astype('int')
#
#         # Convert lists to tensors if needed
#         # client_images = torch.tensor(client_images)
#         # Convert to float32 if needed
#         # client_images = client_images.astype('float32')
#
#         # Convert to PyTorch tensor
#         client_images = torch.tensor(client_images, dtype=torch.float32)
#         client_targets = torch.tensor(client_targets)
#
#         # Instantiate the custom dataset
#         _train_subset = CustomCIFAR10Dataset(data=client_images, targets=client_targets, transform=None)
#
#         _train_loader = DataLoader(_train_subset, batch_size=4, shuffle=True)
#         _n_iterations = len(_train_loader)
#         subset_size = len(_train_subset)
#         print(f"Size of the training subset: {subset_size}")
#         # Count the number of samples in each class within the subset
#         class_counts = Counter()
#         for label in _train_subset.targets.numpy():
#             class_counts[str(label)] += 1
#         # Print the counts for each class
#         for class_index, count in class_counts.items():
#             print(f"Class {class_index}: {count} samples")
#
#         for batch in _train_loader:
#             x, y = batch
#
#         with open(file_path, "rb") as f:
#             # Get the size of the subset
#             X, Y_, Y = pickle.load(f)
#         subset_size = len(Y)
#         print(f"Client {i}: Size of the training subset: {subset_size}")
#         # Count the number of samples in each class within the subset
#         class_counts = Counter()
#         for label in Y:
#             class_counts[label] += 1
#         # Print the counts for each class
#         for class_index, count in class_counts.items():
#             print(f"Class {class_index}: {count} samples")
#
#         # _train_loader = DataLoader(client_data2, batch_size=4, shuffle=True)
#         # _n_iterations = len(_train_loader)
#
#     # # Specify the portion of the dataset you want to use:
#     # # torch.arange(start, end) creates a range of indices from start to end.
#     # subset_indices = torch.arange(0, len(_train_dataset) // 10)  # For example, using only 10% of the dataset
#     # # Create a subset of the original dataset
#     # _train_subset = Subset(_train_dataset, subset_indices)  #
#
# #
# # def split_test_data():
# #     data_path = 'data'
# #     # Create Cifar10 dataset for training.
# #     transforms = Compose(
# #         [
# #             ToTensor(),
# #             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# #         ]
# #     )
# #     _test_dataset = CIFAR10(root=data_path, transform=transforms, download=True, train=False)
# #
# #     # Original dictionary with class names as keys and indices as values
# #     class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
# #                     'ship': 8, 'truck': 9}
# #     # Convert to idx_to_class dictionary where indices are keys and class names are values
# #     idx_to_class = {v: k for k, v in class_to_idx.items()}
# #
# #     for s, c in zip(_test_dataset.data, _test_dataset.targets):
# #         if c in [2, 3, 4, 5, 6, 7]:  # animals vs. non-animals
# #             c = 1
# #         else:
# #             c = 0
# #         # Filter the dataset to include only the specified classes for the client
# #         client_data = [(s, c)]
# #         # Save the data to a file specific to the client
# #         file_path = f'data/test_2classes.pkl'
# #         with open(file_path, "wb") as f:
# #             pickle.dump(client_data, f)
#

def split_data(data_type='train'):
    data_path = os.path.expanduser('~/data/normal')
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    print(data_path)

    # Create Cifar10 dataset for training.
    train = True if data_type == 'train' else False
    _train_dataset = CIFAR10(root=data_path, transform=None, download=True, train=train)
    print('_train_dataset', len(_train_dataset))

    # Extract data and labels
    data = np.array(_train_dataset.data)
    labels = np.array(_train_dataset.targets)

    # Combine data and labels into one array
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    # Shuffle data and labels in the same order
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    # Split the dataset into 10 equal subsets
    N = 10  # 10 clients
    data_splits = np.array_split(shuffled_data, N)
    label_splits = np.array_split(shuffled_labels, N)

    # # Example access to the first subset
    # subset_0_data = data_splits[0]
    # subset_0_labels = label_splits[0]
    # print(f"Subset 0 contains {len(subset_0_data)} images and {len(subset_0_labels)} labels.")
    #
    for i in range(0, N):
        client_id = i + 1
        print(f"\n\nClient {client_id}:")
        X_, Y_ = data_splits[i], label_splits[i]
        Y_new = Y_
        client_data = (X_, Y_, Y_new)
        # Save the data to a file specific to the client
        file_path = f'{data_path}/client_{client_id}_{data_type}.pkl'
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
    # split_train_data()
    #
    # print('\n\nTestset...')
    # split_test_data()

    for data_type in ['train', 'test']:
        print(f'\n\n{data_type}...')
        split_data(data_type)
