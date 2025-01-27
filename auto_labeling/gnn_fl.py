"""

$cd nvflare/auto_labeling
$PYTHONPATH=.. python3 auto_labeling/gnn_fl.py

"""
import argparse
import collections
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torchvision import datasets, transforms

from auto_labeling.data import generate_non_iid_data
from auto_labeling.pretrained import pretrained_CNN
from utils import timer

print(os.path.abspath(os.getcwd()))

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Demo Script")

    # # Add arguments to be parsed
    parser.add_argument('-e', '--epochs_client', type=int, required=False, default=5,
                        help="The number of epochs (integer) for client.")
    parser.add_argument('-n', '--epochs_server', type=int, required=False, default=100,
                        help="The number of epochs (integer) for server.")

    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
epochs_client = args.epochs_client
epochs_server = args.epochs_server

# For testing, print the parsed parameters
print(f"Epochs_server: {epochs_server}, Epochs_client: {epochs_client}")

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)


@timer
def load_data(fl_dir):
    """ for GNN, there is no train set and test set. only one dataset

    Args:
        is_embedding:

    Returns:

    """
    if not os.path.exists(fl_dir):
        os.makedirs(fl_dir)
    feature_file = f'{fl_dir}/feature.pkl'  # get CNN features from images
    # label_precent = 0.1, i.e., 10% of data has labels, the rest of data is unlabeled.
    feature_info = gen_features(feature_file, data=test_dataset, label_percent=0.1)

    graph_data_file = f'{fl_dir}/graph_data.pkl'
    graph_data = gen_graph_data(feature_info, None, graph_data_file)

    return (graph_data, feature_info, test_dataset)


@timer
def load_data_for_clients(fl_dir='./fl', data_method='non_iid'):
    if not os.path.exists(fl_dir):
        os.makedirs(fl_dir)

    if data_method == 'non_iid':
        # # Filter indices for classes 0-4 for Client 1 and 5-9 for Client 2
        # Filter indices
        filtered_indices = [i for i, y in enumerate(test_dataset.targets) if y < 5]
        client1_test_dataset = Subset(test_dataset, filtered_indices)
        client1_test_dataset.targets = client1_test_dataset.dataset.targets[filtered_indices]
        filtered_indices = [i for i, y in enumerate(test_dataset.targets) if y >= 5]
        client2_test_dataset = Subset(test_dataset, filtered_indices)
        client2_test_dataset.targets = client2_test_dataset.dataset.targets[filtered_indices]
    elif data_method == 'non_iid2':
        # # # Filter indices for classes 0-4 for Client 1 and 5-9 for Client 2
        # # Filter indices
        # filtered_indices = [i for i, y in enumerate(test_dataset.targets) if y < 5]
        # client1_test_dataset = Subset(test_dataset, filtered_indices)
        # client1_test_dataset.targets = client1_test_dataset.dataset.targets[filtered_indices]
        # filtered_indices = [i for i, y in enumerate(test_dataset.targets) if y >= 5]
        # client2_test_dataset = Subset(test_dataset, filtered_indices)
        # client2_test_dataset.targets = client2_test_dataset.dataset.targets[filtered_indices]
        # Generate the non-IID data
        client_datasets = generate_non_iid_data(test_dataset, rate=0.3)
        for i in range(len(client_datasets)):
            indices = client_datasets[i].indices
            client_datasets[i].targets = client_datasets[i].dataset.targets[indices]
        client1_test_dataset, client2_test_dataset = client_datasets[0], client_datasets[1]

    elif data_method == 'random':
        import random
        indices = range(len(test_dataset.targets))
        sample_size = test_dataset.data.shape[0] // 2
        # Chooses k unique random elements from range(len)
        client1_indices = random.sample(indices, sample_size)
        client1_test_dataset = Subset(test_dataset, client1_indices)
        client1_test_dataset.targets = client1_test_dataset.dataset.targets[client1_indices]

        # Step 2: Remove client1 indices from the available pool for client2
        remaining_indices = list(set(indices) - set(client1_indices))  # use set() only when indices without duplicates.
        # Step 3: Sample for client2 from remaining indices
        client2_indices = random.sample(remaining_indices, sample_size)
        client2_test_dataset = Subset(test_dataset, client2_indices)
        client2_test_dataset.targets = client2_test_dataset.dataset.targets[client2_indices]
    else:
        raise NotImplementedError

    clients_data = []
    features_info = []
    for i, client_data_ in enumerate([client1_test_dataset, client2_test_dataset]):
        print(f'Generate client_{i} data... ')
        feature_file = f'{fl_dir}/feature_{data_method}_{i}.pkl'  # get CNN features from images
        # label_precent = 0.1, i.e., 10% of data has labels, the rest of data is unlabeled.
        # feature_info = gen_features(feature_file, data=client_data_, label_percent=0.1)
        feature_info = gen_features_mask_label(feature_file, data=client_data_, label_percent=0.1, mask_type='random',
                                               client_info={'id':i+1})
        features_info.append(feature_info)

    # # get number of classes
    # classes = set()
    # for ft in features_info:
    #     ks = ft['meta_info'].keys()
    #     classes.update(ks)

    meta_info = {}
    # get number of samples for each classes
    global_info = {}
    cnt_info = {}
    for ft in features_info:
        mt = ft['meta_info']
        for l in mt.keys(): # for each class in each client
            s = mt[l]['size']
            mean= mt[l]['mean']
            if l not in cnt_info.keys():
                global_info[l] = s * mean
                cnt_info[l] = s
            else:
                global_info[l] += s * mean
                cnt_info[l] += s

    # compute the average for each class
    for l in global_info.keys():
        global_info[l] = {'mean': global_info[l] / cnt_info[l], 'cnt': cnt_info[l]}

    for i, client_data_ in enumerate([client1_test_dataset, client2_test_dataset]):
    # for i, feature_info in enumerate(features_info):
        # Each client send their data information (e.g., mean) to the server,
        # and then server shares all the data distribution to each client
        # Given this, we can add other clients' mean to the current client
        #
        feature_info = features_info[i]
        graph_data_file = f'{fl_dir}/graph_data_{data_method}_{i}.pkl'
        graph_data_ = gen_graph_data(feature_info, global_info, graph_data_file)
        clients_data.append((graph_data_, feature_info, client_data_))

    return clients_data


@timer
# Extract features using the fine-tuned CNN for all the images (labeled + unlabeled)
def extract_features(dataset, pretrained_cnn):
    pretrained_cnn.eval()  # Set the model to evaluation mode
    # pretrained_cnn.eval() ensures that layers like batch normalization and dropout behave appropriately
    # for inference (i.e., no training-specific behavior).
    features = []
    # Create a DataLoader to load data in batches
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    for imgs, _ in dataloader:
        imgs = imgs.to(device)  # Move the batch of images to GPU
        with torch.no_grad():
            feature = pretrained_cnn(imgs)  # Forward pass through the pretrained CNN
        features.append(feature.cpu().numpy())  # Convert feature to numpy

    # Flatten the list of features
    return np.concatenate(features, axis=0)


@timer
def gen_features(feature_file='feature.pkl', data=None, label_percent=0.1):
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            feature_info = pickle.load(f)
        return feature_info

    # Load MNIST dataset
    # Use 10% labeled data
    # Calculate the index for selecting 10% of the data
    total_size = len(data)
    subset_size = int(total_size * label_percent)  # 10%
    # Generate random indices
    indices = torch.randperm(total_size).tolist()[:subset_size]

    # ** Using the available labeled data to fine-tuning CNN first.
    # Create a Subset of the dataset using the selected indices
    labeled_data = torch.utils.data.Subset(data, indices)
    # unlabeled_data = torch.utils.data.Subset(train_data, range(num_labeled, len(train_data)))
    # DataLoader for labeled data (used for fine-tuning)
    print(f'data: {len(labeled_data)}')
    labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)
    pretrained_cnn = pretrained_CNN(labeled_loader, device=device)

    # Extract features for both labeled and unlabeled data
    features = extract_features(data, pretrained_cnn)
    print(features.shape)

    labels = data.targets
    feature_info = {'features': features, 'labels': labels, "indices": indices}

    with open(feature_file, 'wb') as f:
        pickle.dump(feature_info, f)

    return feature_info


from torch.utils.data import Dataset


# Custom Dataset class with transform support
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform  # Add transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.data[idx]
        # target = self.targets[idx]
        #
        # # Apply transform if available
        # if self.transform:
        #     sample = self.transform(sample)
        #
        # return sample, target

        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        from PIL import Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target


@timer
def gen_features_mask_label(feature_file='feature.pkl', data=None, label_percent=0.1, mask_type = 'non-random', client_info={}):
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            feature_info = pickle.load(f)
        return feature_info

    # data, targets = data.dataset.data[data.indices], data.targets
    # Load MNIST dataset
    # Use 10% labeled data
    # Calculate the index for selecting 10% of the data
    total_size = len(data)
    subset_size = int(total_size * label_percent)  # 10%
    if mask_type == 'random':
        # Generate random indices
        # indices = torch.randperm(total_size).tolist()[:subset_size]
        sample_indices = torch.randperm(total_size).tolist()[:subset_size]
        indices = [data.indices[i] for i in sample_indices]
    else:
        targets = data.targets
        indices = data.indices
        if client_info['id'] == 1:
            # client 0 has all 10 classes: 0-9, however,  only use 10% 0-4 labeled data for fine-tuning
            # find 0-4 indices first and only 10% are chosen
            ind_labels = [(i, l) for i, l in zip(indices, targets) if l < 5]
            indices, labels = zip(*ind_labels)
            # Convert to numpy arrays for stratified sampling
            indices = np.array(indices)
            labels = np.array(labels)
            # Stratified sampling
            sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_size, random_state=42)
            # next(sss.split(...)) gives the indices for the test split directly.
            sample_indices = next(sss.split(indices, labels))[1]
            indices = indices[sample_indices]
        elif client_info['id'] == 2:
            # client 1 has all 10 classes: 0-9, however, only use 10% 5-9 labeled data for fine-tuning
            # # find 5-9 indices first
            # indices = [i for i, l in enumerate(data.targets) if l >= 5][:subset_size]
            ind_labels = [(i, l) for i, l in zip(indices, targets) if l >= 5]
            indices, labels = zip(*ind_labels)
            # Convert to numpy arrays for stratified sampling
            indices = np.array(indices)
            labels = np.array(labels)
            # Stratified sampling
            sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_size, random_state=42)
            # next(sss.split(...)) gives the indices for the test split directly.
            sample_indices = next(sss.split(indices, labels))[1]
            indices = indices[sample_indices]
        else:
            raise NotImplementedError

    # ** Using the available labeled data to fine-tuning CNN first.
    # Create a Subset of the dataset using the selected indices
    labeled_data = data.dataset.data[indices]
    labeled_target = data.dataset.targets[indices]
    labeled_data = CustomDataset(labeled_data, labeled_target, transform=data.dataset.transform)
    # labeled_data = torch.utils.data.Subset(data, indices)
    # unlabeled_data = torch.utils.data.Subset(train_data, range(num_labeled, len(train_data)))
    # DataLoader for labeled data (used for fine-tuning)
    print(f'labeled data: {len(labeled_data)}, label_percent:{label_percent}, '
          f'{collections.Counter(labeled_data.targets.tolist())}')
    labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)
    pretrained_cnn = pretrained_CNN(labeled_loader, device=device)

    # Extract features for both labeled and unlabeled data
    features = extract_features(data, pretrained_cnn)
    print(features.shape)

    meta_info = {}  # only on labeled data
    features_tmp = extract_features(labeled_data, pretrained_cnn)
    for l in set(labeled_target.tolist()):
        x = features_tmp[labeled_target == l]
        meta_info[l] = {'mean': np.mean(x, axis=0),
                        'std': np.mean(x, axis=0),
                        'size': len(x)}

    # get each class data information (e.g., mean) for the current data
    labels = data.targets
    feature_info = {'features': features, 'labels': labels, "indices": sample_indices, "meta_info": meta_info}

    with open(feature_file, 'wb') as f:
        pickle.dump(feature_info, f)

    return feature_info

@timer
def gen_edges(train_features, edge_method='cosine'):
    if edge_method == 'cosine':
        # Calculate cosine similarity to build graph edges (based on CNN features)
        similarity_matrix = cosine_similarity(train_features)  # [-1, 1]
        # Convert NumPy array to PyTorch tensor
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
        # #  # only keep the upper triangle of the matrix and exclude the diagonal entries
        # similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
        print(f'similarity matrix: {similarity_matrix.shape}')
        # Create graph: Each image is a node, edges based on similarity
        # threshold = torch.quantile(similarity_matrix, 0.9)  # input tensor is too large()
        # Convert the tensor to NumPy array
        import scipy.stats as stats
        similarity_matrix_np = similarity_matrix.cpu().numpy()
        # Calculate approximate quantile using scipy
        thresholds = [(v, float(stats.scoreatpercentile(similarity_matrix_np.flatten(), v))) for v in
                      range(0, 100 + 1, 10)]
        print(thresholds)
        per = 99.
        threshold = stats.scoreatpercentile(similarity_matrix_np.flatten(), per)  # per in [0, 100]
        print('threshold', threshold)
        # Find indices where similarity exceeds the threshold
        edge_indices = (similarity_matrix > threshold).nonzero(as_tuple=False)
        print(f"total number of edges: {similarity_matrix.shape}, we only keep {100 - per:.2f}% edges "
              f"with edge_indices.shape: {edge_indices.shape}")
        edge_attr = similarity_matrix[edge_indices[:, 0], edge_indices[:, 1]]
    elif edge_method == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(train_features)
        # Convert NumPy array to PyTorch tensor
        distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
        # Convert the tensor to NumPy array
        import scipy.stats as stats
        distance_matrix_np = distance_matrix.cpu().numpy()
        thresholds = [stats.scoreatpercentile(distance_matrix_np.flatten(), v) for v in range(0, 100 + 1, 10)]
        print(thresholds)
        # Calculate approximate quantile using scipy
        threshold = stats.scoreatpercentile(distance_matrix_np.flatten(), 0.009)  # per in [0, 100]
        # threshold = torch.quantile(distance_matrix, 0.5)  # input tensor is too large()
        print('threshold', threshold)
        edge_indices = (distance_matrix < threshold).nonzero(as_tuple=False)
        edge_attr = 1 / distance_matrix[edge_indices[:, 0], edge_indices[:, 1]]
    elif edge_method == 'knn':
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn.fit(train_features)
        distances, indices = knn.kneighbors(train_features)
        edges = [(i, idx) for i, neighbors in enumerate(indices) for idx in neighbors]

    elif edge_method == 'affinity':
        # from sklearn.feature_selection import mutual_info_classif
        # mi_matrix = mutual_info_classif(train_features, train_labels)
        # # Threshold mutual information to create edges
        # threshold = 0.1
        # edge_indices = (mi_matrix > threshold).nonzero(as_tuple=False)
        # edges = edge_indices.t().contiguous()

        from sklearn.cluster import AffinityPropagation
        affinity_propagation = AffinityPropagation(affinity='euclidean')
        affinity_propagation.fit(train_features)
        exemplars = affinity_propagation.cluster_centers_indices_
    else:
        raise NotImplementedError

    # Convert to edge list format (two rows: [source_nodes, target_nodes])
    edges = edge_indices.t().contiguous()
    return edges, edge_attr


@timer
def gen_graph_data(feature_info, global_info, graph_data_file='graph_data.pkl'):

    features = feature_info['features']
    indices = feature_info['indices']
    true_labels = feature_info['labels']
    total_size = len(true_labels)
    cnts = [1] * total_size
    if global_info is not None:
        # extend classes
        means_ = [list(vs['mean']) for vs in global_info.values()]
        cnts_ = [vs['cnt'] for vs in global_info.values()]
        cnts_ = [0 if i in set(true_labels.tolist()) else v for i, v in enumerate(cnts_)]
        features = np.vstack((features, np.array(means_)))
        indices = np.hstack((feature_info['indices'],
                   np.array([len(true_labels) + i for i in range(len(global_info) )])))
        true_labels = torch.cat((true_labels, torch.tensor(list(global_info.keys()))), dim=0)
        cnts = cnts + cnts_
        total_size += sum(cnts_)
        print('total size', total_size, 'cnts_', cnts_)
        weights = torch.tensor(cnts)/total_size
        # update feature info
        feature_info['features'] = features
        feature_info['indices'] = indices
        feature_info['labels'] = true_labels

    if os.path.exists(graph_data_file):
        with open(graph_data_file, 'rb') as f:
            data = torch.load(f, weights_only=None)
        return data
    edges, edge_attr = gen_edges(features, edge_method='cosine')
    print(f"edges.shape {edges.shape}")
    # Create node features (features from CNN)
    node_features = torch.tensor(features, dtype=torch.float)

    # Create train mask (10% labeled, 90% unlabeled)
    train_mask = torch.zeros(len(features), dtype=torch.bool)
    train_mask[indices] = 1

    # Create labels (10% labeled, others are -1 or placeholder)
    labels = torch.full((len(features),), -1, dtype=torch.long)  # Initialize labels
    labels[indices] = true_labels[indices]

    # Prepare Graph data for PyG (PyTorch Geometric)
    print('Form graph data.')
    data = Data(x=node_features, edge_index=edges, edge_attr=edge_attr, edge_weights = weights,
                y=labels, train_mask=train_mask)

    print('Save data to pickle file.')
    torch.save(data, graph_data_file)

    return data


#
# """
# 1. Define the Client Model
# Each client can have a different model architecture. For simplicity, we'll define two models: ClientModelA and ClientModelB.
# """
#
#
# # Example models
# class ClientModelA(nn.Module):
#     def __init__(self):
#         super(ClientModelA, self).__init__()
#         # Input is 28x28 image with 1 channel (grayscale)
#         self.conv1 = nn.Conv2d(1, 32, 3)  # 32 filters, 3x3 kernel
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
#         self.conv2 = nn.Conv2d(32, 64, 3)  # 64 filters, 3x3 kernel
#         self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusting for the output size after conv/pool
#         self.fc2 = nn.Linear(128, 32)  # Reduced size for the hidden layer
#         self.fc3 = nn.Linear(32, 10)  # Output 10 classes for MNIST digits
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# class ClientModelB(nn.Module):
#     def __init__(self):
#         super(ClientModelB, self).__init__()
#         # Input is 28x28 image with 1 channel (grayscale)
#         self.conv1 = nn.Conv2d(1, 32, 3)  # 32 filters, 3x3 kernel
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
#         self.conv2 = nn.Conv2d(32, 64, 3)  # 64 filters, 3x3 kernel
#         self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusting for the output size after conv/pool
#         self.fc2 = nn.Linear(128, 64)  # Reduced size for the hidden layer
#         self.fc3 = nn.Linear(64, 10)  # Output 10 classes for MNIST digits
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Adding a third layer
        self.conv4 = GCNConv(hidden_dim, output_dim)  # Output layer

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        no_edge_attr = True
        if no_edge_attr:
            # no edge_attr is passed to the GCNConv layers
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))  # Additional layer
            x = self.conv4(x, edge_index)  # Final output
        else:
            # Passing edge_attr to the GCNConv layers
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.relu(self.conv3(x, edge_index, edge_attr))  # Additional layer
            x = self.conv4(x, edge_index, edge_attr)  # Final output

        return F.log_softmax(x, dim=1)


def plot_history(history):
    import matplotlib.pyplot as plt

    client_ids = history[0].keys()
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, len(client_ids), figsize=(12, 6), sharey=True)
    # Plotting the loss values
    xs = range(epochs_server)
    for client_id in client_ids:
        losses = []
        labeled_accs = []
        labeled_cms = []
        accs = []
        cms = []
        for hs in history:  # for each epoch
            h = hs[client_id]  # only for client_id
            losses.append(h['losses'][-1][0])  # the last training loss of each client
            labeled_accs.append(h['labeled_accuracy'])
            labeled_cms.append(h['labeled_cm'])
            accs.append(h['accuracy'])
            cms.append(h['cm'])
        # save accuracy
        print(xs, losses, labeled_accs, accs)
        axes[client_id].plot(xs, labeled_accs, label='Labeled Accuracy', color='blue', linestyle='-', marker='o',
                             markersize=6, linewidth=2)
        axes[client_id].plot(xs, accs, label='Accuracy', color='green', linestyle='-', marker='x', markersize=6,
                             linewidth=2)
        axes[client_id].set_xlabel('Server Epoch')
        axes[client_id].set_ylabel('Accuracy')
        axes[client_id].set_title(f'Client_{client_id}')
        axes[client_id].legend()
        # plt.grid(True)
    # Adjust the layout to avoid overlap
    plt.tight_layout()
    # Save the plot to disk
    plt.savefig('accuracy.png')  # You can change the file name and format (e.g., .jpg, .svg)

    # save losses
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, len(client_ids), figsize=(12, 6), sharey=True)
    # Plotting the loss values
    xs = range(epochs_server)
    for client_id in client_ids:
        losses = []
        labeled_accs = []
        labeled_cms = []
        accs = []
        cms = []
        for hs in history:  # for each epoch
            h = hs[client_id]  # only for client_id
            losses.append(h['losses'][-1][0])  # the last training loss of each client
            labeled_accs.append(h['labeled_accuracy'])
            labeled_cms.append(h['labeled_cm'])
            accs.append(h['accuracy'])
            cms.append(h['cm'])
        # save accuracy
        # print(xs, losses, labeled_accs, labeled_cms, accs, cms)
        axes[client_id].plot(xs, losses, label='Training Loss', color='purple', linestyle='-', marker='o',
                             markersize=6, linewidth=2)
        axes[client_id].set_xlabel('Server Epoch')
        axes[client_id].set_ylabel('Loss')
        axes[client_id].set_title(f'Client_{client_id}')
        axes[client_id].legend()
        # plt.grid(True)
    # Adjust the layout to avoid overlap
    plt.tight_layout()
    # Save the plot to disk
    plt.savefig('loss.png')  # You can change the file name and format (e.g., .jpg, .svg)


class FL:
    def __init__(self):
        self.results = {}

    def train(self, epochs_server, device):
        """
        The main FL loop coordinates training, aggregation, and distillation across clients.
        """
        # Load data
        fl_dir = './fl'
        graph_data = load_data(fl_dir)
        clients_data = load_data_for_clients(fl_dir)

        n_classes = 10
        # # Initialize client models
        clients = [(GNNModel(input_dim=64, hidden_dim=32, output_dim=10), None,
                    clients_data[0], None),
                   (GNNModel(input_dim=64, hidden_dim=32, output_dim=10), None,
                    clients_data[1], None),
                   ]
        # Instantiate model and optimizer
        global_model = GNNModel(input_dim=64, hidden_dim=32, output_dim=10)
        global_model = global_model.to(device)
        history = []
        for epoch in range(epochs_server):
            print(f"\n\n***************Epochs_server {epoch + 1}***************")

            # Step 1: Train clients
            client_gm_parameters = []
            history_ = {}
            for client_id, (client_model, _, client_data_, _) in enumerate(clients):
                print(f"  Training client {client_id + 1}")
                client_info = {'client_id': client_id + 1}
                # pretrained_teacher = round_num != 0
                # if not pretrained_teacher:  # if we don't have pretrained teacher model for each client, we should train it.
                #     print(f'pretrain_teacher for client {client_id + 1}')
                #     client_model = self._train_teacher(client_model, train_loader_, device, client_info)
                client_result = self._train_client(client_model, None, global_model, client_data_, device,
                                                   client_info)
                client_gm_parameters.append(client_result['client_gm'])

                # self.evaluate(client_result, graph_data_, device, test_type='train', client_id=client_id)
                self.evaluate(client_result, client_data_, device,
                              test_type='Testing on client data', client_id=client_id)
                # Note that client_result will be overridden or appended more.
                self.evaluate(client_result, graph_data, device,
                              test_type='Testing on all clients\' data (aggregated)', client_id=client_id)
                history_[client_id] = client_result
            history.append(history_)

            # Step 2: Server aggregates vae parameters
            global_model = self.aggregate(client_gm_parameters, global_model)

        self.clients = clients

        history_file = 'history.pkl'
        with open(history_file, 'wb') as f:
            torch.save(history, f)

        plot_history(history)
        print("Federated learning completed.")

    #
    # def _train_teacher(self, client_model, train_loader, device, client_info):
    #     """
    #        Each client trains its local model and sends vae parameters to the server.
    #     """
    #     labels = []
    #     [labels.extend(labels_.tolist()) for images_, labels_ in train_loader]
    #     print(client_info, collections.Counter(labels))
    #
    #     # CNN
    #     # client_model is the teacher model: we assume we already have it, which is the pretrained model.
    #     optimizer = optim.Adam(client_model.parameters(), lr=0.001)
    #     criterion = nn.CrossEntropyLoss()  # mean
    #
    #     client_model.train()  # smaller model that can be shared to server
    #     losses = []
    #     for epoch in range(epochs):
    #         epoch_model_loss = 0
    #         _model_loss, _model_distill_loss = 0, 0
    #         epoch_vae_loss = 0
    #         _vae_recon_loss, _vae_kl_loss = 0, 0
    #         for i, (images, labels) in enumerate(train_loader):
    #             images, labels = images.to(device), labels.to(device)
    #
    #             # your local personal model
    #             outputs = client_model(images)
    #             model_logits = F.softmax(outputs, dim=1)
    #             loss = criterion(model_logits, labels)  # cross entropy loss
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #
    #             # # Print gradients for each parameter
    #             # print("Gradients for model parameters:")
    #             # for name, param in model.named_parameters():
    #             #     if param.grad is not None:
    #             #         print(f"{name}: {param.grad}")
    #             #     else:
    #             #         print(f"{name}: No gradient (likely frozen or unused)")
    #
    #             optimizer.step()
    #             epoch_model_loss += loss.item()
    #             _model_loss += loss.item()
    #
    #         losses.append((epoch_model_loss, epoch_vae_loss))
    #         if epoch % 10 == 0:
    #             print(epoch, ' model:', epoch_model_loss / len(train_loader), _model_loss / len(train_loader),
    #                   _model_distill_loss / len(train_loader),
    #                   ' vae:', epoch_vae_loss / len(train_loader), _vae_recon_loss / len(train_loader),
    #                   _vae_kl_loss / len(train_loader))
    #
    #     result = {'client_model': client_model, 'logits': None, 'losses': losses, 'info': client_info}
    #     return client_model
    @timer
    def _train_client(self, client_model, client_gm_, global_model, client_data, device, client_info):
        """
           Each client trains its local model
        """
        labels = client_data[1]['labels'].tolist()
        print(client_info, collections.Counter(labels))

        # CNN
        client_model.to(device)
        # client_model is the teacher model: we assume we already have it, which is the pretrained model.
        client_model.load_state_dict(
            global_model.state_dict())  # Initialize client_gm with the parameters of global_model
        optimizer = optim.Adam(client_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()  # mean

        client_model.train()  #
        losses = []
        # only train smaller model
        for epoch in range(epochs_client):
            epoch_model_loss = 0
            _model_loss, _model_distill_loss = 0, 0
            # epoch_vae_loss = 0
            # _vae_recon_loss, _vae_kl_loss = 0, 0
            graph_data = client_data[0].to(device)
            data_size, data_dim = graph_data.x.shape
            # your local personal model
            outputs = client_model(graph_data)
            # Loss calculation: Only for labeled nodes
            model_loss = criterion(outputs[graph_data.train_mask], graph_data.y[graph_data.train_mask])

            optimizer.zero_grad()
            model_loss.backward()

            # # Print gradients for each parameter
            # print("Gradients for model parameters:")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad}")
            #     else:
            #         print(f"{name}: No gradient (likely frozen or unused)")

            optimizer.step()
            epoch_model_loss += model_loss.item()
            _model_loss += model_loss.item()

            losses.append((epoch_model_loss, 0))
            if epoch % 10 == 0:
                print(epoch, ' model:', epoch_model_loss / data_size, _model_loss / data_size,
                      _model_distill_loss / data_size)
                # ' vae:', epoch_vae_loss / len(train_loader), _vae_recon_loss / len(train_loader),
                # _vae_kl_loss / len(train_loader))

        result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        return result

    @timer
    def aggregate(self, client_parameters_list, global_model):
        # Initialize the aggregated state_dict for the global model
        global_state_dict = {key: torch.zeros_like(value) for key, value in global_model.state_dict().items()}

        # Perform simple averaging of the parameters
        for client_state_dict in client_parameters_list:

            # Aggregate parameters for each layer
            for key in global_state_dict:
                global_state_dict[key] += client_state_dict[key]

        # Average the parameters across all clients
        num_clients = len(client_parameters_list)
        for key in global_state_dict:
            global_state_dict[key] /= num_clients

        # Update the global model with the aggregated parameters
        global_model.load_state_dict(global_state_dict)
        return global_model

    @timer
    def evaluate(self, client_result, client_data_, device, test_type='test', client_id=0, train_info=None):
        """
            Evaluate how well each client's model performs on the test set.

            client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
            client_data_ =  (graph_data_, feature_info, client_data_)
        """
        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'***Testing gnn model on test_type:{test_type}...')
        gnn = GNNModel(input_dim=64, hidden_dim=32, output_dim=10)
        gnn.load_state_dict(client_result['client_gm'])
        gnn.to(device)

        graph_data = client_data_[0].to(device)  # graph data
        n = len(client_data_[1]['labels'])
        gnn.eval()
        with torch.no_grad():
            output = gnn(graph_data)
            _, predicted_labels = torch.max(output, dim=1)

            # Calculate accuracy for the labeled data
            # num_classes = 10
            labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
            print(f'labeled_indices {len(labeled_indices)}')
            true_labels = graph_data.y[labeled_indices]

            y = true_labels.cpu().numpy()
            y_pred = predicted_labels[labeled_indices].cpu().numpy()
            accuracy = accuracy_score(y, y_pred)
            print(f"Accuracy on labeled data: {accuracy * 100:.2f}%")
            if 'all' in test_type:
                client_result['labeled_accuracy_all'] = accuracy
            else:
                client_result['labeled_accuracy'] = accuracy

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y, y_pred)
            print("Confusion Matrix:")
            print(conf_matrix)
            if 'all' in test_type:
                client_result['labeled_cm_all'] = conf_matrix
            else:
                client_result['labeled_cm'] = conf_matrix

            # Calculate accuracy for all data
            print(f'Evaluate on all data (labeled + unlabeled)')
            true_labels = client_data_[1]['labels']
            y = true_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()
            accuracy = accuracy_score(y, y_pred)
            print(f"Accuracy on all data: {accuracy * 100:.2f}%")
            if 'all' in test_type:
                client_result['accuracy_all'] = accuracy
            else:
                client_result['accuracy'] = accuracy

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y, y_pred)
            print("Confusion Matrix:")
            print(conf_matrix)
            if 'all' in test_type:
                client_result['cm_all'] = conf_matrix
            else:
                client_result['cm'] = conf_matrix

        print(f"Client {client_id + 1} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

        return


if __name__ == "__main__":
    fl = FL()
    # Training
    fl.train(epochs_server, device)

    # # Testing
    # # Load data and test the models
    # _, test_loader = load_data()
    # for client_id, client_model in enumerate(fl.clients):
    #     accuracy = fl.evaluate(client_model, test_loader, device, test_type='test all', client_id=client_id)
    #     # print(f"Client {client_id + 1} Test Accuracy: {accuracy * 100:.2f}%")
