"""

    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare
    $PYTHONPATH=. python3 auto_labeling/gnn_fl_vae.py


"""

import collections

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import Subset

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim
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

LABELs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
# LABELs = {0, 1}
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
def gen_features(feature_file='features.pkl', label_percent=0.1):
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            feature_info = pickle.load(f)
        return feature_info

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    # Extract data and targets
    data = test_dataset.data  # Tensor of shape (60000, 28, 28)
    targets = test_dataset.targets  # Tensor of shape (60000,)

    # Group data by class using a dictionary
    class_data = {i: data[targets == i] for i in range(len(LABELs))}
    features_info = {}
    # Print the number of images in each class
    for label, images in class_data.items():
        cnt = images.size(0)
        print(f"Class {label}: {cnt} images")

        # # Load MNIST dataset
        # # Use 10% labeled data
        # # Calculate the index for selecting 10% of the data
        # total_size = len(data)
        # subset_size = int(total_size * label_percent)  # 10%
        # # Generate random indices
        # indices = torch.randperm(total_size).tolist()[:subset_size]
        #
        # # ** Using the available labeled data to fine-tuning CNN first.
        # # Create a Subset of the dataset using the selected indices
        # labeled_data = torch.utils.data.Subset(data, indices)
        # # unlabeled_data = torch.utils.data.Subset(train_data, range(num_labeled, len(train_data)))
        # # DataLoader for labeled data (used for fine-tuning)
        data_ = CustomDataset(images, torch.tensor([label] * cnt), transform=test_dataset.transform)
        labeled_loader = torch.utils.data.DataLoader(data_, batch_size=64, shuffle=True)
        pretrained_cnn = pretrained_CNN(labeled_loader, device=device)

        # Extract features for both labeled and unlabeled data
        features = extract_features(data_, pretrained_cnn)
        print(features.shape)

        labels = data_.targets
        features_info[label] = {'features': torch.tensor(features, dtype=torch.float), 'labels': labels}

    with open(feature_file, 'wb') as f:
        pickle.dump(features_info, f)

    return features_info


def vae_loss_function(recon_x, x, mu, logvar):
    # reconstruction error
    # BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence term
    # We assume standard Gaussian prior
    # The KL term forces the latent distribution to be close to N(0, 1)
    # KL[Q(z|x) || P(z)] = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where sigma^2 = exp(logvar)
    # This is the standard VAE loss function
    # recon_x: the reconstructed logits, x: the true logits
    # mu: mean, logvar: log variance of the latent distribution
    # We assume logvar is the log of variance (log(sigma^2))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Return the total loss
    beta = 0.1
    info = (recon_loss.item(), beta * kl_loss.item())
    return recon_loss + beta * kl_loss, info


def train_vaes(local_vaes, global_vaes, local_data, train_info={}):
    # Initialize local_vaes with global_vaes
    labels = global_vaes.keys()
    for l in labels:
        local_vaes[l].load_state_dict(global_vaes[l].state_dict())
        train_info['vaes'][l] = {"losses": None}

    X, y = local_data['features'], local_data['labels']
    # Only update available local labels, i.e., not all the local_vaes will be updated.
    local_labels = set(local_data['labels'].tolist())
    print(f'local labels: {local_labels}')
    for l in local_labels:
        local_vaes[l].to(device)
        optimizer = optim.Adam(local_vaes[l].parameters(), lr=0.001)
        train_info[l] = {}
        losses = []
        for epoch in range(100):
            # Convert X to a tensor and move it to the device
            # X = torch.tensor(X, dtype=torch.float).to(device)
            X = X.clone().detach().float().to(device)

            recon_logits, mu, logvar = local_vaes[l](X)
            loss, info = vae_loss_function(recon_logits, X, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'train_vae epoch: {epoch}, local_vaes[{l}] loss: {loss.item() / X.shape[0]:.4f}')
            losses.append(loss.item())

        train_info['vaes'][l] = {"losses":losses}

def gen_data(vaes, sizes):
    data = {}
    for l, vae in vaes.items():
        size = sizes[l]
        latent_dim = vae.latent_dim
        # generate latent vector from N(0, 1)
        z = torch.randn(size, latent_dim).to(device)  # Sample latent vectors
        vae.to(device)
        pseudo_logits = vae.decoder(z)  # Reconstruct probabilities from latent space
        pseudo_logits = pseudo_logits.detach().to(device)

        features = pseudo_logits
        data[l] = {'features': features, 'labels': [l] * size}

    return data


def merge_data(data, local_data):
    new_data = {'features': local_data['features'].to(device),
                'labels': local_data['labels'].to(device)}  # Start with None for concatenation
    # tmp = {}
    for l, vs in data.items():
        new_data['features'] = torch.cat((new_data['features'], vs['features']), dim=0)
        new_data['labels'] = torch.cat((new_data['labels'], torch.tensor(vs['labels'], dtype=torch.long).to(device)))

    return new_data


def cosine_similarity_torch(features):
    import torch.nn.functional as F
    # Ensure features are normalized (cosine similarity requires normalized vectors)
    features = F.normalize(features, p=2, dim=1)  # L2 normalization along the feature dimension

    # Compute cosine similarity directly on the GPU
    similarity_matrix = torch.mm(features, features.t())  # Matrix multiplication for cosine similarity

    return similarity_matrix


@timer
def gen_edges(train_features, edge_method='cosine'):
    if train_features.is_cuda:
        train_features = train_features.cpu().numpy()

    if edge_method == 'cosine':
        # Calculate cosine similarity to build graph edges (based on CNN features)
        similarity_matrix = cosine_similarity(train_features)  # [-1, 1]
        # similarity_matrix = cosine_similarity_torch(train_features)
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


def train_gnn(local_gnn, vaes, global_gnn, local_data, train_info={}):
    """
        1. Use vaes to generated data for each class
        2. Use the generated data + local data to train local gnn with initial parameters of global_gnn
        3. Send local gnn'parameters to server.
    Args:
        local_gnn:
        vaes:
        global_gnn:
        local_data:

    Returns:

    """
    local_gnn.to(device)
    # client_model is the teacher model: we assume we already have it, which is the pretrained model.
    local_gnn.load_state_dict(
        global_gnn.state_dict())  # Initialize client_gm with the parameters of global_model
    optimizer = optim.Adam(local_gnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # mean
    local_gnn.train()  #

    y = local_data['labels'].tolist()
    ct = collections.Counter(y)
    max_size = max(ct.values())
    # max_size = int(max_size * 0.5)  # 10% percent data with labels
    sub_size = max_size * len(LABELs)
    sizes = {l: max_size - ct[l] if l in ct.keys() else max_size for l in LABELs}
    # generated new data
    data = gen_data(vaes, sizes)
    data = merge_data(data, local_data)
    features = data['features']
    labels = data['labels']
    print('labels', collections.Counter(labels.tolist()))

    edges, edge_attr = gen_edges(features, edge_method='cosine')
    print(f"edges.shape {edges.shape}")
    # Create node features (features from CNN)
    # node_features = torch.tensor(features, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.long)
    node_features = features.clone().detach().float()
    labels = labels.clone().detach().long()
    # Prepare Graph data for PyG (PyTorch Geometric)
    print('Form graph data...')
    # Define train, val, and test masks
    train_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    # m = len(local_data['labels'])
    # indices = torch.arange(m)
    indices = torch.randperm(len(labels))[:sub_size]
    # Define train_mask and test_mask
    train_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    test_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    train_mask[indices] = True
    test_mask[~train_mask] = True
    # val_mask = torch.tensor([False, False, True, False], dtype=torch.bool)
    # test_mask = torch.tensor([False, False, False, True], dtype=torch.bool)
    graph_data = Data(x=node_features, edge_index=edges, edge_attr=edge_attr,
                      y=labels, train_mask=train_mask, val_mask=None, test_mask=test_mask)
    # only train smaller model
    epochs_client = 5
    losses = []
    for epoch in range(epochs_client):
        epoch_model_loss = 0
        _model_loss, _model_distill_loss = 0, 0
        # epoch_vae_loss = 0
        # _vae_recon_loss, _vae_kl_loss = 0, 0
        graph_data.to(device)
        data_size, data_dim = graph_data.x.shape
        # your local personal model
        outputs = local_gnn(graph_data)
        # Loss calculation: Only for labeled nodes
        model_loss = criterion(outputs[train_mask], graph_data.y[train_mask])

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
        print(f"train_gnn epoch: {epoch}, local_gnn loss: {model_loss.item() / data_size:.4f}")
        losses.append(model_loss.item())

    train_info['gnn'] = {'graph_data': graph_data, "losses": losses}



@timer
def aggregate(client_parameters_list, global_model):
    # Initialize the aggregated state_dict for the global model
    global_state_dict = {key: torch.zeros_like(value).to(device) for key, value in global_model.state_dict().items()}

    # Perform simple averaging of the parameters
    for client_state_dict in client_parameters_list:

        # Aggregate parameters for each layer
        for key in global_state_dict:
            global_state_dict[key] += client_state_dict[key].to(device)

    # Average the parameters across all clients
    num_clients = len(client_parameters_list)
    for key in global_state_dict:
        global_state_dict[key] /= num_clients

    # Update the global model with the aggregated parameters
    global_model.load_state_dict(global_state_dict)


def aggregate_vaes(vaes, global_vaes):
    labels = global_vaes.keys()
    for l in labels:
        client_parameters_list = [local_vaes[l].state_dict() for client_i, local_vaes in vaes.items()]
        aggregate(client_parameters_list, global_vaes[l])


def aggregate_gnns(gnns, global_gnn):
    client_parameters_list = [local_gnn.state_dict() for client_i, local_gnn in gnns.items()]
    aggregate(client_parameters_list, global_gnn)


class VAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, latent_dim=5):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean of latent distribution
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent distribution
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        # print(x.device, self.fc1.weight.device, self.fc1.bias.device)
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h3 = torch.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))  # Sigmoid for normalized output
        # return torch.softmax(self.fc4(h3), dim=1)  # softmax for normalized output
        return self.fc4(h3)  # output

    def forward(self, x):
        mu, logvar = self.encoder(x)  # Flatten input logits
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
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


@timer
def evaluate(local_gnn, local_data, device, test_type='test', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.

        client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        client_data_ =  (graph_data_, feature_info, client_data_)
    """
    # After training, the model can make predictions for both labeled and unlabeled nodes
    print(f'***Testing gnn model on test_type:{test_type}...')
    # gnn = local_gnn(input_dim=64, hidden_dim=32, output_dim=10)
    # gnn.load_state_dict(client_result['client_gm'])
    gnn = local_gnn
    gnn.to(device)

    graph_data = train_info['gnn']['graph_data'].to(device)  # graph data
    gnn.eval()
    with torch.no_grad():
        output = gnn(graph_data)
        _, predicted_labels = torch.max(output, dim=1)

        # Calculate accuracy for the labeled data
        # num_classes = 10
        # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        # print(f'labeled_indices {len(labeled_indices)}')
        true_labels = graph_data.y

        y = true_labels.cpu().numpy()
        y_pred = predicted_labels.cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on labeled data: {accuracy * 100:.2f}%")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        # if 'all' in test_type:
        #     client_result['labeled_cm_all'] = conf_matrix
        # else:
        #     client_result['labeled_cm'] = conf_matrix

        # # Calculate accuracy for all data
        # print(f'Evaluate on all data (labeled + unlabeled)')
        # true_labels = client_data_[1]['labels']
        # y = true_labels.cpu().numpy()
        # y_pred = predicted_labels.cpu().numpy()
        # accuracy = accuracy_score(y, y_pred)
        # print(f"Accuracy on all data: {accuracy * 100:.2f}%")
        # if 'all' in test_type:
        #     client_result['accuracy_all'] = accuracy
        # else:
        #     client_result['accuracy'] = accuracy
        #
        # # Compute the confusion matrix
        # conf_matrix = confusion_matrix(y, y_pred)
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # if 'all' in test_type:
        #     client_result['cm_all'] = conf_matrix
        # else:
        #     client_result['cm'] = conf_matrix

    print(f"Client {client_id + 1} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return

def print_histories(histories):
    num_server_epoches = len(histories)
    num_clients = len(histories[0])
    num_classes = len(histories[0][0]["vaes"])
    print('num_server_epoches:', num_server_epoches, ' num_clients:', num_clients, ' num_classes:', num_classes)
    for c in range(num_clients):
        print(f"\n\nclient {c}")
        for s in range(num_server_epoches):
            client = histories[s][c]
            local_vaes = client['vaes']
            local_gnn = client['gnn']
            print(f'\t*local vaes:', local_vaes.keys(), f' server_epoch: {s}')
            for l in range(num_classes):
                losses_ = []
                if local_vaes[l]['losses'] is None:
                    losses_ = None
                else:
                    losses_ = [f"{v:.2f}" for v in local_vaes[l]['losses']]
                print(f'\t\tlocal vae[{l}]:', losses_)
            # print('\t*local gnn:', [f"{v:.2f}" for v in local_gnn['losses']])


def main():
    num_clients = len(LABELs)
    num_classes = num_clients
    input_dim = 64

    features_info = gen_features(feature_file=f'features_{len(LABELs)}.pkl')

    global_vaes = {l: VAE(input_dim=input_dim, hidden_dim=32, latent_dim=5) for l in LABELs}
    global_gnn = GNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)

    num_server_epoches = 10
    histories = []
    for epoch in range(num_server_epoches):
        # update clients
        vaes = {}
        gnns = {}
        history = {}
        for c in range(num_clients):
            print(f"\n\n***server_epoch:{epoch}, client_{c} ...")
            l = c  # we should have 'num_clients = num_labels'
            train_info = {"vaes":{}, "gnn":{}}
            local_data = features_info[l]
            print(f'client_{c} data:', collections.Counter(local_data['labels'].tolist()))
            local_vaes = {l: VAE(input_dim=input_dim, hidden_dim=32, latent_dim=5) for l in LABELs}  # 10 classes
            print('train_vaes...')
            train_vaes(local_vaes, global_vaes, local_data, train_info)
            vaes[c] = local_vaes

            print('train_gnn...')
            local_gnn = GNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
            train_gnn(local_gnn, global_vaes, global_gnn, local_data, train_info)  # will generate new data
            gnns[c] = local_gnn

            print('evaluate_gnn...')
            # self.evaluate(client_result, graph_data_, device, test_type='train', client_id=client_id)
            evaluate(local_gnn, None, device,
                     test_type='Testing on client data', client_id=c, train_info=train_info)
            # # Note that client_result will be overridden or appended more.
            # evaluate(local_gnn, graph_data, device,
            #               test_type='Testing on all clients\' data (aggregated)', client_id=c)
            history[c] = train_info
        # server aggregation
        aggregate_vaes(vaes, global_vaes)
        aggregate_gnns(gnns, global_gnn)

        histories.append(history)

    history_file = 'histories.pkl'
    with open(history_file, 'wb') as f:
        pickle.dump(histories, f)

    # print_histories(histories)


if __name__ == '__main__':
    main()

    history_file = 'histories.pkl'
    with open(history_file, 'rb') as f:
        histories = pickle.load(f)
    print_histories(histories)
