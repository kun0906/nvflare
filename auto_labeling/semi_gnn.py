"""

$cd nvflare/auto_labeling
$PYTHONPATH=.. python3 auto_labeling/semi_gnn.py

"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networkx.algorithms import similarity
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torchvision import datasets, transforms

from auto_labeling.pretrained import pretrained_CNN
from utils import timer

print(os.path.abspath(os.getcwd()))

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


#
# # Define the Graph Neural Network model
# class GNNModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GNNModel, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = torch.relu(self.conv1(x, edge_index))
#         x = torch.relu(self.conv2(x, edge_index))
#         return x

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


from torch_geometric.nn import GraphSAGE

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphSAGEModel, self).__init__()
        # Assuming the GraphSAGE model is defined here with num_layers
        self.conv1 = GraphSAGE(input_dim, hidden_dim, num_layers)  # Pass num_layers to GraphSAGE
        self.conv2 = GraphSAGE(hidden_dim, output_dim, num_layers)  # If you have more layers
        self.fc = torch.nn.Linear(output_dim, output_dim)  # Assuming 10 output classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4):
        super(GATModel, self).__init__()

        # First GAT layer
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        # Second GAT layer
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.6)

        # Fully connected layer
        self.fc = torch.nn.Linear(output_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GAT layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))

        # Second GAT layer
        x = self.conv2(x, edge_index)

        # Apply log softmax to get class probabilities
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


@timer
# Training loop for GNN
def train_gnn(model, criterion, optimizer, data, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Loss calculation: Only for labeled nodes
        loss = criterion(output[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, train_mask: {sum(data.train_mask)}")


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


def gen_features(feature_file='feature.pkl', label_percent=0.1):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # here we use test set as train set is too big for graph representation
    train_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Use 10% labeled data
    # Calculate the index for selecting 10% of the data
    total_size = len(train_data)
    subset_size = int(total_size * label_percent)  # 10%
    # Generate random indices
    indices = torch.randperm(total_size).tolist()[:subset_size]
    # Create a Subset of the dataset using the selected indices
    labeled_data = torch.utils.data.Subset(train_data, indices)
    # unlabeled_data = torch.utils.data.Subset(train_data, range(num_labeled, len(train_data)))

    # DataLoader for labeled data (used for fine-tuning)
    labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=32, shuffle=True)

    pretrained_cnn = pretrained_CNN(labeled_loader, device=device)

    # Extract features for both labeled and unlabeled data
    train_features = extract_features(train_data, pretrained_cnn)
    print(train_features.shape)

    labels = train_data.targets
    train_info = {'train_features': train_features, 'train_labels': labels, "indices": indices}

    with open(feature_file, 'wb') as f:
        pickle.dump(train_info, f)


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
def gen_graph_data(train_info, graph_data_file='graph_data.pkl'):
    train_features = train_info['train_features']
    indices = train_info['indices']
    train_labels = train_info['train_labels']

    edges, edge_attr = gen_edges(train_features, edge_method='cosine')
    print(f"edges.shape {edges.shape}")
    # Create node features (features from CNN)
    node_features = torch.tensor(train_features, dtype=torch.float)

    # Create train mask (10% labeled, 90% unlabeled)
    train_mask = torch.zeros(len(train_features), dtype=torch.bool)
    train_mask[indices] = 1

    # Create labels (10% labeled, others are -1 or placeholder)
    labels = torch.full((len(train_features),), -1, dtype=torch.long)  # Initialize labels
    labels[indices] = train_labels[indices]

    # Prepare Graph data for PyG (PyTorch Geometric)
    print('Form graph data.')
    data = Data(x=node_features, edge_index=edges, edge_attr=edge_attr, y=labels, train_mask=train_mask)

    print('Save data to pickle file.')
    torch.save(data, graph_data_file)

    return


@timer
def main():
    feature_file = 'feature.pkl'
    if not os.path.exists(feature_file):
        gen_features(feature_file, label_percent=0.1)
    with open(feature_file, 'rb') as f:
        train_info = pickle.load(f)
    # Evaluate with Decision Tree to check if the features extracted by CNN are good.
    # baseline.py

    graph_data_file = 'graph_data.pkl'
    # if not os.path.exists(graph_data_file):
    gen_graph_data(train_info, graph_data_file)

    # Get the size of the file in bytes
    file_size = os.path.getsize(graph_data_file) / 1024 ** 3
    print(f'Loading graph data from {graph_data_file}: {file_size:.2f}GB.')
    with open(graph_data_file, 'rb') as f:
        data = torch.load(f, weights_only=None)
    print(f"Graph_data {data}")
    data.to(device)

    # Initialize model and move to GPU
    gnn = GNNModel(input_dim=64, hidden_dim=32, output_dim=10)
    # gnn = GraphSAGEModel(input_dim=64, hidden_dim=32, output_dim=10)
    # gnn = GATModel(input_dim=64, hidden_dim=32, output_dim=10, heads=2)
    gnn.to(device)
    # Loss and optimizer for GNN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gnn.parameters(), lr=0.01)

    print('Training gnn model...')
    train_gnn(gnn, criterion, optimizer, data, epochs=20)

    # After training, the model can make predictions for both labeled and unlabeled nodes
    print('Testing gnn model...')
    gnn.eval()
    with torch.no_grad():
        output = gnn(data)
        _, predicted_labels = torch.max(output, dim=1)

        # Calculate accuracy for the labeled data
        labeled_indices = data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        print(f'labeled_indices {len(labeled_indices)}')
        true_labels = data.y[labeled_indices]

        y = true_labels.cpu().numpy()
        y_pred = predicted_labels[labeled_indices].cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on labeled data: {accuracy * 100:.2f}%")

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Calculate accuracy for all data
        print(f'Evaluate on all data')
        true_labels = train_info['train_labels']
        y = true_labels.cpu().numpy()
        y_pred = predicted_labels.cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on all data: {accuracy * 100:.2f}%")

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)


if __name__ == '__main__':
    main()
