""" Centralized Benchmark Models On Semi-Supervised datasets

    1. Semi-Supervised Datasets
        Cora
            10% labeled data + 90% unlabeled data

    2. Centralized Models
        1) Decision Tree
            During the training phase, decision tree only can use 10% labeled data.

        2) GNN
            During the training phase, GNN can use 10% labeled data + 90% unlabeled data.


"""
import collections

import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.data import Data


#
#
# def check_raw_planetoid(filename):
#     # https://github.com/kimiyoung/planetoid/tree/master
#     with open(filename, 'rb') as f:
#         # The latin1 encoding is specified to handle Python 2 pickled files in Python 3.
#         data = pickle.load(f, encoding='latin1')
#     data = csr_matrix.todense(data)
#     return data
#
#
# check_raw_planetoid('data/Cora/raw/ind.cora.tx')


def gen_data(client_id):
    in_file = f'data/Cora/data/{client_id}.pkl'

    with open(in_file, 'rb') as f:
        client_data = pickle.load(f)

    # Each client has local ((train+val set) 10% labeled and (test set) 90% unlabeled) data + global (test) data
    X = torch.tensor(client_data['X'])  # local data
    y = torch.tensor(client_data['y'])  # local data
    indices = torch.tensor(client_data['indices'])  # local data indices (obtained from original data indices)
    train_mask = torch.tensor(client_data['train_mask'])
    val_mask = torch.tensor(client_data['val_mask'])
    test_mask = torch.tensor(client_data['test_mask'])
    edge_indices = torch.tensor(client_data['edge_indices'])  # local indices (obtained from local data indices)
    unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in edge_indices.t().tolist()])
    print(f'unique edges: {len(unqiue_edges)} =? edges/2: {edge_indices.shape[1] / 2}')

    edge_weight = torch.ones(len(y), dtype=torch.float)
    gnn_data = Data(x=X, edge_index=edge_indices, edge_weight=edge_weight,
                    y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    gnn_data.num_classes = len(set(y.tolist()))

    # Inspect the dataset
    print(f"Number of Nodes: {gnn_data.num_nodes}")
    print(f"Number of Edges: {gnn_data.num_edges}")
    print(f"Node Feature Size: {gnn_data.x.shape[1]}")
    print(f"Number of Classes: {gnn_data.num_classes}")

    # Get traditional dataset for non-graphical dataset
    X = gnn_data.x.numpy()
    Y = gnn_data.y.numpy()
    train_mask, val_mask, test_mask = gnn_data.train_mask.numpy(), gnn_data.val_mask.numpy(), gnn_data.test_mask.numpy()
    X_train, y_train = X[train_mask], Y[train_mask]
    X_val, y_val = X[val_mask], Y[val_mask]
    X_test, y_test = X[test_mask], Y[test_mask]
    print('\nX_train shape:', X_train.shape, 'Y_train:', collections.Counter(y_train),
          '\nX_val shape:', X_val.shape, 'Y_val:', collections.Counter(y_train),
          '\nX_test shape:', X_test.shape, 'Y_test:', collections.Counter(y_test))
    trad_data = {"train": (X_train, y_train),
                 "val": (X_val, y_val),
                 "test": (X_test, y_test)}

    return client_data, gnn_data, trad_data


def ClassicalML(data):
    # # feature_file = 'feature.pkl'
    # # # semi_ml_pretrain.gen_features(feature_file)
    # # with open(feature_file, 'rb') as f:
    # #     train_info = pickle.load(f)
    # train_features = train_info['train_features']
    # indices = np.array(train_info['indices'])
    # train_labels = train_info['train_labels']
    # # Assuming `train_features` and `train_labels` are your features and labels respectively
    #
    # # # Split the dataset into training and testing sets
    # # X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_features[indices], train_features, train_labels[indices], train_labels
    # # mask = np.array([False] * len(train_features))  # Example boolean mask
    # # mask[indices] = True
    # # X_train, X_test, y_train, y_test = train_features[mask], train_features[~mask], train_labels[mask], train_labels[~mask]

    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    dim = X_train.shape[1]
    num_classes = len(set(y_train))
    print(f'Number of Features: {dim}, Number of Classes: {num_classes}')

    from sklearn.tree import DecisionTreeClassifier
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    from sklearn.ensemble import GradientBoostingClassifier
    gd = GradientBoostingClassifier(random_state=42)

    from sklearn import svm
    svm = svm.SVC(random_state=42)

    # mlp = MLP(dim, 64, num_classes)
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, 'MLP': mlp}
    clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, }

    all_data = client_data['all_data']
    test_mask = all_data['test_mask']
    X_shared_test = all_data['X'][test_mask]
    y_shared_test = all_data['y'][test_mask]
    for clf_name, clf in clfs.items():
        print(f"\nTraining {clf_name}")
        # Train the classifier on the training data
        if clf_name == 'MLP':
            clf.fit(X_train, y_train, X_val, y_val)
        else:
            clf.fit(X_train, y_train)

        print(f"\nTesting {clf_name}")
        for test_type, X_, y_ in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test),
                                  ('shared_test', X_shared_test, y_shared_test)]:
            print(f'Testing on {test_type}')
            # Make predictions on the data
            y_pred_ = clf.predict(X_)
            # Calculate accuracy
            accuracy = accuracy_score(y_, y_pred_)
            print(f"Accuracy of {clf_name}: {accuracy * 100:.2f}%")
            # Compute confusion matrix
            cm = confusion_matrix(y_, y_pred_)
            print(cm)

    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()


import os
import pickle
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv

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

        no_edge_attr = False if edge_attr is None else True
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
            x = F.relu(self.conv3(x, edge_index, edge_attr))  # Additional layer
            # x = F.relu(self.conv3(x, edge_index, edge_attr))  # Additional layer
            # x = F.relu(self.conv3(x, edge_index, edge_attr))  # Additional layer
            x = self.conv4(x, edge_index, edge_attr)  # Final output

        # return F.log_softmax(x, dim=1)
        return x  # as we CrossEntropyLoss()


#
# from torch_geometric.nn import GraphSAGE
#
#
# class GraphSAGEModel(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
#         super(GraphSAGEModel, self).__init__()
#         # Assuming the GraphSAGE model is defined here with num_layers
#         self.conv1 = GraphSAGE(input_dim, hidden_dim, num_layers)  # Pass num_layers to GraphSAGE
#         self.conv2 = GraphSAGE(hidden_dim, output_dim, num_layers)  # If you have more layers
#         self.fc = torch.nn.Linear(output_dim, output_dim)  # Assuming 10 output classes
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)
#
#

# from torch_geometric.nn import GATConv
# from torch_geometric.data import Data
#
#
# class GATModel(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4):
#         super(GATModel, self).__init__()
#
#         # First GAT layer
#         self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
#         # Second GAT layer
#         self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.6)
#
#         # Fully connected layer
#         self.fc = torch.nn.Linear(output_dim, output_dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         # First GAT layer with ReLU activation
#         x = F.relu(self.conv1(x, edge_index))
#
#         # Second GAT layer
#         x = self.conv2(x, edge_index)
#
#         # Apply log softmax to get class probabilities
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)
#

# @timer
# # Training loop for GNN
# def train_gnn(model, criterion, optimizer, data, epochs=10):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#
#         # Forward pass
#         output = model(data)
#
#         # Loss calculation: Only for labeled nodes
#         loss = criterion(output[data.train_mask], data.y[data.train_mask])
#
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, train_mask: {sum(data.train_mask)}")
#


@timer
# Training loop for GNN
def train_gnn(model, criterion, optimizer, data, epochs=10, show=True):
    model.train()

    losses = []
    pre_val_loss = 0
    val_cnt = 0
    val_losses = []
    best = {'accuracy': 0, 'accs': []}
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Loss calculation: Only for labeled nodes
        loss = criterion(output[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, train_mask: {sum(data.train_mask)}")

        # X_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
        val_loss, pre_val_loss, val_cnt, stop_training = early_stopping(model, data, None, epoch, pre_val_loss,
                                                                        val_cnt, criterion, patience=100, best=best)
        val_losses.append(val_loss.item())
        if stop_training:
            model.stop_training = True
            print(f'Early Stopping. Epoch: {epoch}, Loss: {loss:.4f}')
            break

    print('best epoch: ', best['epoch'], ' best accuracy: ', best['accuracy'])
    model.load_state_dict(best['model'])

    if show:
        import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, label='Training Loss', marker='o')
        plt.plot(range(len(val_losses)), val_losses, label='Validating Loss', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        best_acc = best['accuracy']
        epoch = best['epoch']
        plt.title(f'Training and Validation Loss Over Epochs. Best_Acc: {best_acc} at Epoch: {epoch}')
        plt.legend()
        # plt.grid()
        plt.show()

        accs_val = best['accs']
        plt.plot(range(len(accs_val)), accs_val, label='', marker='')
        plt.plot(range(len(accs_val)), accs_val, label='Validating Accuracy', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        best_acc = best['accuracy']
        epoch = best['epoch']
        plt.title(f'Training and Validation Acc Over Epochs. Best_Acc: {best_acc} at Epoch: {epoch}')
        plt.legend()
        # plt.grid()
        plt.show()


# @timer
# # Extract features using the fine-tuned CNN for all the images (labeled + unlabeled)
# def extract_features(dataset, pretrained_cnn):
#     pretrained_cnn.eval()  # Set the model to evaluation mode
#     # pretrained_cnn.eval() ensures that layers like batch normalization and dropout behave appropriately
#     # for inference (i.e., no training-specific behavior).
#     features = []
#     # Create a DataLoader to load data in batches
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
#
#     for imgs, _ in dataloader:
#         imgs = imgs.to(device)  # Move the batch of images to GPU
#         with torch.no_grad():
#             feature = pretrained_cnn(imgs)  # Forward pass through the pretrained CNN
#         features.append(feature.cpu().numpy())  # Convert feature to numpy
#
#     # Flatten the list of features
#     return np.concatenate(features, axis=0)

#
# def gen_features(feature_file='feature.pkl', label_percent=0.1):
#     # Load MNIST dataset
#     transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
#                                     transforms.Normalize((0.5,), (0.5,))])
#     # here we use test set as train set is too big for graph representation
#     train_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     # Use 10% labeled data
#     # Calculate the index for selecting 10% of the data
#     total_size = len(train_data)
#     subset_size = int(total_size * label_percent)  # 10%
#     # Generate random indices
#     indices = torch.randperm(total_size).tolist()[:subset_size]  # only labeled data indices.
#     # Create a Subset of the dataset using the selected indices
#     labeled_data = torch.utils.data.Subset(train_data, indices)
#     # unlabeled_data = torch.utils.data.Subset(train_data, range(num_labeled, len(train_data)))
#
#     # DataLoader for labeled data (used for fine-tuning)
#     labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=32, shuffle=True)
#
#     pretrained_cnn = pretrained_CNN(labeled_loader, device=device)
#
#     # Extract features for both labeled and unlabeled data
#     train_features = extract_features(train_data, pretrained_cnn)
#     print(train_features.shape)
#
#     labels = train_data.targets.numpy()
#     train_info = {'train_features': train_features, 'train_labels': labels, "indices": indices}
#
#     with open(feature_file, 'wb') as f:
#         pickle.dump(train_info, f)


@timer
def gen_edges(train_features, edge_indices=None, edge_method='cosine', train_info={}):
    if train_features.is_cuda:
        train_features = train_features.cpu().numpy()

    threshold = train_info['threshold']
    if edge_method == 'cosine':
        # Calculate cosine similarity to build graph edges (based on CNN features)
        similarity_matrix = cosine_similarity(train_features)  # [-1, 1]
        # similarity_matrix = cosine_similarity_torch(train_features)
        # Convert NumPy array to PyTorch tensor
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
        # #  # only keep the upper triangle of the matrix and exclude the diagonal entries
        # similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
        print(f'similarity matrix: {similarity_matrix.shape}, '
              f'min: {min(similarity_matrix.numpy().flatten())}, '
              f'max: {max(similarity_matrix.numpy().flatten())}')
        # Create graph: Each image is a node, edges based on similarity
        # threshold = torch.quantile(similarity_matrix, 0.9)  # input tensor is too large()
        # Convert the tensor to NumPy array
        if threshold is None:
            import scipy.stats as stats
            similarity_matrix_np = similarity_matrix.cpu().numpy()
            # Calculate approximate quantile using scipy
            thresholds = [(v, float(stats.scoreatpercentile(similarity_matrix_np.flatten(), v))) for v in
                          range(0, 100 + 1, 10)]
            print(thresholds)
            per = 99.0
            threshold = stats.scoreatpercentile(similarity_matrix_np.flatten(), per)  # per in [0, 100]
            train_info['threshold'] = threshold
        else:
            per = 99.0
        print('threshold', threshold)
        # Find indices where similarity exceeds the threshold
        new_edge_indices = (similarity_matrix > threshold).nonzero(
            as_tuple=False)  # two dimensional data [source, targets]
        print(f"total number of edges: {similarity_matrix.shape}, we only keep {100 - per:.2f}% edges "
              f"with edge_indices.shape: {new_edge_indices.shape}")
        edge_weight = similarity_matrix[new_edge_indices[:, 0], new_edge_indices[:, 1]]  # one dimensional data

    elif edge_method == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(train_features)
        # Convert NumPy array to PyTorch tensor
        distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
        # Convert the tensor to NumPy array
        if threshold is None:
            import scipy.stats as stats
            distance_matrix_np = distance_matrix.cpu().numpy()
            thresholds = [stats.scoreatpercentile(distance_matrix_np.flatten(), v) for v in range(0, 100 + 1, 10)]
            print(thresholds)
            # Calculate approximate quantile using scipy
            threshold = stats.scoreatpercentile(distance_matrix_np.flatten(), 0.009)  # per in [0, 100]
            # threshold = torch.quantile(distance_matrix, 0.5)  # input tensor is too large()
            train_info['threshold'] = threshold
        print('threshold', threshold)
        edge_indices = (distance_matrix < threshold).nonzero(as_tuple=False)
        # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))
        max_dist = max(distance_matrix_np)
        edge_weight = (max_dist - distance_matrix[edge_indices[:, 0], edge_indices[:, 1]]) / max_dist

    elif edge_method == 'knn':
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        # When using metric='cosine' in NearestNeighbors, it internally calculates
        # 1 - cosine similarity
        # so the distances returned are always non-negative [0, 2] (similarity:[-1, 1]).
        # Also, by default, the knn from sklearn includes each node as its own neighbor,
        # usually it will be the value in the results.
        knn.fit(train_features)
        distances, indices = knn.kneighbors(train_features)
        # Flatten source and target indices
        source_nodes = []
        target_nodes = []
        num_nodes = len(train_features)
        dists = []
        for node_i_idx in range(num_nodes):
            for j, node_j_idx in enumerate(indices[node_i_idx]):  # Neighbors of node `i`
                if node_i_idx == node_j_idx: continue
                source_nodes.append(node_i_idx)
                target_nodes.append(node_j_idx)
                dists.append(distances[node_i_idx][j])

        # Stack source and target to create edge_index
        new_edge_indices = torch.tensor([source_nodes, target_nodes], dtype=torch.long).t()

        # one dimensional data
        edge_weight = (2 - torch.tensor(dists, dtype=torch.float32)) / 2  # dists is [0, 2], after this, values is [0,1]
        # edge_weight = torch.sparse_coo_tensor(edge_indices, values, size=(num_nodes, num_nodes)).to_dense()
        # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))

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
    new_edge_indices = new_edge_indices.t().contiguous()
    print('new edge indices', new_edge_indices.shape)
    if edge_indices is None:
        edge_indices = new_edge_indices
    else:
        # merge edge indices
        new_edge_indices = new_edge_indices.t().numpy().tolist()
        dt = set([(row[0], row[1]) for row in new_edge_indices])
        edge_weight = edge_weight.numpy().tolist()
        missing_edge_cnt = 0
        for i, j in edge_indices.numpy().tolist():
            if ((i, j) not in dt) or ((j, i) not in dt):
                # print(f'old edge ({i}, {j}) is not in new_edge_indices')
                missing_edge_cnt += 1
                new_edge_indices.append((i, j))
                edge_weight.append(1)  # is it too large?

        print(f'missing edges cnt {missing_edge_cnt}, edge_indices {len(edge_indices)}')
        edge_indices = torch.tensor(new_edge_indices, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    return edge_indices, edge_weight


#
# @timer
# def gen_graph_data(train_info, graph_data_file='graph_data.pkl'):
#     train_features = train_info['train_features']
#     indices = train_info['indices']  # only labeled data indices
#     # all data's indices as ground truth (not used in the training data,
#     # only for test evaluation)
#     train_labels = torch.tensor(train_info['train_labels'])
#
#     edges, edge_attr = gen_edges(train_features, edge_method='cosine')
#     print(f"edges.shape {edges.shape}")
#     # Create node features (features from CNN)
#     node_features = torch.tensor(train_features, dtype=torch.float)
#
#     # Create train mask (10% labeled, 90% unlabeled)
#     train_mask = torch.zeros(len(train_features), dtype=torch.bool)
#     train_mask[indices] = 1  # only mask the labeled data as 1, others as 0
#
#     # Create labels (10% labeled, others are -1 or placeholder)
#     labels = torch.full((len(train_features),), -1, dtype=torch.long)  # Initialize labels
#     labels[indices] = train_labels[indices]
#
#     # Prepare Graph data for PyG (PyTorch Geometric)
#     print('Form graph data.')
#     data = Data(x=node_features, edge_index=edges, edge_attr=edge_attr, y=labels, train_mask=train_mask)
#
#     print('Save data to pickle file.')
#     torch.save(data, graph_data_file)
#
#     return
#


def early_stopping(model, X_val, y_val, epoch, pre_val_loss, val_cnt, criterion, patience=10, best={}):
    # Validation phase
    model.eval()
    val_loss = 0.0
    stop_training = False
    with torch.no_grad():
        if y_val is not None:  # is not graph data
            outputs_ = model(X_val)
            loss_ = criterion(outputs_, y_val)
            accuracy = accuracy_score(y_val, np.argmax(outputs_, axis=1))
        else:
            data = X_val  # here must be graph data
            outputs_ = model(data)
            _, predicted_labels = torch.max(outputs_, dim=1)
            # Loss calculation: Only for labeled nodes
            loss_ = criterion(outputs_[data.val_mask], data.y[data.val_mask])

            accuracy = accuracy_score(data.y[data.val_mask].tolist(), predicted_labels[data.val_mask].tolist())
            # print(f"epoch: {epoch} Accuracy on val data: {accuracy * 100:.2f}%")
        val_loss += loss_

    best['accs'].append(accuracy)

    if best['accuracy'] < accuracy:
        best['model'] = model.state_dict()
        best['epoch'] = epoch
        best['val_loss'] = val_loss
        best['accuracy'] = accuracy

    if epoch == 0:
        pre_val_loss = val_loss
        return val_loss, pre_val_loss, val_cnt, stop_training

    if val_loss <= pre_val_loss:
        pre_val_loss = val_loss
        val_cnt = 0
    else:  # if val_loss > pre_val_loss, it means we should consider early stopping.
        val_cnt += 1
        if val_cnt >= patience:
            # training stops.
            stop_training = True
    return val_loss, pre_val_loss, val_cnt, stop_training


def gen_test_edges(graph_data, X_test, y_test, test_edges, global_lp, generated_size, train_info):
    X_local = graph_data.x
    y_local = graph_data.y
    local_size = len(X_local)
    train_mask = graph_data.train_mask

    # plot_data(X_local.numpy(), y_local.numpy(), train_mask.numpy(), generated_size, X_test.numpy(), y_test.numpy(), train_info)

    debug = False
    if debug:
        features = torch.cat((X_local, X_test), dim=0)
        labels = torch.cat((y_local, y_test), dim=0)
        start_idx = len(X_local)
        # features = X_test
        # labels = y_test
        # start_idx = 0
        # Combine all edges
        # edge_indices = test_edges.t()
        edge_indices = torch.combinations(torch.arange(len(features)), r=2).t()
        edge_weights = torch.ones((edge_indices.shape[1],))
        print(f'total edges between all nodes: {edge_indices.shape}')
        return features, labels, edge_indices, edge_weights, start_idx

    existed_edge_indices = graph_data.edge_index
    existed_weights = graph_data.edge_weight.tolist()
    print(f'edges between existed nodes ({len(X_local)}): {existed_edge_indices.shape}')

    # # If current client has classes (0, 1, 2, 3), then predict edges for new nodes (such as, 4, 5, 6)
    edge_threshold = train_info['edge_threshold']
    new_nodes = X_test
    # new_node_pairs = torch.combinations(torch.arange(len(new_nodes)), r=2).t()
    # z = global_lp(new_nodes, new_node_pairs)
    # new_probs = global_lp.decode(z, new_node_pairs)
    # new_edges = new_node_pairs.t()[new_probs.flatten() > threshold].t()
    # # adjust new_edges indices
    # new_edges = local_size + new_edges
    # # new_weights = [1] * len(new_edges) # not correct
    # new_weights = [1] * new_edges.shape[1]
    new_edges = test_edges.t()
    # adjust cross_edges indices for new nodes
    new_edges = local_size + new_edges  # new_edges.shape is 2xN
    new_weights = [1] * new_edges.shape[1]
    print(f'new edges between new nodes ({len(new_nodes)}): {new_edges.shape}')

    # Predict edges between new and existing nodes
    existed_nodes = X_local
    cross_pairs = torch.cartesian_prod(torch.arange(0, local_size),
                                       torch.arange(local_size, local_size + len(new_nodes))).t()
    # z = global_lp(existed_nodes, existed_new_pairs)
    features = torch.cat((existed_nodes, new_nodes), dim=0)
    labels = torch.cat((y_local, y_test), dim=0)
    # z = global_lp(features, cross_pairs)  # here, we use all train_features.
    # Assuming `model` is your trained GNN model
    global_lp.eval()  # Set model to evaluation mode
    # Compute embeddings for existing and test nodes
    z_existed = global_lp(existed_nodes, existed_edge_indices)  # Embeddings for existing nodes
    z_test = global_lp(new_nodes, edge_index=new_edges - local_size)  # Embeddings for test nodes (no edges)
    z = torch.cat([z_existed, z_test], dim=0)
    cross_probs = F.sigmoid(global_lp.decode(z, cross_pairs))
    print_histgram(cross_probs.detach().cpu().numpy())
    cross_edges = cross_pairs.t()[cross_probs.flatten() > edge_threshold].t()
    # adjust cross_edges indices for new nodes
    # cross_edges[1, :] = local_size + cross_edges[1, :]  # new_edges.shape is 2xN
    # cross_edges = torch.zeros((2, 0), dtype=torch.long)
    cross_weights = [1] * cross_edges.shape[1]
    print(f'new edges between existed nodes ({len(existed_nodes)}) and new nodes ({len(new_nodes)}): '
          f'{cross_edges.shape}')

    # Combine all edges
    edge_indices = torch.cat([existed_edge_indices, new_edges, cross_edges], dim=1)
    edge_weights = existed_weights + new_weights + cross_weights
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    print(f'total edges between all nodes: {edge_indices.shape}')

    start_idx = len(y_local)
    return features, labels, edge_indices, edge_weights, start_idx


def extract_edges(orig_indices, edges_set):
    edges = []
    for i, orig_i in enumerate(orig_indices):
        for j, orig_j in enumerate(orig_indices):
            e = (orig_i, orig_j)
            if e in edges_set:
                edges.append([i, j])

    return np.asarray(edges)


def check_edges(edges, edges2):
    print(f'len(edges): {len(edges)}, len(edges2): {len(edges2)}')
    edges_set = set([(a, b) for a, b in edges])
    edges2_set = set([(a, b) for a, b in edges2])
    print(f'len(edges_set): {len(edges_set)}, len(edges2_set): {len(edges2_set)}')
    print(f'edges_set-edges2_set: {edges_set - edges2_set}, {edges2_set - edges_set}')
    print(f'edges_set==edges2_set: {edges_set == edges2_set}')


def evaluate_shared_test(local_gnn, local_data, graph_data):
    X_local = graph_data.x
    y_local = graph_data.y
    edges_local = graph_data.edge_index
    local_size = len(graph_data.y)
    original_local_indices = local_data['original_indices']

    test_type = 'Shared test set'
    train_info = {'edge_threshold': 0.5}
    # After training, the model can make predictions for both labeled and unlabeled nodes
    print(f'***Testing gnn model on test_type:{test_type}...')
    all_data = local_data['all_data']
    # all_indices = all_data['indices']
    # edge_indices = torch.tensor(['edge_indices']).to(device)  # all edge_indices
    X, Y = torch.tensor(all_data['X']).to(device), torch.tensor(all_data['y']).to(device)

    # Shared test data
    shared_test_mask = torch.tensor(all_data['test_mask']).to(device)
    X_test = X[shared_test_mask].to(device)
    y_test = Y[shared_test_mask].to(device)
    original_test_indices = all_data['indices'][shared_test_mask]

    # X_test_indices = torch.tensor(['edge_indices_test'], dtype=torch.int).to(device)
    print(f'X_test: {X_test.size()}, {collections.Counter(y_test.tolist())}')
    edge_indices_test = torch.tensor(all_data['edge_indices_test']).t().to(device)
    # Adjust test index
    edge_indices_test = local_size + edge_indices_test

    features = torch.cat((X_local, X_test), dim=0)
    labels = torch.cat((y_local, y_test), dim=0)

    # how to compute cross edges between X_local and X_test?
    # cross_edges = torch.zeros((2, 0), dtype=torch.int64).to(device)
    edges_set = set([(i, j) for i, j in all_data['edge_indices'].T])
    cross_edges = []
    for i, orig_i in enumerate(original_local_indices):
        for j, orig_j in enumerate(original_test_indices):
            e = (orig_i, orig_j)
            if e in edges_set:
                cross_edges.append([i, j + local_size])
            e = (orig_j, orig_i)
            if e in edges_set:
                cross_edges.append([j + local_size, i])
    cross_edges = torch.tensor(cross_edges, dtype=torch.int64).t().to(device)
    print(f'edge_local: {edges_local.shape}, edge_test: {edge_indices_test.shape}, cross_edges: {cross_edges.shape}')
    edges = torch.cat((edges_local, edge_indices_test, cross_edges), dim=1)
    edge_weight = torch.ones(edges.shape[1], )

    edges2 = extract_edges(np.concatenate([original_local_indices, original_test_indices], axis=0), edges_set)
    check_edges(edges.t().numpy(), edges2)

    graph_data = Data(x=features, edge_index=edges, edge_weight=edge_weight,
                      y=labels)
    graph_data.to(device)

    # evaluate the data
    gnn = local_gnn
    gnn.to(device)
    gnn.eval()
    with torch.no_grad():
        output = gnn(graph_data)
        _, predicted_labels = torch.max(output, dim=1)

        # # here we use new edges (based on local data and test set), so the performance is different from only use
        # # local data
        # # only on local data
        # print('Evaluate on local data...')
        # predicted_labels_ = predicted_labels[:local_size]
        # # Calculate accuracy for the labeled data
        # # num_classes = 10
        # # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        # # print(f'labeled_indices {len(labeled_indices)}')
        # true_labels = graph_data.y[:local_size]
        #
        # # predicted_labels = predicted_labels[graph_data.train_mask]
        # # true_labels = true_labels[graph_data.train_mask]
        # # predicted_labels = predicted_labels[graph_data.test_mask]
        # # true_labels = true_labels[graph_data.test_mask]
        #
        # y = true_labels.cpu().numpy()
        # y_pred = predicted_labels_.cpu().numpy()
        # accuracy = accuracy_score(y, y_pred)
        # print(f"Accuracy on local data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # # if 'all' in test_type:
        # #     client_result['labeled_accuracy_all'] = accuracy
        # # else:
        # #     client_result['labeled_accuracy'] = accuracy
        #
        # # Compute the confusion matrix
        # conf_matrix = confusion_matrix(y, y_pred)
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # # if 'all' in test_type:
        # #     client_result['labeled_cm_all'] = conf_matrix
        # # else:
        # #     client_result['labeled_cm'] = conf_matrix

        # only on test set
        print('Evaluate on shared test data...')
        predicted_labels_ = predicted_labels[local_size:]
        # Calculate accuracy for the labeled data
        # num_classes = 10
        # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        # print(f'labeled_indices {len(labeled_indices)}')
        true_labels = graph_data.y[local_size:]

        # predicted_labels = predicted_labels[graph_data.train_mask]
        # true_labels = true_labels[graph_data.train_mask]
        # predicted_labels = predicted_labels[graph_data.test_mask]
        # true_labels = true_labels[graph_data.test_mask]

        y = true_labels.cpu().numpy()
        y_pred = predicted_labels_.cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
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

    print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


@timer
def GraphNN(data):
    data.to(device)

    using_default_edge = True
    if using_default_edge:
        # The default edge_index represents the citation links between papers, which is a semantic relationship.
        # These edges are manually curated based on whether one paper cites another in the dataset.
        # (Based on each paper's references)
        pass
    else:
        train_mask = data.train_mask
        X_train = data.x[train_mask]
        using_combined_edges = False
        if using_combined_edges:
            edge_indices = data.edge_index.t()
        else:  # only use new edges generated based on similarity or knn.
            edge_indices = None
        edge_indices, edge_attr = gen_edges(X_train, edge_indices, edge_method='cosine', train_info={'threshold': None})
        print(f"edges.shape {edge_indices.shape}")
        # data.edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges: tensor([], size=(2, 0), dtype=torch.int64)
        data.edge_index = edge_indices  # update with new edge indices
        data.edge_attr = edge_attr  # use weights

        # graph_data = Data(x=node_features, edge_index=edge_indices, edge_weight=edge_weight,
        #                   y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    _, input_dim = data.x.shape
    classes = data.y.numpy().tolist()
    num_classes = len(set(classes))
    print(f'input_dim: {input_dim}, y: {collections.Counter(classes)}')
    data.to(device)

    # Initialize model and move to GPU
    gnn = GNNModel(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
    gnn.to(device)
    # Loss and optimizer for GNN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gnn.parameters(), lr=0.001)

    print('Training gnn model...')
    train_gnn(gnn, criterion, optimizer, data, epochs=1000)

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

        # Calculate accuracy for Test data
        print(f'Evaluate on test data')
        true_labels = data.y[data.test_mask]
        y = true_labels.numpy()
        y_pred = predicted_labels[data.test_mask].cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on test data: {accuracy * 100:.2f}%")

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

    # Calculate accuracy for Shared test data
    print(f'Evaluate on Shared test data')
    evaluate_shared_test(gnn, client_data, data)


if __name__ == '__main__':

    num_clients = 4
    for client_id in range(num_clients):
        print(f'\nclient_{client_id}')
        client_data, graph_data, trad_data = gen_data(client_id)

        # ClassicalML(trad_data)

        GraphNN(graph_data)
