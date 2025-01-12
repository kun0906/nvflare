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
from sklearn.metrics import accuracy_score, confusion_matrix
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
        for j in range(i + 1, m):
            a, b = test_indices[i], test_indices[j]
            if (a, b) in edge_indices:
                new_original_edge_indices.append([a, b])
                new_edge_indices.append([i, j])
            if (b, a) in edge_indices:
                new_original_edge_indices.append([b, a])
                new_edge_indices.append([j, i])

    unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in new_edge_indices])
    print(f'unique edges: {len(unqiue_edges)}, edge_indices/2: {len(new_edge_indices) / 2}')
    return X_test, y_test, np.array(new_edge_indices), np.array(new_original_edge_indices)


def evaluate_ML(X_train, y_train, X_val, y_val, X_test, y_test,
                X_shared_test, y_shared_test, verbose=10):
    if verbose > 5:
        print('---------------------------------------------------------------')
        print('Evaluate Classical ML on each client...')
    ml_info = {}

    dim = X_train.shape[1]
    num_classes = len(set(y_train))
    if verbose > 5:
        print(f'Number of Features: {dim}, Number of Classes: {num_classes}')
        print(f'\tX_train: {X_train.shape}, y_train: '
              f'{collections.Counter(y_train.tolist())}')
        print(f'\tX_val: {X_val.shape}, y_val: '
              f'{collections.Counter(y_val.tolist())}')
        print(f'\tX_test: {X_test.shape}, y_test: '
              f'{collections.Counter(y_test.tolist())}')

        print(f'\tX_shared_test: {X_shared_test.shape}, y_test: '
              f'{collections.Counter(y_shared_test.tolist())}')

        print(f'Total (without X_shared_val): X_train + X_val + X_test + X_shared_test = '
              f'{X_train.shape[0] + X_val.shape[0] + X_test.shape[0] + X_shared_test.shape[0]}')

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

    # all_data = client_data['all_data']
    # test_mask = all_data['test_mask']
    # X_shared_test = all_data['X'][test_mask]
    # y_shared_test = all_data['y'][test_mask]
    for clf_name, clf in clfs.items():
        if verbose > 5:
            print(f"\nTraining {clf_name}")
        # Train the classifier on the training data
        if clf_name == 'MLP':
            clf.fit(X_train, y_train, X_val, y_val)
        else:
            clf.fit(X_train, y_train)
        if verbose > 5:
            print(f"Testing {clf_name}")
        for test_type, X_, y_ in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test),
                                  ('shared_test', X_shared_test, y_shared_test)
                                  ]:
            if verbose > 5:
                print(f'Testing on {test_type}')
            # Make predictions on the data
            y_pred_ = clf.predict(X_)
            # Calculate accuracy
            accuracy = accuracy_score(y_, y_pred_)
            if verbose > 5:
                print(f"Accuracy of {clf_name}: {accuracy * 100:.2f}%")
            # Compute confusion matrix
            cm = confusion_matrix(y_, y_pred_)
            if verbose > 5:
                print(cm)
            ml_info[clf_name] = {test_type: {'accuracy': accuracy, 'cm': cm}}
    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()

    return ml_info


def evaluate_GNN(X_train0, y_train0, indices_train0,
                 X_val0, y_val0, indices_val0,
                 X_test0, y_test0, indices_test0,
                 X_test, y_test, test_indices,
                 X, y, edge_indices):
    import os
    import pickle
    import torch.nn.functional as F
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    from torch_geometric.nn import GCNConv

    print(os.path.abspath(os.getcwd()))

    # Check if GPU is available and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import numpy as np
    from scipy.sparse import csr_matrix
    from torch_geometric.data import Data

    def gen_data():
        indices = np.arange(len(X))
        edge_indices_set = set(map(tuple, edge_indices.numpy().T))

        X0 = np.concatenate([X_train0, X_val0, X_test0], axis=0)
        y0 = np.concatenate([y_train0, y_val0, y_test0])

        train_mask = torch.tensor([False] * len(X0), dtype=torch.bool)
        train_mask[:len(X_train0)] = True

        val_mask = torch.tensor([False] * len(X0), dtype=torch.bool)
        val_mask[len(X_train0):len(X_train0) + len(X_val0)] = True

        test_mask = torch.tensor([False] * len(X0), dtype=torch.bool)
        test_mask[len(X_train0) + len(X_val0):] = True

        indices0 = np.concatenate([indices_train0, indices_val0, indices_test0])
        edge_indices0 = []
        for i, orig_i in enumerate(indices0):
            for j, orig_j in enumerate(indices0):
                e = (orig_i, orig_j)
                if e in edge_indices_set:
                    edge_indices0.append((i, j))
        edge_weight0 = torch.ones(len(edge_indices0), dtype=torch.float)

        X0 = torch.tensor(X0, dtype=torch.float)
        y0 = torch.tensor(y0, dtype=torch.int64)
        edge_indices0 = torch.tensor(edge_indices0, dtype=torch.long).t()
        edge_weight0 = edge_weight0.to(device)

        graph_data = Data(x=X0, edge_index=edge_indices0, edge_weight=edge_weight0,
                          y=y0, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        # graph_data.num_classes = len(set(y.tolist()))

        return graph_data, np.arange(len(indices0)), indices0

    graph_data, graph_indices0, orig_indices0 = gen_data()

    class GNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Adding a third layer
            self.conv4 = GCNConv(hidden_dim, output_dim)  # Output layer

        def forward(self, data):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

            no_edge_weight = False
            if no_edge_weight:
                # no edge_weight is passed to the GCNConv layers
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))  # Additional layer
                x = self.conv4(x, edge_index)  # Final output
            else:
                # Passing edge_weight to the GCNConv layers
                x = F.relu(self.conv1(x, edge_index, edge_weight))
                # x = F.relu(self.conv2(x, edge_index, edge_weight))    # add more layers will lead to worse performance
                # x = F.relu(self.conv3(x, edge_index, edge_weight))  # Additional layer
                x = self.conv4(x, edge_index, edge_weight)  # Final output

            # return F.log_softmax(x, dim=1)
            return x

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
            plt.title(f'Best_Acc: {best_acc:.2f} at Epoch: {epoch}')
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
            plt.title(f'Best_Acc: {best_acc:.2f} at Epoch: {epoch}')
            plt.legend()
            # plt.grid()
            plt.show()

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
            edge_weight = (2 - torch.tensor(dists,
                                            dtype=torch.float32)) / 2  # dists is [0, 2], after this, values is [0,1]
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
            edge_indices, edge_attr = gen_edges(X_train, edge_indices, edge_method='cosine',
                                                train_info={'threshold': None})
            print(f"edges.shape {edge_indices.shape}")
            # data.edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges: tensor([], size=(2, 0), dtype=torch.int64)
            data.edge_index = edge_indices  # update with new edge indices
            data.edge_attr = edge_attr  # use weights

            # graph_data = Data(x=node_features, edge_index=edge_indices, edge_weight=edge_weight,
            #                   y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        _, input_dim = data.x.shape
        classes = data.y.cpu().numpy().tolist()
        num_classes = len(set(classes))
        print(f'input_dim: {input_dim}, X.shape: {data.x.shape}, y: {collections.Counter(classes)}')
        print(f'train: {data.train_mask.sum()}, val: {data.val_mask.sum()}, test: {data.test_mask.sum()}')
        data.to(device)

        # Initialize model and move to GPU
        gnn = GNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
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
            y = true_labels.cpu().numpy()
            y_pred = predicted_labels[data.test_mask].cpu().numpy()
            accuracy = accuracy_score(y, y_pred)
            print(f"Accuracy on test data: {accuracy * 100:.2f}%")

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y, y_pred)
            print("Confusion Matrix:")
            print(conf_matrix)

        # # Calculate accuracy for Shared test data
        # print(f'Evaluate on Shared test data')
        # evaluate_shared_test(gnn, client_data, data)

        return gnn

    gnn = GraphNN(graph_data)

    # Calculate accuracy for Shared test data
    print(f'Evaluate on Shared test data')
    # new_graph_indices = graph_indices0 + (len(graph_indices0) + np.arange(len(test_indices)))
    new_orig_indices = np.concatenate([orig_indices0, test_indices])
    edge_indices_set = set(map(tuple, edge_indices.numpy().T))
    new_indices = []
    for i, orig_i in enumerate(new_orig_indices):
        for j, orig_j in enumerate(new_orig_indices):
            e = (orig_i, orig_j)
            if e in edge_indices_set:
                new_indices.append((i, j))

    new_weight = torch.ones(len(new_indices), dtype=torch.float)

    new_X = torch.cat([graph_data.x, torch.tensor(X_test).to(device)], dim=0)
    new_y = torch.cat([graph_data.y, torch.tensor(y_test, dtype=torch.long).to(device)], dim=0)
    new_indices = torch.tensor(new_indices, dtype=torch.long).t().to(device)
    new_weight = new_weight.to(device)

    new_train_mask = torch.cat(
        [graph_data.train_mask, torch.tensor([False] * len(X_test), dtype=torch.bool).to(device)])
    new_val_mask = torch.cat([graph_data.val_mask, torch.tensor([False] * len(X_test), dtype=torch.bool).to(device)])
    new_test_mask = torch.cat([graph_data.test_mask, torch.tensor([True] * len(X_test), dtype=torch.bool).to(device)])

    new_graph_data = Data(x=new_X, edge_index=new_indices, edge_weight=new_weight,
                          y=new_y, train_mask=new_train_mask, val_mask=new_val_mask, test_mask=new_test_mask)
    # graph_data.num_classes = len(set(y.tolist()))

    # After training, the model can make predictions for both labeled and unlabeled nodes
    print('Testing gnn model on Shared test data...')
    data = new_graph_data
    print(f'train: {data.train_mask.sum()}, val: {data.val_mask.sum()}, test: {data.test_mask.sum()}')
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
        y = true_labels.cpu().numpy()
        y_pred = predicted_labels[data.test_mask].cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on test data: {accuracy * 100:.2f}%")

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)


def plot_data(X, y, train_info={}):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # # Reduce dimensions to 2D using PCA
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X)

    from sklearn.manifold import TSNE
    for perplexity in [2, 5, 10, 20, 30, 50, 100, 200]:
        if len(X) - 1 < perplexity: continue
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X)

        # Plot training data
        for label in np.unique(y):
            plt.scatter(
                X_2d[y == label, 0],
                X_2d[y == label, 1],
                label=f'Class {label}',
                marker='o',
                alpha=0.7
            )

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(fontsize='small')
        plt.title(f'Perplexity: {perplexity}')
        # plt.grid(True)
        # fig_file = f'{in_dir}/plots/client_{client_id}/epoch_{server_epoch}.png'
        # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        # plt.savefig(fig_file, dpi=100)
        plt.show()
        plt.clf()


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
    print(f'unique edges: {len(unqiue_edges)} =? edge_indices/2: {len(edge_indices) / 2}, '
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

    # plot_data(X_train, y_train)
    # plot_data(X_val, y_val)
    # plot_data(X_test, y_test)
    # return

    # # evaluate classical machine learning models
    # evaluate_ML(X_train, y_train, X_val, y_val, X_test, y_test)

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
    for label_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f'\n***{label_rate * 100}% data are labeled***')
        n_nodes = 0
        n_edges = 0
        X_train0, y_train_0, X_val0, y_val0, X_test0, y_test0 = None, None, None, None, None, None
        indices_train0, indices_val0, indices_test0 = None, None, None
        for i, classes_ in enumerate(classes_list):
            # We only split the train set for clients. Given this split, we will miss edges like 0->1, 2->3, ...
            # client 0: only has classes 0 and 3, and edges 0->3 or 3->0
            # client 1: ...
            print(f'\nClient_{i}: class {classes_}')
            # for each client, we split the client data into train, val and test because we assume each client has
            # small labeled data + large unlabeled data
            X_, y_, train_indices_, edge_indices_, original_edge_indices_ = extract_xy_edges(X, Y,
                                                                                             train_indices,
                                                                                             edge_indices,
                                                                                             classes_)
            print(f'X.shape: {X_.shape}, y: {collections.Counter(y_)}, '
                  f'n_edges: {edge_indices_.shape[1]}, '
                  f'train_indices_: {len(train_indices_)}, where min_index: {min(train_indices_)}, and '
                  f'max_index: {max(train_indices_)} ')
            n_nodes += len(train_indices_)
            n_edges += len(edge_indices_)
            client_data = split_train_val_test(X_, y_, train_indices_, edge_indices_, original_edge_indices_,
                                               test_size=1 - label_rate, val_size=0.1)

            if X_train0 is None:
                X_train0 = client_data['X'][client_data['train_mask']][:]
                y_train0 = client_data['y'][client_data['train_mask']][:]
                indices_train0 = client_data['indices'][:][client_data['train_mask']][:]

                X_val0 = client_data['X'][client_data['val_mask']][:]
                y_val0 = client_data['y'][client_data['val_mask']][:]
                indices_val0 = client_data['indices'][:][client_data['val_mask']][:]

                X_test0 = client_data['X'][client_data['test_mask']][:]
                y_test0 = client_data['y'][client_data['test_mask']][:]
                indices_test0 = client_data['indices'][:][client_data['test_mask']][:]
            else:
                X_train0 = np.concatenate((X_train0, client_data['X'][client_data['train_mask']]), axis=0)
                y_train0 = np.concatenate((y_train0, client_data['y'][client_data['train_mask']]))
                indices_train0 = np.concatenate(
                    (indices_train0, client_data['indices'][:][client_data['train_mask']][:]))

                X_val0 = np.concatenate((X_val0, client_data['X'][client_data['val_mask']]), axis=0)
                y_val0 = np.concatenate((y_val0, client_data['y'][client_data['val_mask']]))
                indices_val0 = np.concatenate(
                    (indices_val0, client_data['indices'][:][client_data['val_mask']][:]))

                X_test0 = np.concatenate((X_test0, client_data['X'][client_data['test_mask']]), axis=0)
                y_test0 = np.concatenate((y_test0, client_data['y'][client_data['test_mask']]))
                indices_test0 = np.concatenate(
                    (indices_test0, client_data['indices'][:][client_data['test_mask']][:]))

            # All clients have shared test set (global test set) to evaluate client model's performance
            # We need all X and Y, and test_mask, which can used to find the edges between train and test set in the future.
            # Otherwise, we will miss the edges between train and test set.
            client_data['all_data'] = {'X': X, 'y': Y, 'indices': indices,
                                       'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
                                       'edge_indices_train': edge_indices_train, 'edge_indices_val': edge_indices_val,
                                       'edge_indices_test': edge_indices_test,
                                       'edge_indices': data.edge_index.numpy()}

            client_data_file = f'{in_dir}/{label_rate}/{i}.pkl'
            os.makedirs(os.path.dirname(client_data_file), exist_ok=True)
            with open(client_data_file, 'wb') as f:
                pickle.dump(client_data, f)

            # evaluate classical machine learning models
        evaluate_ML(X_train0, y_train0, X_val0, y_val0, X_test0, y_test0, X_test, y_test)
        evaluate_GNN(X_train0, y_train0, indices_train0,
                     X_val0, y_val0, indices_val0,
                     X_test0, y_test0, indices_test0,
                     X_test, y_test, test_indices,
                     data.x.numpy(), data.y.numpy(), data.edge_index)
    print(f"*** n_nodes: {n_nodes}, n_edges (undirected): {n_edges}, total undirected edges: {len(edge_indices)}")


def check_client_data():
    for label_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f'\n***{label_rate * 100}% data are labeled***')
        total = 0
        for c in range(num_clients):
            client_data_file = f'{in_dir}/{label_rate}/{c}.pkl'
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
