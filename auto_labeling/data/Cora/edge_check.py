import collections

from sklearn.metrics import accuracy_score, confusion_matrix
from torch_geometric.datasets import Planetoid

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


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


def print_histgram(new_probs, value_type='probs'):
    print(f'***Print histgram of {value_type}, min:{min(new_probs)}, max: {max(new_probs)}***')
    # # Convert the probabilities to numpy for histogram calculation
    # new_probs = new_probs.detach().cpu().numpy()
    # Compute histogram
    hist, bin_edges = torch.histogram(torch.tensor(new_probs), bins=10)
    # Print histogram
    for i in range(len(hist)):
        print(f"\tBin {i}: {value_type} Range ({bin_edges[i]}, {bin_edges[i + 1]}), Frequency: {hist[i]}")


def compute_similarity(X, threshold=0.5, edge_method='cosine', train_info={}):
    if edge_method == 'cosine':
        # Calculate cosine similarity to build graph edges (based on CNN features)
        similarity_matrix = cosine_similarity(X)  # [-1, 1]
        # Set diagonal items to 0
        np.fill_diagonal(similarity_matrix, 0)
        # similarity_matrix = cosine_similarity_torch(train_features)
        # Convert NumPy array to PyTorch tensor
        similarity_matrix = torch.abs(torch.tensor(similarity_matrix, dtype=torch.float32))
        # #  # only keep the upper triangle of the matrix and exclude the diagonal entries
        # similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
        print(f'similarity matrix: {similarity_matrix.shape}')
        # Create graph: Each image is a node, edges based on similarity
        # threshold = torch.quantile(similarity_matrix, 0.9)  # input tensor is too large()
        # Convert the tensor to NumPy array
        print_histgram(similarity_matrix.detach().cpu().numpy().flatten(), value_type='similarity')
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
            # Find indices where similarity exceeds the threshold
            edge_indices = (torch.abs(similarity_matrix) > threshold).nonzero(
                as_tuple=False)  # two dimensional data [source, targets]
            per = 100 - edge_indices.shape[0] / (similarity_matrix.shape[0] ** 2) * 100
        print('threshold', threshold)
        # Find indices where similarity exceeds the threshold
        edge_indices = (torch.abs(similarity_matrix) > threshold).nonzero(
            as_tuple=False)  # two dimensional data [source, targets]
        print(f"total number of edges: {similarity_matrix.shape}, we only keep {100 - per:.2f}% edges "
              f"with edge_indices.shape: {edge_indices.shape}")
        edge_weight = similarity_matrix[edge_indices[:, 0], edge_indices[:, 1]]  # one dimensional data

    return edge_indices.t(), edge_weight


def edge_diff(edge_index, edge_index2):
    """
        diff = edge_index - edge_index2
    Args:
        edge_index:
        edge_index2:

    Returns:

    """
    diff2 = []
    unqiue_edges2 = set(map(tuple, edge_index2.numpy().T))
    for a, b in edge_index.numpy().T:
        e = (a, b)
        if e not in unqiue_edges2:
            diff2.append(e)

    return diff2


def hamming_dist(X, Y):
    # # Calculate pairwise Hamming distances in a vectorized manner
    # return np.sum(X[:, None, :] != Y[None, :, :], axis=2)

    dist = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            dist[i, j] = np.sum(X[i] != Y[j])

    return dist


def jaccard_dist(X, Y):
    from scipy.spatial.distance import jaccard

    dist = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            dist[i, j] = jaccard(X[i], Y[j])

    return dist


def main():
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
    edge_index = data.edge_index
    # # edge_indices = set([(row[0], row[1]) for row in edge_indices])
    # edge_indices = set(map(tuple, data.edge_index.numpy().T))
    # unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in data.edge_index.numpy().T])
    # print(f'unique edges: {len(unqiue_edges)} =? edge_indices/2: {len(edge_indices) / 2}, '
    #       f'edges: {data.edge_index.shape}')

    X_train = X[data.train_mask]
    y_train = Y[data.train_mask]
    X_val = X[data.val_mask]
    y_val = Y[data.val_mask]
    X_test = X[data.test_mask]
    y_test = Y[data.test_mask]

    evaluate_ML(X_train, y_train, X_val, y_val, X_test, y_test, X_test, y_test, verbose=10)
    return

    edge_index2, edge_weight2 = compute_similarity(X, threshold=0.5)

    print('edge_index:', edge_index.shape)
    print('edge_index2: ', edge_index2.shape)
    # diff = edge_diff(edge_index, edge_index2)
    # diff2 = edge_diff(edge_index2, edge_index)
    s1 = set(map(tuple, edge_index.numpy().T))
    s2 = set(map(tuple, edge_index2.numpy().T))
    print(f'edge_index - edge_index2 = ({len(s1 - s2)})')
    print(f'edge_index2 - edge_index = ({len(s2 - s1)})')
    print(f'overlapping edges = ({len(s1.intersection(s2))})')

    # cosine
    for similarity_type, similarity in [('cosine', cosine_similarity(X, X)),
                                        ('hamming', hamming_dist(X, X)),
                                        ('jaccard', jaccard_dist(X, X))]:
        print(similarity_type)
        np.fill_diagonal(similarity, 0)
        print(similarity.shape)
        print_histgram(similarity.flatten(), value_type=similarity_type)

        similarity_ = similarity[edge_index[0, :], edge_index[1, :]]
        print(similarity_.shape)
        print_histgram(similarity_.flatten(), value_type=similarity_type)


if __name__ == '__main__':
    main()
