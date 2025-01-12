from torch_geometric.datasets import Planetoid

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


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
