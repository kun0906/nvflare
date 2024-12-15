"""
    If you don’t know the number of Byzantine clients (which could be 0 or more),
below are a few strategies you can think about：

    1. Standard Deviation-Based Filtering
        scores_mean = scores.mean()
        scores_std = scores.std()
        threshold = scores_mean + 2 * scores_std  # Choose k = 2
        outliers = (scores > threshold).nonzero(as_tuple=True)[0]  # Indices of outliers


    2. Median Absolute Deviation (MAD)
        The MAD is more robust to skewed distributions and heavy-tailed data.
        scores_median = scores.median()
        mad = torch.median(torch.abs(scores - scores_median))
        threshold = scores_median + 3 * mad  # k = 3
        outliers = (scores > threshold).nonzero(as_tuple=True)[0]

    3. Elbow Point Detection

    Once you identified outliers using one of the methods above, then exclude them from aggregation.
    Aggregate the remaining updates using trimmed mean or Krum:

    When Byzantine Count f is Zero
    If there are no Byzantine clients, these methods will still work:
    Trimmed Mean: Simply reduces to mean aggregation if no updates are excluded.
    Krum: Picks the most consistent update, which should match all updates if there are no outliers.



    Better Method for Krum: Approximate Nearest Neighbors (ANN)
    Approximate Nearest Neighbor (ANN) approach  are specifically designed to balance the trade-off between accuracy
    and efficiency, and in many practical cases, they can give results close to the exact nearest neighbors but with
    far better time complexity.

    1. Locality-Sensitive Hashing (LSH):
    Time Complexity: O(nlogn) for searching.
    How it works: LSH is an approximate method that uses hash functions to map similar points to the same bucket
    with high probability. This dramatically reduces the number of comparisons needed.
    When to use: If the data has high dimensions (e.g., embedding spaces, deep learning features) and you don't need
    the exact nearest neighbors, LSH can be very efficient.
    2. Annoy (Approximate Nearest Neighbors Oh Yeah):
    Time Complexity: O(nlogn), with faster search times due to tree-based approaches that are designed to
    trade off accuracy for speed.
    How it works: Annoy builds a forest of trees and allows for efficient nearest neighbor searches. The trade-off is
    that it provides approximate nearest neighbors, but with a much lower time complexity than exact methods.
    When to use: If you want to speed up the nearest neighbor search and are okay with approximations (which still
    often perform well in practice).
    3. Hierarchical Navigable Small World Graphs (HNSW):
    Time Complexity: O(nlogn) for each query.
    How it works: HNSW is a graph-based method that builds a multi-layer graph where each node connects to others
    within a small "window," allowing for fast approximate nearest neighbor search. This method is often used for
    high-dimensional data.
    When to use: HNSW is one of the most efficient and accurate ANN methods, especially when dealing with large-scale
    datasets and high-dimensional data.

"""

import torch

import torch
import numpy as np
from kneed import KneeLocator  # For finding the elbow point dynamically


def compute_pairwise_distances(updates):
    """
    Compute pairwise Euclidean distances between all client updates.
    """
    n_clients = updates.size(0)
    pairwise_distances = torch.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            pairwise_distances[i, j] = torch.norm(updates[i] - updates[j])
    return pairwise_distances


def compute_aggregation_scores(pairwise_distances):
    """
    Compute aggregation scores for each client as the sum of closest distances.
    """
    n_clients = pairwise_distances.size(0)
    scores = torch.zeros(n_clients)
    for i in range(n_clients):
        # Sort distances for each client and sum the closest (n-1) distances
        # scores[i] = pairwise_distances[i].sort().values[:-1].sum()  # Exclude self-distance ?
        scores[i] = pairwise_distances[i].sort().values[1:].sum()  # Exclude self-distance
    return scores


def find_elbow_point(scores):
    """
    Detect the elbow point in sorted scores to identify outliers dynamically.
    """
    sorted_scores, indices = scores.sort()
    x = np.arange(len(sorted_scores))  # Indices (1, 2, ..., n_clients)
    y = sorted_scores.numpy()  # Sorted scores as numpy array

    # Use KneeLocator to find the elbow point
    knee_locator = KneeLocator(x, y, curve='convex', direction='increasing')
    elbow_index = knee_locator.knee

    # Identify outliers (indices beyond the elbow point)
    outliers = indices[elbow_index + 1:]  # All indices from the elbow onward (excluding elbow itself)
    print(indices, elbow_index, outliers)
    return outliers


def robust_aggregation(updates, outliers):
    """
    Perform robust aggregation by excluding outliers and computing the mean.
    """
    # Create a mask to exclude outliers
    mask = torch.ones(updates.size(0), dtype=torch.bool)
    mask[outliers] = False  # Mark outliers as False
    filtered_updates = updates[mask]  # Select only non-outliers
    global_model = filtered_updates.mean(dim=0)  # Simple mean of non-outliers
    return global_model


def find_elbow_point_plain(scores):
    sorted_scores, indices = scores.sort()
    x = np.arange(len(sorted_scores))  # Indices (1, 2, ..., n_clients)
    y = sorted_scores.numpy()  # Sorted scores as numpy array
    print(f'y: {y}')

    # Fit a line to the first and last points
    m = (y[-1] - y[0]) / (x[-1] - x[0])  # slope: m = (y1-y0)/(x1-x0)
    b = y[0] - m * x[0]  # intercept: y = mx + b, -> b = y-mx
    line = m * x + b

    # Compute distances to the line
    distances_to_line = np.abs(y - line) / np.sqrt(m ** 2 + 1)

    # Find the elbow point
    elbow_index = np.argmax(distances_to_line).item()

    print(indices, elbow_index, indices[elbow_index + 1:])

    import matplotlib.pyplot as plt

    plt.plot(x, y, label="Scores", marker='*')
    plt.plot(x, line, label="Fitted Line", linestyle="--", marker="o")
    plt.scatter([elbow_index], [y[elbow_index].item()], color="red", label="Elbow Point")
    plt.legend()
    # plt.xlabel("Client Index")
    plt.ylabel("Scores")
    plt.title("Elbow Point Detection")
    plt.show()

    return elbow_index


def krum(filtered_updates):
    # Compute pairwise distances
    pairwise_distances = torch.cdist(filtered_updates, filtered_updates)

    # Aggregate distances
    n_neighbors = filtered_updates.size(0) - 2  # Use n-2 neighbors
    krum_scores = torch.zeros(filtered_updates.size(0))
    for i in range(filtered_updates.size(0)):
        krum_scores[i] = pairwise_distances[i].sort().values[:n_neighbors].sum()

    # Select the update with the lowest Krum score
    best_candidate_index = torch.argmin(krum_scores)
    global_model = filtered_updates[best_candidate_index]

    return global_model


def fast_krum(updates, f):
    """

    Args:
        updates:
        f: the number of byzantine machines to consider.

    Returns:

    """
    # pip install faiss-cpu  # For CPU-only version, or faiss-gpu for GPU version
    # Instead of computing exact pairwise distances, you can use an ANN algorithm like FAISS
    # (Facebook AI Similarity Search) or HNSW (Hierarchical Navigable Small World) to quickly find the nearest
    # neighbors without computing the full distance matrix.
    import torch
    import faiss  # FAISS for efficient nearest neighbors search
    # Here’s an optimized version of your Krum algorithm with Approximate Nearest Neighbors using FAISS for
    # efficient distance calculations, parallelization for aggregation, and GPU usage
    def compute_pairwise_distances_with_faiss(updates):
        """
        Use FAISS to compute the nearest neighbors efficiently.
        """
        # Convert updates to numpy for FAISS
        updates_np = updates.cpu().numpy().astype(np.float32)

        # Build FAISS index
        index = faiss.IndexFlatL2(updates_np.shape[1])  # Use L2 distance
        index.add(updates_np)

        # Perform nearest neighbor search for each update
        distances, _ = index.search(updates_np, updates.size(0) - 1)  # Exclude self-distance
        return torch.tensor(distances).to(updates.device)

    def compute_krum_scores(faiss_distances, n_neighbors):
        """
        Compute Krum scores based on nearest neighbors.
        """
        krum_scores = faiss_distances[:, :n_neighbors].sum(dim=1)  # Sum up the closest n neighbors
        return krum_scores

    def krum_aggregation(filtered_updates, f):
        """
        Perform Krum aggregation using FAISS for fast nearest neighbor computation.
        """
        # Step 1: Compute pairwise distances using FAISS
        faiss_distances = compute_pairwise_distances_with_faiss(filtered_updates)

        # Step 2: Compute Krum scores
        n_neighbors = filtered_updates.size(0) - f  # Use n-f neighbors
        krum_scores = compute_krum_scores(faiss_distances, n_neighbors)

        # Step 3: Select the best candidate (client with the lowest Krum score)
        best_candidate_index = torch.argmin(krum_scores)
        global_model = filtered_updates[best_candidate_index]

        return global_model

    # Example usage
    torch.manual_seed(42)
    n_clients = 10
    update_size = 5
    filtered_updates = torch.rand((n_clients, update_size)).cuda()  # Assuming GPU

    # Perform Krum aggregation
    global_model = krum_aggregation(filtered_updates, f)

    print(f"Global model (aggregated) from Krum:\n{global_model}")
    return global_model


# pip install hnswlib
import hnswlib
import torch


# Function to compute Krum with HNSW
def krum_with_hnsw(filtered_updates, k=5):
    # Initialize HNSW index
    dim = filtered_updates.size(1)  # dimensionality of the embeddings
    p = hnswlib.HNSWIndex(space='l2', dim=dim)  # using L2 distance (Euclidean)

    # Initialize the HNSW index
    p.init_index(max_elements=filtered_updates.size(0), ef_construction=200, M=16)

    # Add data to the index
    p.add_items(filtered_updates.numpy())

    # Compute pairwise distances using Krum (nearest neighbors)
    krum_scores = torch.zeros(filtered_updates.size(0))

    for i in range(filtered_updates.size(0)):
        # Query the index for the k nearest neighbors
        labels, distances = p.knn_query(filtered_updates[i].numpy(), k=k)

        # Exclude the self-distance (i.e., the point itself)
        krum_scores[i] = torch.sum(torch.tensor(distances[0]))  # sum of the nearest neighbors' distances

    # Select the update with the lowest Krum score
    best_candidate_index = torch.argmin(krum_scores)
    global_model = filtered_updates[best_candidate_index]

    return global_model


#
# # Example usage:
# filtered_updates = torch.randn(10, 128)  # Example tensor with 10 clients and 128-dimensional embeddings
# global_model = krum_with_hnsw(filtered_updates, k=5)
# print(global_model)

# pip install annoy
from annoy import AnnoyIndex
import torch


# Function to compute Krum with Annoy
def krum_with_annoy(filtered_updates, k=5):
    # Initialize Annoy index
    dim = filtered_updates.size(1)  # dimensionality of the embeddings
    t = AnnoyIndex(dim, 'euclidean')

    # Add data to the Annoy index
    for i in range(filtered_updates.size(0)):
        t.add_item(i, filtered_updates[i].numpy())

    # Build the index
    t.build(10)  # 10 trees (you can tune this)

    # Compute pairwise distances using Krum (nearest neighbors)
    krum_scores = torch.zeros(filtered_updates.size(0))

    for i in range(filtered_updates.size(0)):
        # Query the index for the k nearest neighbors
        neighbors = t.get_nns_by_item(i, k, include_distances=True)
        distances = neighbors[1]  # distances to the nearest neighbors

        # Exclude the self-distance (i.e., the point itself)
        krum_scores[i] = torch.sum(torch.tensor(distances))  # sum of the nearest neighbors' distances

    # Select the update with the lowest Krum score
    best_candidate_index = torch.argmin(krum_scores)
    global_model = filtered_updates[best_candidate_index]

    return global_model


#
# # Example usage:
# filtered_updates = torch.randn(10, 128)  # Example tensor with 10 clients and 128-dimensional embeddings
# global_model = krum_with_annoy(filtered_updates, k=5)
# print(global_model)


def main():
    # Example: Simulated updates and aggregation process
    torch.manual_seed(42)  # For reproducibility

    # Simulate client updates (e.g., gradient updates or model parameters)
    n_clients = 10
    update_size = 5
    updates = torch.rand((n_clients, update_size))  # Random updates for demonstration

    # Simulate some Byzantine updates (outliers)
    updates[7] += 10  # Add large noise to client 7
    updates[9] -= 10  # Add large noise to client 9

    # Step 1: Compute pairwise distances
    pairwise_distances = compute_pairwise_distances(updates)

    # Step 2: Compute aggregation scores
    scores = compute_aggregation_scores(pairwise_distances)

    # Step 3: Detect outliers using elbow point detection
    outliers = find_elbow_point(scores)
    elbow_index = find_elbow_point_plain(scores)

    print(f"Identified Outliers: {outliers.tolist()}")

    # Step 4: Perform robust aggregation
    global_model = robust_aggregation(updates, outliers)

    print(f"Global Model after Robust Aggregation:\n{global_model}")


if __name__ == "__main__":
    main()
