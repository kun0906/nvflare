import numpy as np


def pairwise_distances(updates):
    """
    Compute pairwise Euclidean distances between updates.
    :param updates: List of model updates, each being a 1D numpy array.
    :return: Pairwise distance matrix.
    """
    num_updates = len(updates)
    distances = np.zeros((num_updates, num_updates))
    for i in range(num_updates):
        for j in range(i + 1, num_updates):
            distances[i, j] = np.linalg.norm(updates[i] - updates[j])
            distances[j, i] = distances[i, j]
    return distances


def krum(updates, f):
    """
    Krum aggregation for Byzantine-robust federated learning.
    :param updates: List of model updates, each being a 1D numpy array.
    :param f: Number of Byzantine clients to tolerate.
    :return: Selected update (numpy array).
    """
    num_updates = len(updates)
    if num_updates <= 2 * f:
        raise ValueError("Number of updates must be greater than 2 * f.")

    distances = pairwise_distances(updates)
    scores = []

    for i in range(num_updates):
        # Sort distances for the current update and sum the closest (n-f-2) distances
        sorted_distances = np.sort(distances[i])
        score = np.sum(sorted_distances[:num_updates - f - 1])
        scores.append(score)

    # Select the update with the smallest score
    selected_index = np.argmin(scores)
    return updates[selected_index]



# Example updates from clients
updates = [
    np.random.randn(100),  # Update from client 1
    np.random.randn(100),  # Update from client 2
    np.random.randn(100),  # Update from client 3
    np.random.randn(100) + 10,  # Malicious update
]

# Number of Byzantine clients to tolerate
f = 1

# Perform Krum aggregation
aggregated_update = krum(updates, f)
print("Aggregated Update (Krum):", aggregated_update)

