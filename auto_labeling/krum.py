import collections

import numpy as np
import torch

np.set_printoptions(precision=2)

def median(clients_updates, clients_weights, dim=0):
    """
    Compute the weighted median for updates of different shapes using (n,) weights.

    Args:
        updates (list of torch.Tensor): A list of `n` tensors with varying shapes.
        weights (torch.Tensor): A 1D tensor of shape (n,) representing the weights.

    Returns:
        torch.Tensor: The weighted median tensor, matching the shape of the first update.
    """
    n = len(clients_updates)  # Number of updates
    assert clients_weights.shape == (n,), "Weights must be of shape (n,) where n is the number of updates."

    # clients_type_pred = np.array(['benign'] * n, dtype='U20')

    # Flatten all updates to 1D and stack along a new dimension (first dimension)
    flattened_updates = [u.flatten() for u in clients_updates]
    stacked_updates = torch.stack(flattened_updates, dim=dim)  # Shape: (n, total_elements)

    # Broadcast weights to match stacked shape, i.e., replicate the values across columns
    expanded_weights = clients_weights.view(n, 1).expand_as(stacked_updates)

    # Sort updates and apply sorting indices to weights
    sorted_updates, sorted_indices = torch.sort(stacked_updates, dim=dim)
    sorted_weights = torch.gather(expanded_weights, dim, sorted_indices)

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=dim)

    # Find index where cumulative weight reaches 50% of total weight
    total_weight = cumulative_weights[-1]  # Total weight for each element
    median_mask = cumulative_weights >= (total_weight / 2)

    # Find the first index that crosses the 50% threshold
    median_index = median_mask.to(dtype=torch.int).argmax(dim=dim)

    # Gather median values from sorted updates, unsqueeze(dim) add a new dimension
    weighted_median_values = sorted_updates.gather(dim, median_index.unsqueeze(dim)).squeeze(dim)

    # Find the original index of the client whose update is selected as the median
    original_median_indices = sorted_indices.gather(dim, median_index.unsqueeze(dim)).squeeze(dim)

    # Mark the client whose update was chosen as the median
    # clients_type_pred[original_median_indices.numpy()] = 'chosen update'
    print(f'chosen update: {dict(collections.Counter(original_median_indices.tolist()))}, {stacked_updates.numpy()} '
          f'updates[0].shape: {tuple(clients_updates[0].shape)}, clients_weights: {clients_weights.numpy()}')
    return weighted_median_values.view(clients_updates[0].shape), None


def mean(clients_updates, clients_weights):
    # weight average
    update = 0.0
    cnt = 0.0
    for j in range(len(clients_updates)):
        update += clients_updates[j] * clients_weights[j]
        cnt += clients_weights[j]
    update = update / cnt
    return update, None


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


def krum(updates, weights, f, return_average=True):
    """
    Krum aggregation for Byzantine-robust federated learning.
    :param updates: List of model updates, each being a 1D numpy array.
    :param f: Number of Byzantine clients to tolerate.
    :return: Selected update (numpy array).
    """
    clients_type_pred = np.array(['benign'] * len(updates), dtype='U20')
    num_updates = len(updates)
    if num_updates <= 2 * f:
        raise ValueError("Number of updates must be greater than 2 * f.")

    # we don't need the weighted distance here.
    distances = pairwise_distances(updates)
    # print(distances)

    scores = []
    for i in range(num_updates):
        # # Sort distances for the current update and sum the closest (n-f-1) distances
        # sorted_distances = np.sort(distances[i])
        # # The first distance is the self-distance, we should exclude it.
        # score = np.sum(sorted_distances[1:num_updates - f])
        # scores.append(score)

        # Sort distances for the current update and sum the closest (n-f-2) distances
        sorted_indices = np.argsort(distances[i])
        sorted_distances = distances[i][sorted_indices]
        sorted_weights = weights[sorted_indices]

        k = (num_updates - 1) - f
        # weight average
        score = 0.0
        weight = 0.0
        for j in range(1, k + 1):  # the first point is the itself, which should be 0 and we exclude it.
            if sorted_distances[0] != 0:
                raise ValueError
            score += sorted_distances[j] * sorted_weights[j]
            weight += sorted_weights[j]
        score = score / weight

        scores.append(score.item())

    print('Krum scores: ', [f'{v:.2f}' for v in scores])
    if return_average:
        # instead return the smallest value, we return the top weighted average
        # Sort scores
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        sorted_updates = torch.stack(updates)[sorted_indices]
        sorted_clients_type_pred = clients_type_pred[sorted_indices]

        k = (len(updates) - 1) - f
        print(f'k: {k}')

        sorted_clients_type_pred[k + 1:] = 'attacker'
        # **Map the sorted labels back to original order**
        clients_type_pred[sorted_indices] = sorted_clients_type_pred

        # weight average
        update = 0.0
        weight = 0.0
        for j in range(k + 1):
            update += sorted_updates[j] * sorted_weights[j]
            weight += sorted_weights[j]
        update = update / weight
    else:
        # Select the update with the smallest score
        selected_index = np.argmin(scores)
        print(f"selected_index: {selected_index}")
        update = updates[selected_index]

        clients_type_pred[selected_index] = 'chosen update'

    return update, clients_type_pred


def refined_krum(updates, weights, return_average=True):
    """

    Args:
        updates:
        clients_info: # how many samples in a client

    Returns:
        clients_type_pred
    """
    clients_type_pred = np.array(['benign'] * len(updates), dtype='U20')
    num_updates = len(updates)

    distances = pairwise_distances(updates)
    # print(distances)

    scores = []
    for i in range(num_updates):
        # Sort distances for the current update and sum the closest (n-f-2) distances
        sorted_indices = np.argsort(distances[i])
        sorted_distances = distances[i][sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Calculate the halfway point
        n = len(sorted_indices)
        halfway_index = n // 2

        diff_dists = np.diff(sorted_distances)

        # Find the index of the maximum value after the halfway point
        k = np.argmax(diff_dists[halfway_index:]) + halfway_index
        # print("Index of maximum value after halfway:", k)

        # weight average
        score = 0.0
        weight = 0.0
        for j in range(1, k + 1):  # the first point is the itself, which should be 0 and we exclude it.
            if sorted_distances[0] != 0:
                raise ValueError
            score += sorted_distances[j] * sorted_weights[j]
            weight += sorted_weights[j]
        score = score / weight

        scores.append(score.item())

    print('Refined_Krum scores: ', [f'{v:.2f}' for v in scores])

    if return_average:
        # instead return the smallest value, we return the top weighted average
        # Sort scores
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        sorted_updates = torch.stack(updates)[sorted_indices]  # not vstack() or hstack()
        sorted_clients_type_pred = clients_type_pred[sorted_indices]

        diff_dists = np.diff(sorted_scores)
        # Find the index of the maximum value (after the halfway point)
        k = np.argmax(diff_dists)
        print(f'k: {k}')

        sorted_clients_type_pred[k + 1:] = 'attacker'

        # **Map the sorted labels back to original order**
        clients_type_pred[sorted_indices] = sorted_clients_type_pred

        # weight average
        update = 0.0
        weight = 0.0
        for j in range(k + 1):
            update += sorted_updates[j] * sorted_weights[j]
            weight += sorted_weights[j]
        update = update / weight
    else:
        # Select the update with the smallest score
        selected_index = np.argmin(scores)
        print(f"selected_index: {selected_index}")
        update = updates[selected_index]

        clients_type_pred[selected_index] = 'chosen update'

    # print(update)
    # print(clients_type_pred)
    return update, clients_type_pred


def main():
    results = []
    N = 1000
    for i in range(N):
        print(f'\nthe {i}th trial: ')
        # Example updates from clients
        dim = 2
        clients_updates = [
            np.random.randn(dim),  # Update from client 1
            np.random.randn(dim),  # Update from client 2
            np.random.randn(dim),  # Update from client 3
            np.random.randn(dim) + 10,  # Malicious update
        ]
        clients_updates = [torch.tensor(v) for v in clients_updates]
        clients_info = np.array([1, 1, 1, 1])

        # Number of Byzantine clients to tolerate
        f = 1

        # Perform Krum aggregation
        print('Krum...')
        aggregated_update = krum(clients_updates, f, return_average=True)
        print("Aggregated Update (Krum):", aggregated_update)
        print('\nRefined Krum...')
        aggregated_update2 = refined_krum(clients_updates, clients_info, return_average=True)
        print("Aggregated Update (Refined Krum):", aggregated_update2)

        if np.sum(aggregated_update2.numpy() - aggregated_update.numpy()) != 0:
            print("Different updates were aggregated")
            results.append(clients_updates)

    print(f'\naccuracy: {1 - len(results) / N}')


if __name__ == '__main__':
    main()
