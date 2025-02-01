import numpy as np
import torch


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


def krum(updates, f, return_average=False):
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
    # print(distances)
    scores = []

    for i in range(num_updates):
        # Sort distances for the current update and sum the closest (n-f-1) distances
        sorted_distances = np.sort(distances[i])
        # The first distance is the self-distance, we should exclude it.
        score = np.sum(sorted_distances[1:num_updates - f])
        scores.append(score)

    print(scores)
    if return_average:
        # instead return the smallest value, we return the top weighted average
        # Sort scores
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_updates = torch.stack(updates)[sorted_indices]

        diff_dists = np.diff(sorted_scores)
        # Find the index of the maximum value (after the halfway point)
        k = np.argmax(diff_dists)
        print(f'k: {k}')
        # weight average
        update = 0.0
        cnt = 0.0
        for j in range(k + 1):
            update += sorted_updates[j]
            cnt += 1
        update = update / cnt
    else:
        # Select the update with the smallest score
        selected_index = np.argmin(scores)
        print(selected_index)
        update = updates[selected_index]

    return update


def refined_krum(updates, clients_info, return_average=True):
    """

    Args:
        updates:
        clients_info: # how many samples in a client

    Returns:
        clients_type_pred
    """
    clients_type_pred = np.array(['benign'] * len(updates), dtype='U10')
    num_updates = len(updates)

    distances = pairwise_distances(updates)
    # print(distances)

    scores = []
    for i in range(num_updates):
        # Sort distances for the current update and sum the closest (n-f-2) distances
        sorted_indices = np.argsort(distances[i])
        sorted_distances = distances[i][sorted_indices]
        sorted_info = clients_info[sorted_indices]

        # Calculate the halfway point
        n = len(sorted_indices)
        halfway_index = n // 2

        diff_dists = np.diff(sorted_distances)

        # Find the index of the maximum value after the halfway point
        k = np.argmax(diff_dists[halfway_index:]) + halfway_index
        # print("Index of maximum value after halfway:", k)

        # score = np.mean(sorted_distances[:k])   # sample average

        # weight average
        score = 0.0
        cnt = 0.0
        for j in range(1, k + 1):  # the first point is the itself, which should be 0 and we exclude it.
            if sorted_distances[0] != 0:
                raise ValueError
            score += sorted_distances[j] * sorted_info[j]
            cnt += sorted_info[j]
        score = score / cnt

        scores.append(score)

    print(f'scores: {scores}')

    if return_average:
        # instead return the smallest value, we return the top weighted average
        # Sort scores
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_info = clients_info[sorted_indices]
        sorted_updates = torch.stack(updates)[sorted_indices]   # not vstack() or hstack()
        sorted_clients_type_pred = clients_type_pred[sorted_indices]

        diff_dists = np.diff(sorted_scores)
        # Find the index of the maximum value (after the halfway point)
        k = np.argmax(diff_dists)
        print(f'k: {k}')

        sorted_clients_type_pred[k+1:] = 'attacker'

        # **Map the sorted labels back to original order**
        clients_type_pred[sorted_indices] = sorted_clients_type_pred

        # weight average
        update = 0.0
        cnt = 0.0
        for j in range(k + 1):
            update += sorted_updates[j] * sorted_info[j]
            cnt += sorted_info[j]
        update = update / cnt
    else:
        # Select the update with the smallest score
        selected_index = np.argmin(scores)
        print(selected_index)
        update = updates[selected_index]

        clients_type_pred[selected_index] = 'attacker'

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
