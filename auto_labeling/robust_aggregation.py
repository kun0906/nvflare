import collections

import numpy as np
import torch
from sklearn.random_projection import GaussianRandomProjection

np.set_printoptions(precision=2)


#
# def median_array(clients_updates, clients_weights, dim=0):
#     """
#     Compute the weighted median for updates of different shapes using (n,) weights.
#
#     Args:
#         updates (list of torch.Tensor): A list of `n` tensors with varying shapes.
#         weights (torch.Tensor): A 1D tensor of shape (n,) representing the weights.
#
#     Returns:
#         torch.Tensor: The weighted median tensor, matching the shape of the first update.
#     """
#     n = len(clients_updates)  # Number of updates
#     assert clients_weights.shape == (n,), "Weights must be of shape (n,) where n is the number of updates."
#
#     # predict_clients_type = np.array(['honest'] * n, dtype='U20')
#
#     # Flatten all updates to 1D and stack along a new dimension (first dimension)
#     flattened_updates = [u.flatten() for u in clients_updates]
#     stacked_updates = torch.stack(flattened_updates, dim=dim)  # Shape: (n, total_elements)
#
#     # Broadcast weights to match stacked shape, i.e., replicate the values across columns
#     expanded_weights = clients_weights.view(n, 1).expand_as(stacked_updates)
#
#     # Sort updates and apply sorting indices to weights
#     sorted_updates, sorted_indices = torch.sort(stacked_updates, dim=dim)
#     sorted_weights = torch.gather(expanded_weights, dim, sorted_indices)
#
#     # Compute cumulative weights
#     cumulative_weights = torch.cumsum(sorted_weights, dim=dim)
#
#     # Find index where cumulative weight reaches 50% of total weight
#     total_weight = cumulative_weights[-1]  # Total weight for each element
#     median_mask = cumulative_weights >= (total_weight / 2)
#
#     # Find the first index that crosses the 50% threshold
#     median_index = median_mask.to(dtype=torch.int).argmax(dim=dim)
#
#     # Gather median values from sorted updates, unsqueeze(dim) add a new dimension
#     weighted_median_values = sorted_updates.gather(dim, median_index.unsqueeze(dim)).squeeze(dim)
#
#     # Find the original index of the client whose update is selected as the median
#     original_median_indices = sorted_indices.gather(dim, median_index.unsqueeze(dim)).squeeze(dim)
#
#     # Mark the client whose update was chosen as the median
#     # predict_clients_type[original_median_indices.numpy()] = 'chosen update'  # {stacked_updates.numpy()}
#     print(f'chosen update: {dict(collections.Counter(original_median_indices.tolist()))}, '
#           f'updates[0].shape: {tuple(clients_updates[0].shape)}, clients_weights: {clients_weights.numpy()}')
#     return weighted_median_values.view(clients_updates[0].shape), None
#

def trimmed_median(clients_updates, clients_weights, p=0.1, verbose=1):
    return median(clients_updates, clients_weights, trimmed_average=True, p=p, verbose=verbose)

def median(clients_updates, clients_weights, trimmed_average=False, p=0.1, verbose=1):
    """
        Compute the weighted median for flattened tensors.

        Args:
            clients_updates (list of torch.Tensor): A list of `n` flattened tensors (e.g., shape (d,)).
            clients_weights (torch.Tensor): A 1D tensor of shape (n,) representing the weights.
            p: which percentage of data will be removed when compute the weighted average.
        Returns:
            torch.Tensor: The weighted median tensor, matching the shape of the first update.
        """
    n = len(clients_updates)
    assert clients_weights.shape == (n,), "Weights must be of shape (n,) where n is the number of updates."

    # Stack updates into a matrix of shape (n, d)
    stacked_updates = torch.stack(clients_updates)

    # PyTorch does not explicitly force -0.0 to appear before 0.0 (because it treats -0.0 equals to 0)
    # , as some other libraries (like NumPy) might do.
    # PyTorch uses a stable sorting algorithm, which means it preserves the original order when values are equal.
    # Since 0.0 and -0.0 are numerically equal, the one that appears first in the original tensor will stay first.
    # Sort updates along the first dimension (client axis)
    sorted_updates, sorted_indices = torch.sort(stacked_updates, dim=0)

    # Sort weights accordingly
    sorted_weights = clients_weights[sorted_indices]

    # Compute cumulative sum of weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)

    total_weight = cumulative_weights[-1]  # Total weight for each element

    if trimmed_average:
        # Trim the top and bottom 10% of updates, considering weights
        weight_threshold = p * total_weight
        lower_bound = torch.searchsorted(cumulative_weights.T, weight_threshold.reshape(-1, 1), side="left")
        upper_bound = torch.searchsorted(cumulative_weights.T, (total_weight - weight_threshold).reshape(-1, 1),
                                         side="right")
        trimmed_updates = torch.stack(
            [sorted_updates[lb:ub + 1, i] for i, (lb, ub) in enumerate(zip(lower_bound, upper_bound))]).T
        trimmed_weights = torch.stack(
            [sorted_weights[lb:ub + 1, i] for i, (lb, ub) in enumerate(zip(lower_bound, upper_bound))]).T

        # Compute weighted average
        weighted_median_values = torch.sum(trimmed_updates * trimmed_weights, dim=0) / torch.sum(
            trimmed_weights, dim=0)
    else:
        # Find first index where cumulative weight reaches 50% of total weight
        median_index = (cumulative_weights >= (total_weight / 2)).to(dtype=torch.int).argmax(dim=0)

        # Select median values
        weighted_median_values = sorted_updates[median_index, torch.arange(stacked_updates.shape[1])]

        # Find the original client indices whose updates were chosen
        original_median_indices = sorted_indices[median_index, torch.arange(stacked_updates.shape[1])]

        # print(f'Chosen update indices: {dict(collections.Counter(original_median_indices.tolist()))}')
        # Mark the client whose update was chosen as the median
        # predict_clients_type[original_median_indices.numpy()] = 'chosen update'  # {stacked_updates.numpy()}
        if verbose >= 5:
            print(f'chosen update: {dict(collections.Counter(original_median_indices.tolist()))}, '
                  f'updates[0].shape: {tuple(clients_updates[0].shape)}, clients_weights: {clients_weights.numpy()}')

    return weighted_median_values, None


def medoid(clients_updates, clients_weights, trimmed_average=False, p=0.1, verbose=1):
    """
    Computes the medoid of a given set of vectors V.
    :param V: List or numpy array of shape (n, d), where n is the number of vectors, d is the dimension.
    :return: Medoid vector from V
    """

    # Stack updates into a matrix of shape (n, d)
    stacked_updates = torch.stack(clients_updates)
    n = len(stacked_updates)

    # Compute sum of distances for each point to all others
    distance_sums = np.array([np.sum(np.linalg.norm(stacked_updates - stacked_updates[i], axis=1)) for i in range(n)])

    # Select the medoid (the point with the minimum sum of distances)
    medoid_index = np.argmin(distance_sums)
    return stacked_updates[medoid_index]


def geometric_median(updates, weights, max_iters=100, tol=1e-6, verbose=1):
    """
    Compute the geometric median of a set of vectors using Weiszfeld's algorithm.

    Args:
        updates: List of client updates (torch tensors or numpy arrays).
        weights: Weights corresponding to each client update.
        max_iters: Maximum number of iterations for convergence.
        tol: Tolerance for stopping criteria.
        verbose: Verbosity level (0: Silent, 1: Basic info, 5+: Debug info).

    Returns:
        The geometric median update.
    """
    updates = torch.stack(updates) if isinstance(updates[0], torch.Tensor) else np.stack(updates)
    weights = weights / torch.sum(weights)

    # Initial estimate: weighted average
    median = torch.sum(weights.view(-1, 1) * updates, dim=0) if isinstance(updates, torch.Tensor) else np.sum(weights[:, None] * updates, axis=0)

    for iteration in range(max_iters):
        distances = torch.norm(updates - median, dim=1) if isinstance(updates, torch.Tensor) else np.linalg.norm(updates - median, axis=1)
        distances = torch.clamp(distances, min=tol)  # Avoid division by zero

        inv_distances = 1.0 / distances
        weighted_inv_distances = weights * inv_distances

        new_median = torch.sum(weighted_inv_distances.view(-1, 1) * updates, dim=0) / torch.sum(weighted_inv_distances) if isinstance(updates, torch.Tensor) else \
                     np.sum(weighted_inv_distances[:, None] * updates, axis=0) / np.sum(weighted_inv_distances)

        if verbose >= 20:
            print(f"Iteration {iteration}: Distance = {torch.norm(new_median - median).item() if isinstance(updates, torch.Tensor) else np.linalg.norm(new_median - median)}")

        # Convergence check
        if torch.norm(new_median - median).item() < tol if isinstance(updates, torch.Tensor) else np.linalg.norm(new_median - median) < tol:
            break

        median = new_median

    if verbose >= 1:
        print(f"Geometric median found in {iteration + 1} iterations.")

    return median, None


def mean(clients_updates, clients_weights, trimmed_average=True, p=0.1, verbose=1):
    """
    Compute the trimmed weighted mean by removing the top and bottom `trim_ratio` fraction of values.

    Args:
        updates (list of torch.Tensor): A list of `n` tensors of shape (d1, d2, ..., dk).
        weights (torch.Tensor): A 1D tensor of shape (n,) representing the weights.
        trim_ratio (float): The fraction of extreme values to remove from both ends (default: 0.1).

    Returns:
        torch.Tensor: The trimmed weighted mean tensor of shape (d1, d2, ..., dk).
    """
    n = len(clients_updates)
    assert clients_weights.shape == (n,), "Weights must be of shape (n,) where n is the number of updates."
    assert 0 <= p < 0.5, "Trim ratio must be between 0 and 0.5."

    if trimmed_average:
        # Stack updates into a matrix of shape (n, d)
        stacked_updates = torch.stack(clients_updates)
        # Sort by distance and apply weighted trimming
        sorted_updates, sorted_indices = torch.sort(stacked_updates, dim=0)
        sorted_weights = clients_weights[sorted_indices]
        cumulative_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = cumulative_weights[-1]
        lower_bound = torch.searchsorted(cumulative_weights, p * total_weight)
        upper_bound = torch.searchsorted(cumulative_weights, (1 - p) * total_weight)
        valid_indices = sorted_indices[lower_bound:upper_bound + 1]

        # Apply mask to updates and weights
        trimmed_updates = stacked_updates[valid_indices, :]
        trimmed_weights = clients_weights[valid_indices]

        # Compute weighted mean after trimming
        weighted_mean = (torch.sum(trimmed_updates * trimmed_weights.reshape((-1, 1)), dim=0)
                         / torch.sum(trimmed_weights))

    else:
        # # Compute weighted mean
        # weighted_mean = torch.sum(clients_updates * clients_weights.view([-1] + [1] * (clients_updates.dim() - 1)),
        #                           dim=0) / torch.sum(clients_weights, dim=0)

        # weight average
        update = 0.0
        cnt = 0.0
        for j in range(len(clients_updates)):
            update += clients_updates[j] * clients_weights[j]
            cnt += clients_weights[j]
        weighted_mean = update / cnt

    return weighted_mean, None


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


def krum(updates, weights, f, trimmed_average=False, random_projection=False, verbose=1):
    """
        Krum aggregation for Byzantine-robust federated learning.
    :param updates: List of model updates, each being a 1D numpy array.
    :param f: Number of Byzantine clients to tolerate.
    :return: Selected update (numpy array).
    """
    predict_clients_type = np.array([[f'Update {i}'] for i in range(len(updates))], dtype='U20')
    # Normalize weights
    weights = weights / torch.sum(weights)

    num_updates = len(updates)
    if 2 * f + 2 >= num_updates:
        raise ValueError("Number of updates must be greater than 2 * f + 2.")

    if random_projection:
        # Apply random projection
        n_features = len(updates[0])  # Length of the parameter vector
        k = int(np.log(num_updates))  # Reduced dimension using log(n)

        # Create a random projection transformer
        transformer = GaussianRandomProjection(n_components=k)
        projected_updates = transformer.fit_transform(np.array(updates))

        # we don't need the weighted distance here.
        distances = pairwise_distances(projected_updates)
    else:
        # we don't need the weighted distance here.
        distances = pairwise_distances(updates)

    if verbose >= 10:
        print(distances)

    k = num_updates - f - 2  # for Krum, it should be n-f-2. It is fixed in Krum
    scores = []
    for i in range(num_updates):
        # Sort distances for the current update and sum the closest (n-f-2) distances
        # # The first distance is the self-distance, we should exclude it.
        sorted_indices = np.argsort(distances[i])
        sorted_distances = distances[i][sorted_indices]
        sorted_weights = weights[sorted_indices]
        # weight average
        score = 0.0
        weight = 0.0
        if sorted_distances[0] != 0:
            raise ValueError
        for j in range(1, k + 1):  # the first point is the itself, which should be 0 and we exclude it.
            score += sorted_distances[j] * sorted_weights[j]
            weight += sorted_weights[j]
        score = score / weight

        scores.append(score.item())

    if verbose >= 5:
        print('Krum scores: ', [f'{v:.2f}' for v in scores])
    if trimmed_average:
        # instead return the smallest value, we return the top weighted average
        # Sort scores
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        sorted_updates = torch.stack(updates)[sorted_indices]
        sorted_predict_clients_type = predict_clients_type[sorted_indices]

        m = num_updates - f  # for the Krum paper, m = n-f
        if verbose >= 5:
            print(f'm: {m}')
        sorted_predict_clients_type[m:] = 'Byzantine'
        # **Map the sorted labels back to original order**
        predict_clients_type[sorted_indices] = sorted_predict_clients_type

        # weight average
        update = 0.0
        weight = 0.0
        for j in range(m):  # start from 0 to k-1
            update += sorted_updates[j] * sorted_weights[j]
            weight += sorted_weights[j]
        update = update / weight
    else:
        # Select the update with the smallest score
        selected_index = np.argmin(scores)
        if verbose >= 5:
            print(f"selected_index: {selected_index}")
        update = updates[selected_index]

        predict_clients_type[selected_index] = 'Chosen Update'

    return update, predict_clients_type


def krum_with_random_projection(updates, weights, f, trimmed_average=False, verbose=1):
    return krum(updates, weights, f, trimmed_average=trimmed_average, random_projection=True, verbose=verbose)


def refined_krum_with_random_projection(updates, weights, trimmed_average=False, verbose=1):
    return refined_krum(updates, weights, trimmed_average=trimmed_average, random_projection=True, verbose=verbose)


def refined_krum(updates, weights, trimmed_average=False, random_projection=False, verbose=1):
    """

    Args:
        updates:
        clients_info: # how many samples in a client

    Returns:
        predict_clients_type
    """
    predict_clients_type = np.array([[f'Update {i}'] for i in range(len(updates))], dtype='U20')
    # Normalize weights
    weights = weights / torch.sum(weights)

    num_updates = len(updates)

    if random_projection:
        # Apply random projection
        # n_features = len(updates[0])  # Length of the parameter vector
        k = 10 * int(np.log(num_updates))  # Reduced dimension using log(n)

        # Create a random projection transformer
        transformer = GaussianRandomProjection(n_components=k)
        projected_updates = transformer.fit_transform(np.array(updates))

        # we don't need the weighted distance here.
        distances = pairwise_distances(projected_updates)
    else:
        # we don't need the weighted distance here.
        distances = pairwise_distances(updates)

    if verbose >= 10:
        print(distances)

    scores = []
    for i in range(num_updates):
        # Sort distances for the current update and sum the closest (n-f-2) distances
        sorted_indices = np.argsort(distances[i])
        sorted_distances = distances[i][sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Calculate the middle point
        n = len(sorted_indices)
        middle_index = (n - 1) // 2  # lower index for even updates

        # Initialize max difference and index
        max_diff = -1
        k = middle_index  # default value for k
        # Find the first occurrence of the max diff in one pass
        for j in range(middle_index, n - 1):
            t = sorted_distances[j + 1] - sorted_distances[j]
            if t < 0 or sorted_distances[j] < 0 or sorted_distances[j + 1] < 0:
                # if it is true, there must be an error as the distance must be >=0.
                raise ValueError
            if t > max_diff:
                max_diff = t
                k = j  # Store the first occurrence of max diff

        if verbose >= 20:
            # print(f'*** j: {j} ***')
            print(f'sorted_distances: {sorted_distances}')
            # print('diff_dists: ', diff_dists)
            print("Index of maximum distance difference after half of values:", k)

        # weight average
        score = 0.0
        weight = 0.0
        if sorted_distances[0] != 0:
            raise ValueError
        for j in range(1, k + 1):  # the first point is the itself, which should be 0 and we exclude it.
            score += sorted_distances[j] * sorted_weights[j]
            weight += sorted_weights[j]
        score = score / weight

        scores.append(score.item())

    if verbose >= 5:
        print('Refined_Krum scores: ', [f'{v:.2f}' for v in scores])

    if trimmed_average:
        # instead return the smallest value, we return the top weighted average
        # Sort scores
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        sorted_updates = torch.stack(updates)[sorted_indices]  # not vstack() or hstack()
        sorted_predict_clients_type = predict_clients_type[sorted_indices]

        # Find the index of the maximum value (after the halfway point)
        # Calculate the middle point
        n = len(sorted_indices)
        middle_index = (n - 1) // 2

        # Initialize max difference and index
        max_diff = -1
        k = middle_index  # default value for k
        # Find the first occurrence of the max diff in one pass
        for j in range(middle_index, n - 1):
            t = sorted_scores[j + 1] - sorted_scores[j]
            if t < 0 or sorted_scores[j] < 0 or sorted_scores[j + 1] < 0:
                # if it is true, there must be an error as the distance must be >=0.
                raise ValueError
            if t > max_diff:
                max_diff = t
                k = j  # Store the first occurrence of max diff

        if verbose >= 20:
            print(f'trimmed_average: sorted_scores: {sorted_scores}')
            print("Index of maximum scores difference after half of values:", k)

        m = k + 1  # we will average over top m closet updates
        if verbose >= 5:
            print(f'm: {m}')
        sorted_predict_clients_type[m:] = 'Byzantine'
        # **Map the sorted labels back to original order**
        predict_clients_type[sorted_indices] = sorted_predict_clients_type

        # weight average
        update = 0.0
        weight = 0.0
        for j in range(m):  # note here is k+1 because we want to add all values before k+1
            update += sorted_updates[j] * sorted_weights[j]
            weight += sorted_weights[j]
        update = update / weight
    else:
        # Select the update with the smallest score
        selected_index = np.argmin(scores)
        if verbose >= 5:
            print(f"selected_index: {selected_index}")
        update = updates[selected_index]

        predict_clients_type[selected_index] = 'Chosen Update'

    # print(update)
    # print(predict_clients_type)
    return update, predict_clients_type


def trimmed_mean(updates, weights, trim_ratio=0.1, verbose=1):
    """
    Computes the trimmed mean of client updates.

    Args:
        updates: List of client updates (torch tensors).
        weights: Corresponding weights for each client.
        trim_ratio: Fraction of extreme values to trim from both ends.
        verbose: Verbosity level.

    Returns:
        Aggregated update.
    """
    predict_clients_type = np.array([[f'Update {i}'] for i in range(len(updates))], dtype='U20')
    num_updates = len(updates)
    if num_updates == 0:
        raise ValueError("No updates provided.")

    # Normalize weights
    weights = weights / torch.sum(weights)

    # Convert updates to tensor for easier manipulation
    updates = torch.stack(updates)

    # Compute Euclidean norms of updates
    norms = torch.norm(updates, dim=1)

    # Sort updates based on their norms
    sorted_indices = torch.argsort(norms)
    sorted_updates = updates[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute trim bounds
    trim_count = max(1, int(trim_ratio * num_updates))
    lower, upper = trim_count, num_updates - trim_count

    if verbose >= 5:
        print(f"Trimming {trim_count} smallest and {trim_count} largest updates.")

    # Compute weighted trimmed mean
    trimmed_updates = sorted_updates[lower:upper]
    trimmed_weights = sorted_weights[lower:upper]

    if len(trimmed_updates) == 0:
        raise ValueError("Trimming removed all updates. Reduce trim_ratio.")

    update = torch.sum(trimmed_updates * trimmed_weights[:, None], dim=0) / torch.sum(trimmed_weights)
    predict_clients_type[lower:upper] = "Chosen Update"

    return update, predict_clients_type



def main():
    results = []
    N = 1000
    verbose = 20
    for i in range(N):
        print(f'\nthe {i}th trial: ')
        # Example updates from clients
        dim = 2
        clients_updates = [
            np.random.randn(dim),  # Update from client 1
            np.random.randn(dim),  # Update from client 2
            np.random.randn(dim),  # Update from client 3
            np.random.randn(dim),  # Update from client 4
            np.random.randn(dim) + 10,  # Malicious update
        ]
        clients_updates = [torch.tensor(v) for v in clients_updates]
        weights = torch.tensor([1, 1, 1, 1, 1])
        # Number of Byzantine clients to tolerate
        f = 1

        # Perform Krum aggregation
        trimmed_average = True
        print('Krum...')
        aggregated_update = krum(clients_updates, weights, f, trimmed_average, verbose=verbose)
        print("Aggregated Update (Krum):", aggregated_update)
        print('\nRefined Krum...')
        aggregated_update2 = refined_krum(clients_updates, weights, trimmed_average, verbose=verbose)
        print("Aggregated Update (Refined Krum):", aggregated_update2)

        # if np.sum(aggregated_update2.numpy() - aggregated_update.numpy()) != 0:
        #     print("Different updates were aggregated")
        #     results.append(clients_updates)
        break
    print(f'\naccuracy: {1 - len(results) / N}')


if __name__ == '__main__':
    main()
