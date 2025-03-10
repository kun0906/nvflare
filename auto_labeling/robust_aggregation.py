import collections
import time
import numpy as np
import torch
from sklearn.random_projection import GaussianRandomProjection

from change_point_detection import binary_segmentation, find_significant_change_point

np.set_printoptions(precision=4)


def cw_mean(clients_updates, clients_weights, verbose=1):
    """
    Compute weighted empirical coordinate-wise mean.

    Args:
        clients_updates: A 2D tensor of shape (N, d)
        clients_weights: A 1D tensor of shape (N,) representing the weights.

    Returns:
       1D torch: The weighted mean tensor of shape (d, ).
    """
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
    predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
    unique_ws, counts = torch.unique(clients_weights, return_counts=True)
    if verbose >= 5:
        print(f'Unique weights: {unique_ws}, counts: {counts}')

    # Compute weighted mean
    # # weight average
    # update = 0.0
    # cnt = 0.0
    # for j in range(len(clients_updates)):
    #     update += clients_updates[j] * clients_weights[j]
    #     cnt += clients_weights[j]
    # weighted_mean = update / cnt

    # Optimized version
    # clients_weights is a 1D tensor with shape (N, ), after clients_weights.unsqueeze(1), we will get a 2D tensor
    # with shape (N, 1)
    # When we do "clients_updates * clients_weights.unsqueeze(1)", clients_weights.unsqueeze(1) will first be
    # broadcasted/duplicated along the new dimension, then perform the element-wise multiplication (Hadamard product),
    # not matrix multiplication, because of broadcasting.
    weighted_mean = torch.sum(clients_updates * clients_weights.unsqueeze(1), dim=0) / torch.sum(clients_weights)

    return weighted_mean, predict_clients_type


def trimmed_mean(clients_updates, clients_weights, trim_method='cumulative_weights', trim_ratio=0.1, verbose=1):
    """
    Computes the trimmed mean of client updates.
    Two common methods for trimming the data: trimmed by count and trimmed by cumulative weights.

    Args:
        clients_updates: List of client updates (torch tensors).
        clients_weights: Corresponding weights for each client. It must be 1D tensor of shape (N,).
        trim_method: Method used to trim the data. cumulative_weights (default) or count
        trim_ratio: Fraction of extreme values to trim from both ends.
        verbose: Verbosity level.

    Returns:
        Aggregated update.
    """
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
    predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
    unique_ws, counts = torch.unique(clients_weights, return_counts=True)
    if verbose >= 5:
        print(f'Unique weights: {unique_ws}, counts: {counts}')
    if trim_ratio < 0 or trim_ratio >= 0.5:
        raise ValueError(f'Trimming ratio ({trim_ratio}) must be between 0 and 0.5')

    if trim_method == 'count':
        # Here we use l2 norm to trim data, not coordinate-wise trimmed mean
        # trimmed by count: simple, not consider weights when trimming
        # Normalize weights
        clients_weights = clients_weights / torch.sum(clients_weights)

        # Compute Euclidean norms of updates
        norms = torch.norm(clients_updates, dim=1)  # l2 norm
        print('l2_norm: ', norms)

        # Sort updates based on their norms
        sorted_indices = torch.argsort(norms)  # which sort algorithm is used by default? quicksort is the default.
        sorted_updates = clients_updates[sorted_indices]
        sorted_weights = clients_weights[sorted_indices]
        sorted_predict_clients_type = predict_clients_type[sorted_indices]
        # Compute trim bounds
        # It will always remove at least 1 lower, and 1 upper, even trim_ratio=0
        trim_count = max(1, int(trim_ratio * N))
        lower, upper = trim_count, N - trim_count

        if verbose >= 5:
            print(f"Trimming {trim_count} smallest and {trim_count} largest updates.")

        # Compute weighted trimmed mean
        trimmed_updates = sorted_updates[lower:upper]
        # trimmed by count
        # If we have different weights, trim values like this is not a good way
        trimmed_weights = sorted_weights[lower:upper]

        if len(trimmed_updates) == 0:
            raise ValueError("Trimming removed all updates. Reduce trim_ratio.")

        weighted_update = torch.sum(trimmed_updates * trimmed_weights.unsqueeze(1), dim=0) / torch.sum(trimmed_weights)
        sorted_predict_clients_type[:lower] = 'Removed'
        sorted_predict_clients_type[upper:] = 'Removed'
        sorted_predict_clients_type[lower:upper] = "Chosen Update"
        predict_clients_type[sorted_indices] = sorted_predict_clients_type
    else:  # default
        # trimmed by cumulative weights. weighted coordinate-wise trimmed mean
        # Sort by distance and apply weighted trimming
        sorted_updates, sorted_indices = torch.sort(clients_updates, dim=0)  # sorted by each dimension
        # sorted clients_weights by each dimension and get the same shape as sorted_updates
        sorted_weights = clients_weights[sorted_indices]
        cumulative_weights = torch.cumsum(sorted_weights, dim=0)
        # D = sorted_updates.shape[1]
        total_weight = cumulative_weights[-1]

        # # original idea
        # weighted_update = torch.zeros((D,))
        # for dim in range(D):
        #     # lower_bound = torch.searchsorted(cumulative_weights[:, dim], trim_ratio * total_weight[dim],
        #     #                                  right=False)
        #     # When you slice a tensor, PyTorch will return a view of the original tensor. A view does not necessarily
        #     # guarantee contiguous memory.
        #     # The .contiguous() method returns a contiguous tensor that has the same data but stored in a
        #     # contiguous block of memory. This improves performance when performing operations that require a
        #     # contiguous block, such as torch.searchsorted.
        #     lower_bound = torch.searchsorted(cumulative_weights[:, dim].contiguous(), trim_ratio * total_weight[dim],
        #                                      right=False)
        #     upper_bound = torch.searchsorted(cumulative_weights[:, dim].contiguous(),
        #                                      (1 - trim_ratio) * total_weight[dim],
        #                                      right=True)
        #     print(f'For the {dim}th dimension, trimming lower index {lower_bound} and upper index {upper_bound}')
        #     # Apply mask to updates and weights
        #     up = sorted_updates[lower_bound:upper_bound + 1, dim]
        #     ws = sorted_weights[lower_bound:upper_bound + 1, dim]
        #     weighted_update[dim] = torch.sum(up * ws) / torch.sum(ws)
        # print(weighted_update)

        # Optimized implementation
        weight_threshold = trim_ratio * total_weight
        # torch.searchsorted(sorted_sequence, values), the operation works individually for each row in sorted_sequence.
        # i.e., for each row in sorted_sequence, we find the index that can insert one of value from values to
        # maintain the current row in the sorted order.
        # After cumulative_weights.T, each row represents all the weights for one dimension.
        # then for each row, torch.searchsorted(cumulative_weights.T, weight_threshold.reshape(-1, 1), side="left")
        # will find the (column) index where we can insert one weight from weight_threshold, so we will get D lower
        # indices in lower_bound
        lower_bound = torch.searchsorted(cumulative_weights.T.contiguous(), weight_threshold.reshape(-1, 1),
                                         side="left")
        upper_bound = torch.searchsorted(cumulative_weights.T.contiguous(), (total_weight -
                                                                             weight_threshold).reshape(-1, 1),
                                         side="right")
        # for each dimension, we trim each lower and upper bound data, then stack them togather.
        # strict adherence to weight thresholds (even if unbalanced)
        # notice that, if using lb:ub, we might remove different number of values from lower and upper side, however,
        # each side meets the search condition.
        # if using lb:ub+1, we will remove the same number of values from lower and upper side, however, we will
        # include more values (that might not meet our search condition) into the final results.
        # So we prefer to meet the search condition in trim_method='cumulative_weights', as trim_method='counts'
        # removes the same number of values from both sides.
        # you can try different weight_threshold to see the difference, e.g., weight_threshold=1.5 vs. 1.0
        trimmed_updates = torch.stack(
            [sorted_updates[lb:ub, dim] for dim, (lb, ub) in enumerate(zip(lower_bound, upper_bound))]).T
        trimmed_weights = torch.stack(
            [sorted_weights[lb:ub, dim] for dim, (lb, ub) in enumerate(zip(lower_bound, upper_bound))]).T

        if verbose >= 5:
            print(f"sorted_updates.shape: {sorted_updates.shape}, trimmed_updates.shape: {trimmed_updates.shape}, "
                  f"lower weight_threshold: {weight_threshold}, upper weight_threshold: "
                  f"{total_weight - weight_threshold}")
        # Compute weighted average
        weighted_update = torch.sum(trimmed_updates * trimmed_weights, dim=0) / torch.sum(trimmed_weights, dim=0)
        if verbose >= 5:
            print(f"weighted_update: {weighted_update}")
        # predict_clients_type[sorted_indices] = ?

    return weighted_update, predict_clients_type


def cw_median(clients_updates, clients_weights, verbose=1):
    """
        Compute the weighted median for flattened tensors.

        Args:
            clients_updates (list of torch.Tensor): A list of `N` flattened tensors (e.g., shape (d,)).
            clients_weights (torch.Tensor): A 1D tensor of shape (N,) representing the weights.
        Returns:
            torch.Tensor: The weighted median tensor, matching the shape of the first update.
        """
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
    predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
    unique_ws, counts = torch.unique(clients_weights, return_counts=True)
    if verbose >= 5:
        print(f'Unique weights: {unique_ws}, counts: {counts}')

    # PyTorch does not explicitly force -0.0 to appear before 0.0 (because it treats -0.0 equals to 0)
    # , as some other libraries (like NumPy) might do.
    # PyTorch uses a stable sorting algorithm, which means it preserves the original order when values are equal.
    # Since 0.0 and -0.0 are numerically equal, the one that appears first in the original tensor will stay first.
    # Sort updates along the first dimension (client axis)
    sorted_updates, sorted_indices = torch.sort(clients_updates, dim=0)  # sorted by each dimension: quicksort
    # sorted clients_weights by each dimension and get the same shape as sorted_updates
    sorted_weights = clients_weights[sorted_indices]

    # Compute cumulative sum of weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)

    total_weight = cumulative_weights[-1]  # Total weight for each element

    # Find first index where cumulative weight reaches 50% of total weight
    # .to(torch.int) will get the lower index
    median_index = (cumulative_weights >= (total_weight / 2)).to(dtype=torch.int).argmax(dim=0)

    # Select median values
    D = clients_updates.shape[1]
    # weighted_median_values = sorted_updates[median_index] # not correct.
    # sorted_updates[median_index, torch.arange(D)] picks the median_index[d]-th value from the d-th column.
    # torch.arange(D) represents the column indices.
    weighted_median = sorted_updates[median_index, torch.arange(D)]

    if verbose >= 5:
        # Find the original client indices whose updates were chosen
        original_median_indices = sorted_indices[median_index, torch.arange(D)]

        # print(f'Chosen update indices: {dict(collections.Counter(original_median_indices.tolist()))}')
        # Mark the client whose update was chosen as the median
        # predict_clients_type[original_median_indices.numpy()] = 'chosen update'  # {stacked_updates.numpy()}
        print(f'Chosen update: {dict(collections.Counter(original_median_indices.tolist()))}, '
              f'updates[0].shape: {tuple(clients_updates[0].shape)}, clients_weights: {clients_weights.numpy()}')

    return weighted_median, predict_clients_type


def geometric_median(clients_updates, clients_weights, init_value=None, max_iters=100, tol=1e-6, verbose=1):
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
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
    predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
    unique_ws, counts = torch.unique(clients_weights, return_counts=True)
    if verbose >= 5:
        print(f'Unique weights: {unique_ws}, counts: {counts}')

    # normalize weights first, which will be used in the following iteration
    clients_weights = clients_weights / torch.sum(clients_weights)

    if init_value is not None:
        estimated_center = init_value
    else:
        # Initial estimate: weighted average
        estimated_center = torch.sum(clients_updates * clients_weights.view(-1, 1), dim=0)

    for iteration in range(max_iters):
        distances = torch.norm(clients_updates - estimated_center, dim=1)
        distances = torch.clamp(distances, min=tol)  # Avoid division by zero

        inv_distances = 1.0 / distances
        weighted_inv_distances = clients_weights * inv_distances

        weighted_update = torch.sum(weighted_inv_distances.view(-1, 1) * clients_updates, dim=0) / torch.sum(
            weighted_inv_distances)

        diff = torch.norm(weighted_update - estimated_center).item()
        # Convergence check
        if diff < tol:
            break

        if verbose >= 20:
            print(f"Iteration {iteration}: Distance = {diff}")

        estimated_center = weighted_update

    if verbose >= 5:
        print(f"Geometric median found in {iteration + 1} iterations.")

    return weighted_update, predict_clients_type


def pairwise_distances(clients_updates):
    """
    Compute pairwise Euclidean distances between updates.
    :param updates: tensors
    :return: Pairwise distance matrix.
    """
    # num_updates = len(updates)
    # distances = np.zeros((num_updates, num_updates))
    # for i in range(num_updates):
    #     for j in range(i + 1, num_updates):
    #         distances[i, j] = np.linalg.norm(updates[i] - updates[j])
    #         distances[j, i] = distances[i, j]

    # # Numpy Version
    # updates = np.array(clients_updates)  # Ensure updates is a NumPy array
    # # Use broadcasting to compute pairwise distances
    # diff = updates[:, np.newaxis, :] - updates[np.newaxis, :, :]
    # distances = np.linalg.norm(diff, axis=2)
    # distances = torch.tensor(distances, device=clients_updates.device)

    # Tensor Version
    diff = clients_updates.unsqueeze(1) - clients_updates.unsqueeze(0)  # Broadcasting to compute pairwise differences
    distances = torch.norm(diff, dim=2)  # Compute Euclidean norm along the last dimension

    return distances


#
# def medoid(clients_updates, clients_weights, trimmed_average=False, p=0.1, verbose=1):
#     """
#
#     """
#
#     N, D = clients_updates.shape
#     assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
#     predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
#     unique_ws, counts = torch.unique(clients_weights, return_counts=True)
#     if verbose >= 5:
#         print(f'Unique weights: {unique_ws}, counts: {counts}')
#
#     # we don't need the weighted distance here.
#     distances = pairwise_distances(clients_updates)
#     # Compute sum of distances for each point to all others
#     distance_sums = torch.sum(distances, dim=0)
#
#     # Select the medoid (the point with the minimum sum of distances)
#     medoid_index = torch.argmin(distance_sums)
#
#     return clients_updates[medoid_index], predict_clients_type

def medoid(clients_updates, clients_weights, trimmed_average=False, upper_trimmed_ratio=0.1, verbose=1):
    """
    Computes the medoid from a set of client updates based on pairwise distances.

    Args:
        clients_updates (torch.Tensor): Tensor of shape (N, D) containing client updates.
        clients_weights (torch.Tensor): Tensor of shape (N,) containing weights for each client.
        trimmed_average (bool, optional): Whether to use trimmed averaging (not implemented here). Defaults to False.
        p (float, optional): A parameter for potential extensions (not used here). Defaults to 0.1.
        verbose (int, optional): Verbosity level for debugging. Defaults to 1.

    Returns:
        torch.Tensor: The selected medoid update.
        np.ndarray: Array indicating the predicted client type.
    """
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
    predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
    unique_ws, counts = torch.unique(clients_weights, return_counts=True)
    if verbose >= 5:
        print(f'Unique weights: {unique_ws}, counts: {counts}')

    # Compute pairwise distances (without weighted distances)
    distances = pairwise_distances(clients_updates)

    # Compute sum of distances for each point to all others
    total_distances = torch.sum(distances, dim=0)

    if trimmed_average:
        sorted_indices = torch.argsort(total_distances)
        sorted_updates = clients_updates[sorted_indices]
        sorted_weights = clients_weights[sorted_indices]
        sorted_predict_clients_type = predict_clients_type[sorted_indices]
        # # Remove by count,
        # m = N - f
        # # Compute weighted average of the top `m` smallest updates
        # update = torch.sum(sorted_updates[:m] * sorted_weights[:m].unsqueeze(1), dim=0) / torch.sum(sorted_weights[:m])

        # Remove by cumulative weights
        cumulative_weights = torch.cumsum(sorted_weights, dim=0)
        # D = sorted_updates.shape[1]
        total_weight = cumulative_weights[-1]

        # Optimized implementation
        weight_threshold = upper_trimmed_ratio * total_weight  # one value
        upper_bound = torch.searchsorted(cumulative_weights.contiguous(), total_weight - weight_threshold,
                                         side="right")
        trimmed_updates = sorted_updates[:upper_bound]
        trimmed_weights = sorted_weights[:upper_bound]
        sorted_predict_clients_type[upper_bound:] = 'Byzantine'
        if verbose >= 5:
            print(f"sorted_updates.shape: {sorted_updates.shape}, trimmed_updates.shape: {trimmed_updates.shape}, "
                  f"upper weight_threshold: {total_weight - weight_threshold}")
        # Compute weighted average
        weighted_update = torch.sum(trimmed_updates * trimmed_weights.unsqueeze(1), dim=0) / torch.sum(trimmed_weights)
        if verbose >= 5:
            print(f"weighted_update: {weighted_update}")
        predict_clients_type[sorted_indices] = sorted_predict_clients_type
        update = weighted_update
    else:
        # Select the medoid (point with the smallest sum of distances)
        medoid_index = torch.argmin(total_distances)
        predict_clients_type[medoid_index] = 'Chosen Update'
        update = clients_updates[medoid_index]

    return update, predict_clients_type


def krum(clients_updates, clients_weights, f, trimmed_average=False, random_projection=False, k_factor=10,
         random_state=42, verbose=1):
    """
    Krum aggregation for Byzantine-robust federated learning.

    :param clients_updates: Tensor of shape (N, d) containing model updates.
    :param clients_weights: Tensor of shape (N,) containing update weights.
    :param f: Number of Byzantine clients to tolerate.
    :param trimmed_average: Whether to use trimmed averaging instead of selecting one update.
    :param random_projection: Whether to apply random projection before computing distances.
    :param k_factor: Dimensionality reduction factor for random projection.
    :param random_state: Random seed for projection.
    :param verbose: Verbosity level.
    :return: Aggregated update and client type predictions.
    """
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
    predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
    unique_ws, counts = torch.unique(clients_weights, return_counts=True)
    if verbose >= 5:
        print(f'Unique weights: {unique_ws}, counts: {counts}')

    if 2 * f + 2 >= N:
        raise ValueError("Number of updates must be greater than 2 * f + 2.")

    # Normalize weights
    clients_weights = clients_weights / clients_weights.sum()
    # predict_clients_type = torch.tensor([f'Update {i}' for i in range(N)], dtype=torch.string)

    # Compute pairwise distances:   # we don't need the weighted distance here.
    if random_projection:
        projected_updates = conduct_random_projection(clients_updates, k_factor, random_state)
        distances = pairwise_distances(projected_updates)
    else:
        distances = pairwise_distances(clients_updates)

    if verbose >= 20:
        print(distances)

    k = N - f - 2  # Fixed Krum formula

    # Compute Krum scores
    scores = torch.zeros(N, dtype=torch.float32)
    for i in range(N):
        sorted_indices = torch.argsort(distances[i])
        sorted_distances = distances[i][sorted_indices]
        sorted_weights = clients_weights[sorted_indices]

        # Ensure first distance is self-distance (zero)
        if sorted_distances[0] != 0:
            raise ValueError("First distance should be zero (self-distance).")

        # # weight average
        # score = 0.0
        # weight = 0.0
        # if sorted_distances[0] != 0:
        #     raise ValueError
        # for j in range(1, k + 1):  # the first point is the itself, which should be 0 and we exclude it.
        #     score += sorted_distances[j] * sorted_weights[j]
        #     weight += sorted_weights[j]
        # score = score / weight

        # Compute weighted Krum score
        weighted_score = (sorted_distances[1:k + 1] * sorted_weights[1:k + 1]).sum() / sorted_weights[1:k + 1].sum()
        scores[i] = weighted_score

    if verbose >= 5:
        print('Krum scores:', [f'{v:.2f}' for v in scores.tolist()])

    if trimmed_average:
        # Select top `N-f` updates for trimmed averaging
        sorted_indices = torch.argsort(scores)
        sorted_weights = clients_weights[sorted_indices]
        sorted_updates = clients_updates[sorted_indices]
        sorted_predict_clients_type = predict_clients_type[sorted_indices]

        m = N - f
        if verbose >= 5:
            print(f'm: {m}')

        sorted_predict_clients_type[m:] = 'Byzantine'
        predict_clients_type[sorted_indices] = sorted_predict_clients_type

        # Compute weighted average of top `m` updates
        weighted_update = torch.sum(sorted_updates[:m] * sorted_weights[:m, None], dim=0) / sorted_weights[:m].sum()
    else:
        # Select update with the lowest Krum score
        selected_index = torch.argmin(scores)
        if verbose >= 5:
            print(f"Selected index: {selected_index}")
        weighted_update = clients_updates[selected_index]
        predict_clients_type[selected_index] = 'Chosen Update'

    return weighted_update, predict_clients_type


# def krum_with_random_projection(clients_updates, clients_weights, f, trimmed_average=False, k_factor=1,
#                                 random_state=42, verbose=1):
#     return krum(clients_updates, clients_weights, f, trimmed_average, True, k_factor,
#                 random_state, verbose)


def adaptive_krum(clients_updates, clients_weights, trimmed_average=False, random_projection=False, k_factor=10,
                  random_state=42, verbose=1):
    """
    adaptive Krum aggregation for Byzantine-robust federated learning.

    Args:
        clients_updates (Tensor): Client model updates, shape (N, d).
        clients_weights (Tensor): Weights of client updates, shape (N,).
        trimmed_average (bool): Whether to use trimmed averaging.
        random_projection (bool): Whether to apply random projection.
        k_factor (int): Projection dimensionality reduction factor.
        random_state (int): Random seed for projection.
        verbose (int): Verbosity level.

    Returns:
        Tuple[Tensor, np.ndarray]: Aggregated update and client type predictions.
    """
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."
    predict_clients_type = np.array([[f'Update {i}'] for i in range(N)], dtype='U20')
    unique_ws, counts = torch.unique(clients_weights, return_counts=True)
    if verbose >= 5:
        print(f'Unique weights: {unique_ws}, counts: {counts}')

    # Normalize weights
    clients_weights = clients_weights / torch.sum(clients_weights)

    # Compute pairwise distances:   # we don't need the weighted distance here.
    if random_projection:
        projected_updates = conduct_random_projection(clients_updates, k_factor, random_state)
        distances = pairwise_distances(projected_updates)
    else:
        distances = pairwise_distances(clients_updates)

    if verbose >= 20:
        print(distances)

    scores = torch.zeros(N, dtype=torch.float32)
    ks = []
    for i in range(N):
        # Sort distances for the current update and sum the closest (N-f-2) distances
        sorted_indices = np.argsort(distances[i])
        sorted_distances = distances[i][sorted_indices]
        sorted_weights = clients_weights[sorted_indices]

        # # Calculate the middle point
        # n2 = len(sorted_indices)  # n2 must be equal to N
        # middle_index = (n2 - 1) // 2  # lower index for even updates
        #
        # # Initialize max difference and index
        # max_diff = -1
        # k = middle_index  # default value for k
        # # Find the first occurrence of the max diff in one pass
        # for j in range(middle_index, N - 1):
        #     t = sorted_distances[j + 1] - sorted_distances[j]
        #     if t < 0 or sorted_distances[j] < 0 or sorted_distances[j + 1] < 0:
        #         # if it is true, there must be an error as the distance must be >=0.
        #         raise ValueError("Distance values must be non-negative.")
        #     if t > max_diff:
        #         max_diff = t
        #         k = j  # Store the first occurrence of max diff

        # breakpoints = binary_segmentation(sorted_distances)
        # h = n-f , 2+2f < n => 2*f <= n-2-1, so f <= (n-3)//2, h = n - f = n - (n-3)//2
        # each point must be >= half of data neighbors, as f is strictly less than half of data
        h = N - (N-3)//2    # the number of honest points
        k = find_significant_change_point(sorted_distances, start=h)
        ks.append(k)
        if verbose >= 20:
            # print(f'*** j: {j} ***')
            print(f'sorted_distances: {sorted_distances}')
            # print('diff_dists: ', diff_dists)
            print("Index of maximum distance difference after half of values:", k)

        # Ensure first distance is self-distance (zero)
        if sorted_distances[0] != 0:
            raise ValueError("First distance should be zero (self-distance).")

        scores[i] = (sorted_distances[1:k] * sorted_weights[1:k]).sum() / sorted_weights[1:k].sum()

    if verbose >= 5:
        print('adaptive_Krum scores: ', [f'{v:.2f}' for v in scores.tolist()])
        print(f'ks: {ks}')

    if trimmed_average:
        # instead return the smallest value, we return the top weighted average
        # Sort scores
        sorted_indices = torch.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = clients_weights[sorted_indices]
        sorted_updates = clients_updates[sorted_indices]
        sorted_predict_clients_type = predict_clients_type[sorted_indices]

        # # Find the index of the maximum value (after the halfway point)
        # # Calculate the middle point
        # middle_index = (N - 1) // 2
        #
        # # Initialize max difference and index
        # max_diff = -1
        # k = middle_index  # default value for k
        # # Find the first occurrence of the max diff in one pass
        # for j in range(middle_index, N - 1):
        #     t = sorted_scores[j + 1] - sorted_scores[j]
        #     if t < 0 or sorted_scores[j] < 0 or sorted_scores[j + 1] < 0:
        #         # if it is true, there must be an error as the distance must be >=0.
        #         raise ValueError("Score values must be non-negative.")
        #     if t > max_diff:
        #         max_diff = t
        #         k = j  # Store the first occurrence of max diff

        k = find_significant_change_point(sorted_scores, start=h)

        if verbose >= 20:
            print(f'trimmed_average: sorted_scores: {sorted_scores}')
            print("Index of maximum scores difference after half of values:", k)

        m = k  # we will average over top m closet updates
        if verbose >= 5:
            print(f'm: {m}')
        sorted_predict_clients_type[m:] = 'Byzantine'
        # **Map the sorted labels back to original order**
        predict_clients_type[sorted_indices] = sorted_predict_clients_type

        weighted_update = torch.sum(sorted_updates[:m] * sorted_weights[:m, None], dim=0) / torch.sum(
            sorted_weights[:m])
    else:
        # Select the update with the smallest score
        selected_index = np.argmin(scores)
        if verbose >= 5:
            print(f"selected_index: {selected_index}")
        weighted_update = clients_updates[selected_index]

        predict_clients_type[selected_index] = 'Chosen Update'

    # print(weighted_update)
    # print(predict_clients_type)
    return weighted_update, predict_clients_type


# def adaptive_krum_with_random_projection(clients_updates, clients_weights,
#                                          trimmed_average=False, k_factor=10,
#                                          random_state=42, verbose=1):
#     return adaptive_krum(clients_updates, clients_weights, trimmed_average, True, k_factor,
#                          random_state, verbose)


def conduct_random_projection(updates, k_factor, random_state=42, verbose=1):
    """
    Applies random projection to reduce the dimensionality of model updates.

    Args:
        updates (torch.Tensor): A tensor of shape (num_updates, num_features) representing model updates.
        k_factor (int): Factor controlling the reduced dimension, calculated as k_factor * log(num_updates).
        random_state (int, optional): Seed for reproducibility. Default is 42.
        verbose (int, optional): Verbosity level for logging. Default is 1.

    Returns:
        torch.Tensor: Projected updates with reduced dimensionality.
    """
    start = time.time()
    N, D = updates.shape

    k = k_factor * int(np.log(N))  # Reduced dimension using log(N)
    if k >= D:
        raise ValueError("k must be smaller than D.")
    # k = min(D, k_factor * int(np.log(N)))
    print(f'Projected dimensions: {k}, N: {N}, D: {D}')

    # # Create a random projection transformer
    # transformer = GaussianRandomProjection(n_components=k)
    # projected_updates = transformer.fit_transform(np.array(updates))
    # Generate a random projection matrix with shape (num_features, k)

    # Set the seed for reproducibility
    torch.manual_seed(random_state)  # This sets the seed for all random operations

    random_projection_matrix = torch.randn(D, k, dtype=torch.float).to(updates.device)
    # Perform the projection by multiplying the updates with the projection matrix
    projected_updates = torch.matmul(updates, random_projection_matrix)

    end = time.time()
    if verbose >= 1:
        print(f'Random projection completed in {end - start:.4f} seconds')
        print(f'Projected updates shape: {projected_updates.shape}')
    return projected_updates


def main():
    import time
    results = []
    verbose = 20
    time_taken_list = []
    num_repetitions = 10
    for i in range(num_repetitions):
        print(f'\nthe {i}th trial: ')
        # Example updates from clients
        dim = 30000
        # clients_updates = [
        #     np.random.randn(dim),  # Update from client 1
        #     np.random.randn(dim),  # Update from client 2
        #     np.random.randn(dim),  # Update from client 3
        #     np.random.randn(dim),  # Update from client 4
        #     np.random.randn(dim) + 10,  # Malicious update
        # ]
        # if number of clients is too small, with Random Projection will take more time.
        n = 300
        # Number of Byzantine clients to tolerate
        f = (n-3)//2
        h = n-f
        clients_updates = [np.random.randn(dim)] * h + [np.random.randn(dim) + f]
        clients_updates = torch.stack([torch.tensor(v, dtype=torch.float) for v in clients_updates])
        weights = torch.tensor([1] * len(clients_updates))

        # Perform Krum aggregation
        trimmed_average = True
        # print('Krum...')
        # aggregated_update = krum_with_random_projection(clients_updates, weights, f, trimmed_average, verbose=verbose)
        # print("Aggregated Update (Krum):", aggregated_update)
        print('\nadaptive Krum...')
        start = time.time()
        aggregated_update2 = adaptive_krum(clients_updates, weights, trimmed_average, verbose=verbose)
        end = time.time()
        time_taken = end - start
        print("Aggregated Update (adaptive Krum):", aggregated_update2, time_taken)

        print('\nadaptive Krum with Random Projection...')
        start = time.time()
        aggregated_update2 = adaptive_krum(clients_updates, weights, trimmed_average, random_projection=True,
                                                                  verbose=verbose)
        end = time.time()
        time_taken2 = end - start
        print("Aggregated Update (adaptive Krum) with RP:", aggregated_update2, time_taken2)

        time_taken_list.append([time_taken, time_taken2])
        # if np.sum(aggregated_update2.numpy() - aggregated_update.numpy()) != 0:
        #     print("Different updates were aggregated")
        #     results.append(clients_updates)
        # break
    print(f'\naccuracy: {1 - len(results) / num_repetitions}')

    from matplotlib import pyplot as plt
    xs = range(len(time_taken_list))
    ys = [vs[0] for vs in time_taken_list]
    plt.plot(xs, ys, label='Without RP', color='b', marker='*')
    ys = [vs[1] for vs in time_taken_list]
    plt.plot(xs, ys, label='with RP', color='r', marker='s')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
