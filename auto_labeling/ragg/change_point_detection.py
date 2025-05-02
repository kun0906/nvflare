import numpy as np

import torch

import numpy as np


def pelt(data, penalty=1):
    """
    Pruned Exact Linear Time (PELT) for change point detection.

    Arguments:
    - data: The time series data.
    - penalty: The penalty term added for each change point to control the number of segments.

    Returns:
    - breakpoints: List of detected change points.
    - final_cost: The final minimal cost after segmentation.
    """
    n = len(data)  # Number of data points
    cost = np.inf * np.ones(n + 1)  # Initialize cost array with infinity
    cost[0] = 0  # Base case: cost for an empty segment is zero

    prev = -1 * np.ones(n + 1, dtype=int)  # Array to track previous breakpoints

    for end in range(1, n + 1):  # Iterate over all possible endpoints for segments
        for start in range(0, end):  # For each end point, try all possible start points
            # Compute the cost for this segment
            segment_cost = cost_function(data, start, end)

            # Check the total cost for the segmentation (previous cost + new segment cost + penalty)
            current_cost = cost[start] + segment_cost + penalty

            # Update cost and previous breakpoint if a better segmentation is found
            if current_cost < cost[end]:
                cost[end] = current_cost
                prev[end] = start

    # Reconstruct the breakpoints from the 'prev' array
    breakpoints = []
    end = n
    while prev[end] != -1:
        breakpoints.append(prev[end])
        end = prev[end]

    breakpoints = list(reversed(breakpoints))  # Reverse to get the correct order
    return breakpoints, cost[n]


def cost_function(data, start, end):
    """ Calculate the cost for the segment [start:end] as the sum of squared errors """
    segment = data[start:end]
    mean_val = torch.mean(segment)  # Mean of the segment
    cost = torch.sum((segment - mean_val) ** 2)  # Sum of squared errors (SSE)
    return cost


def binary_segmentation(data, min_size=5):
    """
    Binary Segmentation for change point detection.

    Arguments:
    - data: Time series data (array)
    - min_size: Minimum size of segment for further segmentation (avoid infinite recursion)

    Returns:
    - breakpoints: List of detected change points
    """

    def find_change_point(start, end):
        """Find the change point between [start:end]"""
        best_cost = float('inf')
        best_index = -1
        for i in range(start + 1, end):
            # Check the cost for splitting at each possible point
            left_cost = cost_function(data, start, i)
            right_cost = cost_function(data, i, end)
            total_cost = left_cost + right_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_index = i

        return best_index

    def segment_recursive(start, end):
        """ Recursively split the data into segments and find change points """
        if end - start <= min_size:
            return []  # Base case: segment too small to split further

        change_point = find_change_point(start, end)

        if change_point == -1:
            return []  # No valid change point

        # Recursively segment the left and right parts
        left_points = segment_recursive(start, change_point)
        right_points = segment_recursive(change_point, end)

        return left_points + [change_point] + right_points

    # Detect breakpoints using binary segmentation
    breakpoints = segment_recursive(0, len(data))
    return breakpoints


#
# def find_significant_change_point(data, start=1):
#     """
#     Find the most significant change point in the data.
#
#     Arguments:
#     - data: Time series data (torch tensor)
#
#     Returns:
#     - change_point: The index of the most significant change point
#     """
#     n = len(data)
#     best_cost = float('inf')
#     change_point = -1
#
#     # Iterate over possible change points
#     for i in range(start, n):
#         left_cost = cost_function(data, 0, i)  # Cost for the left segment [0:i)
#         right_cost = cost_function(data, i, n)  # Cost for the right segment [i:n)
#
#         total_cost = left_cost + right_cost  # Total cost for splitting at point i
#
#         if total_cost < best_cost:  # We are minimizing the cost
#             best_cost = total_cost
#             change_point = i
#
#     return change_point
#


def elbow_index_max_distance(values):
    """
    Find the elbow point as the index with max perpendicular distance
    to the line connecting first and last points (geometric method).

    Args:
        values (list or np.ndarray): Sorted list of values (e.g., distances)

    Returns:
        int: Index of elbow point
    """
    values = np.array(values)
    n = len(values)
    if n < 3:
        return None  # Not enough points

    # Coordinates of first and last points
    x = np.arange(n)
    x_start, x_end = 0, n - 1
    y_start, y_end = values[0], values[-1]

    # Line vector
    line_vec = np.array([x_end - x_start, y_end - y_start])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    max_dist = -1
    elbow_idx = -1

    for i in range(1, n - 1):  # Avoid endpoints
        point = np.array([x[i] - x_start, values[i] - y_start])
        # Compute perpendicular distance to the line
        proj = np.dot(point, line_vec_norm) * line_vec_norm
        orth = point - proj
        dist = np.linalg.norm(orth)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    return elbow_idx


def filter_large_values(distances):
    """
        Filter large values based on distances with median as pivot.
    Args:
        distances: must be sorted distances

    Returns:
        jump_idx: the first value (with this jump idx) that is larger than that (median + left_max_diff)
    """
    n = len(distances)
    mid = (n - 1) // 2
    median = distances[mid]  # lower median
    # print(median)
    left_max_diff = median - distances[0]
    assert left_max_diff >= 0

    for i in range(mid + 1, n):
        if distances[i] > (median + left_max_diff):
            return i

    # If no value is larger than (median + left_max_diff), we return the length
    return n


def find_significant_change_point(data, start=1):
    """
    Find the most significant change point in O(n) time.

    Arguments:
    - data: Time series data (torch tensor)

    Returns:
    - change_point: The index of the most significant change point
    """
    jump_idx = filter_large_values(data)
    data = data[: jump_idx]     # excluding the outliers or extremely large value first.

    n = len(data)

    # Compute prefix sums for efficient mean and variance calculation
    prefix_sum = torch.cumsum(data, dim=0)
    prefix_sq_sum = torch.cumsum(data ** 2, dim=0)

    best_cost = float('inf')
    change_point = n

    for i in range(start, n):  # We start from 1 to ensure two non-empty segments
        # Compute left segment mean and squared error
        # Here is correct, as the index is started from 0, so prefix_sum[i-1] is the sum of the first i values,
        # so we divide by i (not i-1)
        left_mean = prefix_sum[i - 1] / i
        # SSE = \sum_t^{i} ((x_t - mean)^2) = \sum_t^{i} (x_t^2 - 2 x_t mean + mean^2)
        # = \sum_t^{i} x_t^2 - 2 \sum_t^{i} x_t mean + i * mean^2 = \sum_t^{i} x_t^2 - i * mean^2
        left_sse = prefix_sq_sum[i - 1] - i * left_mean ** 2  # SSE = sum(x^2) - N * mean^2

        # Compute right segment mean and squared error
        right_n = n - i
        right_mean = (prefix_sum[n - 1] - prefix_sum[i - 1]) / right_n
        right_sse = (prefix_sq_sum[n - 1] - prefix_sq_sum[i - 1]) - right_n * right_mean ** 2

        total_cost = left_sse + right_sse  # Minimize the sum of squared errors

        if total_cost < best_cost:
            best_cost = total_cost
            change_point = i+1      # the ith item is included in left sum, so we should return i+1

    return change_point


def find_change_point_with_knee(sorted_distances, start):
    from kneed import KneeLocator
    # Use KneeLocator to find the jump point
    knee = KneeLocator(range(len(sorted_distances)), sorted_distances, curve='convex', direction='increasing')
    index_jump_point = knee.knee

    return max(index_jump_point, start)


def second_derivative_jump(values):
    # Compute first and second differences
    first_diff = np.diff(values)
    second_diff = np.diff(first_diff)
    # Pad to align index
    second_diff = np.pad(second_diff, (2, 0), mode='constant')
    return np.argmax(second_diff)


def max_difference_jump(values):
    differences = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return np.argmax(differences)


def max_ratio_jump(values):
    """, not work for this: [0.3, 10, 20, 50, 100, 100, 10000]
    Find the index where the ratio of (value[i+1] / value[i]) is maximized.
    This helps identify a sudden jump in sorted distances.
    """
    ratios = [values[i + 1] / values[i] if values[i] > 0 else 0 for i in range(len(values) - 1)]
    return np.argmax(ratios)


def mean_split_index(values):
    min_diff = float('inf')
    best_index = -1
    for i in range(1, len(values) - 1):  # avoid trivial splits
        mean_left = np.mean(values[:i])
        mean_right = np.mean(values[i:])
        diff = abs(mean_left - mean_right)
        # print(i, mean_left, mean_right, diff)
        # diff = mean_left + mean_right
        if diff < min_diff:
            min_diff = diff
            best_index = i
    return best_index


def mean_split_index2(values):
    best_score = -1
    best_index = -1
    for i in range(1, len(values) - 1):  # avoid trivial splits
        left = values[:i]
        right = values[i:]
        mean_left = np.mean(left)
        mean_right = np.mean(right)
        if mean_left == 0:  # prevent divide-by-zero
            continue
        ratio = mean_right / mean_left
        if ratio > best_score:
            best_score = ratio
            best_index = i
    return best_index


def log_knee(values):
    from kneed import KneeLocator
    log_vals = np.log1p(values)  # log1p to handle zeros safely
    # log_vals = values.copy()
    # print(log_vals)
    x = np.arange(len(values))
    kneedle = KneeLocator(x, log_vals, curve='convex', direction='increasing')
    return kneedle.knee


def pelt_lib(distances):
    import ruptures as rpt
    # PELT expects 2D input
    distances_2d = distances.reshape(-1, 1)

    # Create PELT model
    model = rpt.Pelt(model="l2").fit(distances_2d)

    # Detect change points
    # You can tune the penalty value to control sensitivity
    # penalty_value = 5  # Try adjusting this
    # penalty_value = np.log(len(distances)) * np.var(distances)  # Bayesian Information Criterion:
    #  Max Gradient Heuristic
    gradients = np.diff(distances)
    penalty_value = np.max(gradients)  # or tune multiplier
    change_points = model.predict(pen=penalty_value)

    # Show result
    print("Change points (PELT):", change_points)

    return change_points


def compare():
    import numpy as np
    import matplotlib.pyplot as plt
    from kneed import KneeLocator

    # Sample sorted distances
    distances = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                          0.10, 0.15, 0.2, 0.24, 0.29, 10, 20, 50, 100, 100, 100000000])

    distances = np.array([0.01, 0.011, 0.012, 0.013, 0.014, 0.016, 0.017, 0.018,
                          0.10, 0.2, 0.21, 0.24, 0.29, 10, 20, 50, 100, 100, 100000000])
    # distances = np.log1p(distances)
    # print(distances)
    mid = len(distances) // 2
    median = distances[mid]  # lower median
    print(median)
    left_diff = median - distances[0]

    # preprocessing
    for i, v in enumerate(distances):
        if i > mid:
            if v < (median + left_diff):
                v1 = v
            else:  #
                # v1 = (median + left_diff)
                max_i = i
                break
    distances = distances[mid:max_i]
    print(distances)

    x = np.arange(len(distances))

    # ---- 1. Elbow Method using KneeLocator ----
    knee_locator = KneeLocator(x, distances, curve='convex', direction='increasing')
    elbow_index = knee_locator.knee

    mean_split_idx = mean_split_index(distances)
    mean_split_idx2 = mean_split_index2(distances)
    sse_idx = find_significant_change_point(torch.tensor(distances))
    max_diff_idx = max_difference_jump(distances)
    max_idx = max_ratio_jump(distances)
    second_idx = second_derivative_jump(distances)
    log_knee_idx = log_knee(distances)

    pelt_idx = pelt_lib(distances)
    print(pelt_idx)
    pelt_idx = pelt_idx[0]
    # pelt_idx = pelt(torch.tensor(distances))
    # ---- Output Results ----
    print(f"Elbow method: index = {elbow_index}, value = {distances[elbow_index]}")
    print(f"Mean split:  index = {mean_split_idx}, value = {distances[mean_split_idx]}")
    print(f"Mean split2:  index = {mean_split_idx2}, value = {distances[mean_split_idx2]}")
    print(f"SSE:  index = {sse_idx}, value = {distances[mean_split_idx2]}")
    print(f"max_diff_idx:  index = {max_diff_idx}, value = {distances[max_diff_idx]}")
    print(f"max_idx:  index = {max_idx}, value = {distances[max_idx]}")
    print(f"second_idx:  index = {second_idx}, value = {distances[second_idx]}")
    print(f"log_knee_idx:  index = {log_knee_idx}, value = {distances[log_knee_idx]}")
    # print(f"pelt_idx:  index = {pelt_idx}, value = {distances[pelt_idx]}")

    # ---- Plot for Comparison ----
    plt.figure(figsize=(8, 5))
    plt.plot(x, distances, marker='o', label='Sorted Distances')

    # Mark Elbow
    # plt.axvline(x=elbow_index, color='red', linestyle='--', label=f'Elbow: idx={elbow_index}')
    # plt.scatter(elbow_index, distances[elbow_index], color='red')

    # Mark Mean Split
    # plt.axvline(x=mean_split_idx, color='blue', linestyle='--', label=f'Mean Split: idx={mean_split_idx}')
    # plt.scatter(mean_split_idx, distances[mean_split_idx], color='blue')

    plt.title('Elbow vs. Mean Split for Jump Detection')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    import matplotlib.pyplot as plt

    # h = n-f , 2+2f < n => 2*f <= n-2-1, so f <= (n-3)//2, h = n - f = n - (n-3)//2
    # each point must be >= half of data neighbors, as f is strictly less than half of data
    for N in [5, 6, 7, 8, 9, 10, 100, 1000]:
        f = (N - 3) // 2
        h = N - (N - 3) // 2  # the number of honest points
        h2 = (N + 3) // 2
        print(N, f, h, h2)

    data = torch.randn(100)  # Example time series data as a torch tensor

    plt.plot(range(data.shape[0]), data)
    plt.show()

    breakpoints = binary_segmentation(data)

    print("Detected change points:", breakpoints)

    # Example usage
    # data = torch.randn(100)  # Example time series data as a torch tensor
    change_point = find_significant_change_point(data)
    print("Most significant change point:", change_point)

    # Example usage
    penalty = 10  # Penalty term to control the number of change points
    # Apply PELT algorithm
    breakpoints, final_cost = pelt(data, penalty)

    print("Detected change points:", breakpoints)
    print("Final cost:", final_cost)


if __name__ == '__main__':
    # main()
    compare()
