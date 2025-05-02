import numpy as np

import torch

import numpy as np


def pelt(data, penalty):
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


def find_significant_change_point(data, start=1):
    """
    Find the most significant change point in O(n) time.

    Arguments:
    - data: Time series data (torch tensor)

    Returns:
    - change_point: The index of the most significant change point
    """
    n = len(data)

    # Compute prefix sums for efficient mean and variance calculation
    prefix_sum = torch.cumsum(data, dim=0)
    prefix_sq_sum = torch.cumsum(data ** 2, dim=0)

    best_cost = float('inf')
    change_point = -1

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
            change_point = i

    return change_point

def find_change_point_with_knee(sorted_distances, start):
    from kneed import KneeLocator
    # Use KneeLocator to find the jump point
    knee = KneeLocator(range(len(sorted_distances)), sorted_distances, curve='convex', direction='increasing')
    index_jump_point = knee.knee

    return max(index_jump_point, start)



def main():
    import matplotlib.pyplot as plt

    # h = n-f , 2+2f < n => 2*f <= n-2-1, so f <= (n-3)//2, h = n - f = n - (n-3)//2
    # each point must be >= half of data neighbors, as f is strictly less than half of data
    for N in [5, 6, 7, 8, 9, 10, 100, 1000]:
        f = (N - 3) // 2
        h = N - (N - 3) // 2  # the number of honest points
        h2 = (N+3)//2
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
    main()
