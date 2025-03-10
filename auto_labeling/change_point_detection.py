import numpy as np

import torch


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


def main():
    # h = n-f , 2+2f < n => 2*f <= n-2-1, so f <= (n-3)//2, h = n - f = n - (n-3)//2
    # each point must be >= half of data neighbors, as f is strictly less than half of data
    for N in [5, 6, 7, 8, 9, 10, 100, 1000]:
        f = (N - 3) // 2
        h = N - (N - 3) // 2  # the number of honest points
        print(N, f, h)
    # # Example usage
    # data = np.random.randn(100)  # Example time series data
    # breakpoints = binary_segmentation(data)
    #
    # print("Detected change points:", breakpoints)

    # Example usage
    data = torch.randn(100)  # Example time series data as a torch tensor
    change_point = find_significant_change_point(data)

    print("Most significant change point:", change_point)


if __name__ == '__main__':
    main()
