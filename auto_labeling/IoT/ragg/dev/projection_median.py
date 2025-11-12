import torch
import numpy as np


def partition(arr, pivot):
    less = arr[arr < pivot]
    equal = arr[arr == pivot]
    greater = arr[arr > pivot]
    return less, equal, greater


def median_of_medians(arr):
    """
    Args:
        arr:

    Returns:

    """

    # # If the list is small, just return the median
    # if len(arr) <= 5:
    #     return torch.median(arr).item()
    #
    # # Group elements into chunks of 5
    # if len(arr) % 5:
    #     # Pad the array to be divisible by 5
    #     padding_size = (5 - len(arr) % 5)
    #     arr = torch.cat([arr, torch.full((padding_size,), float('inf'))])  # Pad with inf
    #
    # # Group elements into chunks of 5
    # chunks = arr.view(-1, 5)  # Reshape into chunks of 5 elements
    # medians = torch.median(chunks, dim=1).values  # Median of each chunk
    #
    # # Recursively find the median of the medians
    # return median_of_medians(medians)

    while len(arr) > 5:
        # If the length is not a multiple of 5, pad the array to be divisible by 5
        if len(arr) % 5 != 0:
            padding_size = (5 - len(arr) % 5)
            arr = torch.cat([arr, torch.full((padding_size,), float('inf'))])  # Pad with inf

        # Group elements into chunks of 5
        chunks = arr.view(-1, 5)  # Reshape into chunks of 5 elements
        medians = torch.median(chunks, dim=1).values  # Median of each chunk

        # Update the array with the medians to process the next iteration
        arr = medians

        # Once the array length is <= 5, return the median of the remaining elements
    return torch.median(arr).item()


def select_kth(arr, k):
    # Find the pivot using Median of Medians
    pivot = median_of_medians(arr)

    # Partition the array around the pivot
    less, equal, greater = partition(arr, pivot)

    # Based on the size of 'less', determine the position of the k-th element
    if k < len(less):
        return select_kth(less, k)
    elif k < len(less) + len(equal):
        return equal[0].item()
    else:
        return select_kth(greater, k - len(less) - len(equal))


def find_median(arr):
    n = len(arr)
    # if n % 2 == 1:
    #     return select_kth(arr, n // 2)
    # else:
    #     return (select_kth(arr, n // 2 - 1) + select_kth(arr, n // 2)) / 2

    # always return the lower median
    return select_kth(arr, n // 2 - 1)


def projection_median(points, weights = None, num_projections=10, verbose=False):
    """
    Find the projection median in R^d using the Median of Medians for projected points.

    points: tensor of shape (n, d), where n is the number of points and d is the dimensionality
    num_projections: number of random projections to use (default 10)

    Returns the projection median point in R^d.
    """
    n, d = points.shape

    # Generate random directions in a batch (num_projections, d)
    random_directions = torch.randn(num_projections, d, dtype=points.dtype, device=points.device)
    random_directions /= random_directions.norm(dim=1, keepdim=True)  # Normalize each direction

    # Project all points onto all random directions (n, num_projections)
    projections = torch.matmul(points, random_directions.T)  # Matrix multiplication (n, num_projections)

    # Find the median of each projection (across the rows for each projection direction)
    medians = torch.median(projections, dim=0).values  # Median for each column (projection direction)
    # medians = torch.tensor([find_median(projection_id) for projection_id in projections])
    medians = medians.unsqueeze(dim=1) * random_directions

    # Calculate the mean of the projection medians (final projection median)
    # projection_median_value, indices = torch.median(medians, dim=0)
    projection_median_value = torch.mean(medians, dim=0)

    return projection_median_value


if __name__ == '__main__':

    # Example usage
    # Generate random points in R^d (for example, 100 points in 3D space)
    n, d = 100, 3
    points = torch.randn(n, d)

    # Compute the projection median
    projection_median_value = projection_median(points, num_projections=10)

    print(f"Projection median value: {projection_median_value}")
