"""
    Case 1 (large values): Median aggregation fails due to small but consistent multi-dimensional shifts

        Median Does Not Consider Relationship Between Updates (Geometry)
        The median is computed independently per dimension of the gradient vector.
        It does not consider correlations across dimensions, making it vulnerable to attacks that introduce small
        biases across multiple dimensions.

        The median computes each dimension independently, so it doesn't account for correlations across dimensions.
            This cumulative effect in each dimension can cause the median to drift away from the true central value.
        In contrast, Krum uses Euclidean distance to evaluate the overall similarity of updates,
            making it resilient to such multi-dimensional attacks.

        e.g.,
            honest updates [(0, 1),
                            (1, 0),
                            (0.5, 0.5),
                            (0.5, 0.5)]
            malicious updates [(1, 1),
                                (1, 1),
                                (1, 1)]

            median is (1, 1), which is shifted
            Krum is (0.5, 0.5)

            In fact, any value > 0.5 will shift the median? not work
            E.g.,
            malicious updates [(0.6, 0.8),
                                (0.7, 0.8),
                                (0.9, 1.0)]
            then
            median is (0.6, 0.8)
            Krum is (0.5, 0.5)

     Case 2: Median is Inefficient in High-Dimensional Models
        Deep learning models have millions of parameters, and computing the median across all dimensions is
        computationally expensive.
        Sorting high-dimensional tensors scales poorly compared to Krum, which only computes pairwise distances.
        Krum: Scales better for high-dimensional neural network models.


    Case 3: Sign-flipping attack

"""

import torch

from robust_aggregation import krum, median, refined_krum


def case():
    """ the case shows that Krum is better than Median

        honest point 1 = (0.1, 0.1)
        honest point 2 = (0.15, 0.15)
        honest point 3 = (0.2, 0.2)
        malicious point 1 = (10, 10)
        malicious point 2 = (20, 20)

        Median is (0.2, 0.2) as Median won’t ignore the effect of malicious points. If honest points has outlier,
                median will choose the outlier point
        Krum is (0.15, 0.15): Krum will ignore the malicious points first, then find the smallest point as the
                central point.

    Returns:

    """

    attack_type = "sign_flipping"
    if attack_type == "large_value":
        honest_points = [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2)]
        malicious_points = [(1, 1), (2, 2)]
    elif attack_type == "sign_flipping":
        honest_points = [(0.4, .0), (0.5, 0.6), (0.6, 0.5), (0, 0.4)]
        # honest_points = [(0.4, .0), (0.5, 0.6)]
        malicious_points = [(v[0] * (-1), v[1] * (-1)) for v in honest_points[:len(honest_points) - 1]]
        # malicious_points = [(1.5, 1.2), (1.2, 1.0), (1.0, 2.0)]
    else:
        raise NotImplementedError

    f = len(malicious_points)

    points = [torch.tensor(p) for p in honest_points + malicious_points]
    print(points)
    n = len(points)
    weights = torch.tensor([1] * n)

    point, clients_type = median(points, weights)
    print(f'median: {point}, {clients_type}')
    print()

    point, clients_type = krum(points, weights, f, trimmed_average=False, verbose=30)
    print(f'krum: {point}, {clients_type}')

    # point, clients_type = refined_krum(points, weights, trimmed_average=False, verbose=30)
    # print(f'refined_krum: {point}, {clients_type}')


if __name__ == '__main__':
    case()
