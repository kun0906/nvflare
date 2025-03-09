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
                            (0.1, 0.1),
                            (1, 0),
                            (0.5, 0.5),
                            (0.5, 0.5)]
            malicious updates [(-1, -1)]

            coordinate-wise median = (n-1)//2, the median is always a point of the given data

            median is (0, 0), which is shifted
            Krum is (0.5, 0.5)

            In fact, any value < 0 will shift the median? not work
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
import numpy as np
import robust_aggregation
import matplotlib.pyplot as plt

VERBOSE=30
def median_case():
    """ the case shows that Krum is better than coordinate wise Median (order median)
    """
    # n is the total number of points, where up to f are byzantine, honest = n-f = n//2 + 2
    # Krum requires that 2+2f < n (total points), so f < (n-2)/2, so we set f = (n-2)//2 - 1 = n//2 - 2
    # if n = 5, f < (n-2)/2 = 1.5, so f can be 1.  e.g., n = 5, f = 1, honest = 4

    attack_type = "large_value"
    if attack_type == "large_value":
        # honest_points = [(0.1, 0.1), (0, 1), (1, 0), (0.5, 0.5), (0.5, 0.5)]
        # malicious_points = [(-1, -1)]
        honest_points = np.asarray([(0.5, 0.45), (0.45, 0.5), (0.8, 0.9), (1.0, 0.8)])
        byzantine_points = np.asarray([(1.5, 1.5)])
    elif attack_type == "sign_flipping":
        honest_points = [(0.4, .0), (0.5, 0.6), (0.6, 0.5), (0, 0.4)]
        # honest_points = [(0.4, .0), (0.5, 0.6)]
        byzantine_points = [(v[0] * (-1), v[1] * (-1)) for v in honest_points[:len(honest_points) - 1]]
        # malicious_points = [(1.5, 1.2), (1.2, 1.0), (1.0, 2.0)]
    else:
        raise NotImplementedError

    f = len(byzantine_points)
    # byzantine points must be appended to the end if using the first n-f points to compute true mean
    points = [torch.tensor(p) for p in np.concatenate([byzantine_points, honest_points])]
    points = torch.stack(points)
    print(points)
    n = len(points)
    weights = torch.tensor([1] * n)
    if 2 + 2 * f >= n or f + len(honest_points) != n:
        raise ValueError(f, n)
    print(f'n:{n}, f:{f}, h:{n - f}')
    # True median if only honest points were considered
    true_cw_median, clients_type = robust_aggregation.cw_median(points[:n - f], weights[:n - f])
    print(f'true_cw_median: {true_cw_median}, {clients_type}')

    cw_median, clients_type = robust_aggregation.cw_median(points, weights)
    print(f'cw_median: {cw_median}, {clients_type}')
    print()

    krum_point, clients_type = robust_aggregation.krum(points, weights, f, trimmed_average=False, verbose=30)
    print(f'krum: {krum_point}, {clients_type}')

    # point, clients_type = adaptive_krum(points, weights, trimmed_average=False, verbose=30)
    # print(f'adaptive_krum: {point}, {clients_type}')

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(honest_points[:, 0], honest_points[:, 1], color='blue', label='Honest Points', s=100)
    plt.scatter(byzantine_points[:, 0], byzantine_points[:, 1], color='red', label='Byzantine Point', s=100, marker='x')
    plt.scatter([true_cw_median[0]], [true_cw_median[1]], color='green', label='True MEDIAN (Honest Only)', s=150,
                marker='*')
    plt.scatter([cw_median[0]], [cw_median[1]], color='m', label='CW-MEDIAN (With Byzantine)', s=120,
                marker='P')

    # Labels and legend
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    plt.title("Effect of Byzantine Attack on Coordinate-wise Median")
    plt.legend()
    # plt.grid(True)
    # plt.xlim(0, 2.2)
    # plt.ylim(0, 2.2)
    plt.tight_layout()
    plt.savefig('cw_median_failures.png', dpi=600)

    plt.show()


def trimmed_mean_case():
    """ the case shows that trimmed mean can still be impacted by byzantine clients.
    """
    # n is the total number of points, where up to f are byzantine, honest = n-f = n//2 + 2
    # Krum requires that 2+2f < n (total points), so f < (n-2)/2, so we set f = (n-2)//2 - 1 = n//2 - 2
    # if n = 5, f < (n-2)/2 = 1.5, so f can be 1.
    # if n = 5, f = 1, honest = 4, p = (f/n)/2 = 1/10, trimmed_cnt = max(1, int(p * n)) = max(1, 0.2) = 1.
    # so in our current implementation, we still remove 1 from lower side and 1 from upper side.
    # if n = 10, f = 3, honest = 7. p = (f/n)/2 = 3/20, trimmed_cnt = max(1, int(p * n)) = 3/2 = 1,
    # lower = upper = 1
    # if n = 20, f = 8, honest = 12. p = (f/n)/2 = 8/40, trimmed_cnt = max(1, int(p * n)) = 8/40*20 = 4

    attack_type = "large_value"
    if attack_type == "large_value":
        # n = 10, f = 3, h = 7
        honest_points = np.asarray([(0.5, 0.6), (0.6, 0.5), (0.6, 0.7), (0.7, 0.6),
                                    (0.5, 0.5), (0.52, 0.68), (0.68, 0.52)])
        byzantine_points = np.asarray([(1.5, 1.5), (1.5, 1.8), (1.7, 1.5)])

    elif attack_type == "sign_flipping":
        honest_points = [(0.4, .0), (0.5, 0.6), (0.6, 0.5), (0, 0.4)]
        # honest_points = [(0.4, .0), (0.5, 0.6)]
        byzantine_points = [(v[0] * (-1), v[1] * (-1)) for v in honest_points[:len(honest_points) - 1]]
        # malicious_points = [(1.5, 1.2), (1.2, 1.0), (1.0, 2.0)]
    else:
        raise NotImplementedError

    f = len(byzantine_points)
    # byzantine points must be appended to the end if using the first n-f points to compute true mean
    points = [torch.tensor(p) for p in np.concatenate([honest_points, byzantine_points])]
    points = torch.stack(points)
    print(points)
    n = len(points)
    weights = torch.tensor([1] * n)
    if 2 + 2 * f >= n or f + len(honest_points) != n:
        raise ValueError(f, n)
    print(f'n:{n}, f:{f}, h:{n - f}')
    # True median if only honest points were considered
    true_cw_mean, clients_type = robust_aggregation.cw_mean(points[:n - f], weights[:n - f], verbose=VERBOSE)
    print(f'true_cw_mean: {true_cw_mean}, {clients_type}')

    p = (f / n) / 2
    # p = 0.45
    trimmed_cw_mean, clients_type = robust_aggregation.trimmed_mean(points, weights, trim_ratio=p, verbose=VERBOSE)
    print(f'trimmed_cw_mean: {trimmed_cw_mean}, {clients_type}')
    print()

    # cw_median, clients_type = robust_aggregation.cw_median(points, weights, verbose=VERBOSE)
    # print(f'cw_median: {cw_median}, {clients_type}')
    # print()
    #
    # geo_median, clients_type = robust_aggregation.geometric_median(points, weights, verbose=VERBOSE)
    # print(f'geo_median: {geo_median}, {clients_type}')
    # print()
    # exit(0)
    medoid, clients_type = robust_aggregation.medoid(points, weights, trimmed_average=True, upper_trimmed_ratio=f/n,
                                                     verbose=VERBOSE)
    print(f'medoid: {medoid}, {clients_type}')
    print()

    krum_point, clients_type = robust_aggregation.krum(points, weights, f, trimmed_average=False, verbose=VERBOSE)
    print(f'krum: {krum_point}, {clients_type}')

    # point, clients_type = adaptive_krum(points, weights, trimmed_average=False, verbose=30)
    # print(f'adaptive_krum: {point}, {clients_type}')

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(honest_points[:, 0], honest_points[:, 1], color='blue', label='Honest Points', s=100)
    plt.scatter(byzantine_points[:, 0], byzantine_points[:, 1], color='red', label='Byzantine Point', s=100, marker='x')
    plt.scatter([true_cw_mean[0]], [true_cw_mean[1]], color='green', label='True MEAN (Honest Only)', s=150,
                marker='*')
    plt.scatter([trimmed_cw_mean[0]], [trimmed_cw_mean[1]], color='m', label='TRIMMED-MEAN (With Byzantine)', s=120,
                marker='P')

    # Labels and legend
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    plt.title("Effect of Byzantine Attack on Trimmed (Coordinate-wise) Mean")
    plt.legend()
    # plt.grid(True)
    # plt.xlim(0, 2.2)
    # plt.ylim(0, 2.2)
    plt.tight_layout()
    plt.savefig('trimmed_mean_failures.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    # median_case()
    trimmed_mean_case()
