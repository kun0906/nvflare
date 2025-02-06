import torch
from krum import krum, median


def case():
    """ the case shows that Krum is better than Median

        honest point 1 = (0.1, 0.1)
        honest point 2 = (0.15, 0.15)
        honest point 3 = (0.2, 0.2)
        byzantine point 1 = (10, 10)
        byzantine point 2 = (20, 20)

        Median is (0.2, 0.2) as Median won’t ignore the effect of byzantine points. If honest points has outlier,
                median will choose the outlier point
        Krum is (0.15, 0.15): Krum will ignore the byzantine points first, then find the smallest point as the
                central point.

    Returns:

    """

    honest_points = [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2)]
    byzantine_points = [(10, 10), (20, 20)]
    f = len(byzantine_points)

    points = [torch.tensor(p) for p in byzantine_points + honest_points]
    n = len(points)
    weights = torch.tensor([1] * n)

    point, clients_type = median(points, weights)
    print(f'median: {point}, {clients_type}')
    print()

    point, clients_type = krum(points, weights, f, return_average=False)
    print(f'krum: {point}, {clients_type}')


if __name__ == '__main__':
    case()
