import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# import projection_median
# from autoencoder_mean import robust_mean_estimation

np.set_printoptions(precision=3, suppress=True)


def iterative_remove_point(X, f):
    n, d = X.shape
    indices = set()

    # Pick a random initial point
    idx = np.random.choice(n, size=1, replace=False)[0]

    for _ in range(n - 1):
        while idx in indices:
            idx = np.random.choice(n, size=1, replace=False)[0]

        # Find the farthest point from the current idx
        dists = np.linalg.norm(X - X[idx], axis=1)
        idx2 = np.argmax(dists)
        print(idx2, dists)

        indices.add(idx)
        idx = idx2  # Move to the next point for removal

    # Compute the estimated mean of remaining points
    mask = np.ones(n, dtype=bool)
    mask[list(indices)] = False  # Mask removed points
    estimated_mean = np.mean(X[mask], axis=0)

    return estimated_mean, [0 if v else 1 for v in mask]


def robust_center_exponential_reweighting(X, x_est, r=1.0, max_iters=100, tol=1e-6, verbose=1):
    """
    Compute the robust center of a set of points using Exponential Reweighting.

    Parameters:
    X : ndarray of shape (n, d)
        The input data points (n points in d-dimensional space).
    r : float, optional
        The sharpness parameter controlling weight decay.
    max_iters : int, optional
        Maximum number of iterations for convergence.
    tol : float, optional
        Convergence tolerance for stopping criterion.

    Returns:
    x_est : ndarray of shape (d,)
        The estimated robust center.
    """
    n, d = X.shape

    n_honest_points_lower_bound = n // 2
    for i in range(max_iters):
        # Compute squared Euclidean distances to current estimate
        distances = np.linalg.norm(X - x_est, axis=1)

        # Compute weights using the exponential reweighting scheme
        # weights = np.exp(-r * distances)
        max_exp_value = 100  # Close to the limit where np.exp() overflows
        weights = np.exp(np.clip(-r * distances, -max_exp_value, max_exp_value))  # clip the weights
        # print('\n', i, weights)
        # Normalize weights to avoid some weight is very large or small
        weights /= np.sum(weights)
        # print(i, weights)
        # Identify the top n/2 highest weights
        sorted_indices = np.argsort(weights)[-n_honest_points_lower_bound:]
        min_weight_value = weights[sorted_indices[0]]  # Smallest weight among the top n/2
        # If min_weight_value is zero or very small, set a small adaptive floor (e.g., 1/n)
        if min_weight_value < 1e-6:
            if verbose >= 10: print('*** ', i, min_weight_value, n_honest_points_lower_bound)
            # we need to make sure at least half of points is used to compute the weighted mean
            min_weight_value = 1 / n  # Ensures at least n/2 weights > 0
            weights[weights < min_weight_value] = min_weight_value
            # Normalize weights to sum to 1
            weights /= np.sum(weights)

        # Update x_est as the weighted mean
        x_new = np.sum(weights[:, None] * X, axis=0)
        if verbose >= 5:
            print(i, weights, np.linalg.norm(x_new - x_est))
        # Check for convergence
        if np.linalg.norm(x_new - x_est) < tol:
            break

        x_est = x_new

    return x_est


def robust_center_sigmoid_reweighting(X, x_est, r=1.0, max_iters=100, tol=1e-6, verbose=1):
    """
    Compute the robust center of a set of points using Exponential Reweighting.

    Parameters:
    X : ndarray of shape (n, d)
        The input data points (n points in d-dimensional space).
    r : float, optional
        The sharpness parameter controlling weight decay.
    max_iters : int, optional
        Maximum number of iterations for convergence.
    tol : float, optional
        Convergence tolerance for stopping criterion.

    Returns:
    x_est : ndarray of shape (d,)
        The estimated robust center.
    """
    n, d = X.shape

    n_honest_points_lower_bound = n // 2
    for i in range(max_iters):
        # Compute squared Euclidean distances to current estimate
        distances = np.linalg.norm(X - x_est, axis=1)

        # Compute weights using the exponential reweighting scheme
        # weights = 1 / (1 + np.exp(-r*distances))
        weights = 1 / (1 + np.exp(r * distances))  # the larger distance, the smaller the weight
        # print('\n', i, weights, np.sum(weights))
        weights /= np.sum(weights)
        # # Identify the top n/2 highest weights
        # sorted_indices = np.argsort(weights)[-n_honest_points_lower_bound:]
        # min_weight_value = weights[sorted_indices[0]]  # Smallest weight among the top n/2
        # # If min_weight_value is zero or very small, set a small adaptive floor (e.g., 1/n)
        # if min_weight_value < 1e-6:
        #     if verbose >= 10: print('*** ', i, min_weight_value, n_honest_points_lower_bound)
        #     # we need to make sure at least half of points is used to compute the weighted mean
        #     min_weight_value = 1 / n  # Ensures at least n/2 weights > 0
        #     weights[weights < min_weight_value] = min_weight_value
        #     # Normalize weights to sum to 1
        #     weights /= np.sum(weights)

        # Update x_est as the weighted mean
        x_new = np.sum(weights[:, None] * X, axis=0)
        if verbose >= 5:
            print(i, weights, np.linalg.norm(x_new - x_est))
        # Check for convergence
        if np.linalg.norm(x_new - x_est) < tol:
            break

        x_est = x_new

    return x_est


import torch


def robust_center_exponential_reweighting_tensor(X, x_est, f=1, r=1.0, max_iters=100, tol=1e-6, verbose=1):
    """
    Compute the robust center of a set of points using Exponential Reweighting.

    Parameters:
    X : torch.Tensor of shape (n, d)
        The input data points (n points in d-dimensional space).
    x_est : torch.Tensor of shape (d,)
        Initial estimate of the robust center.
    r : float, optional
        The sharpness parameter controlling weight decay.
    max_iters : int, optional
        Maximum number of iterations for convergence.
    tol : float, optional
        Convergence tolerance for stopping criterion.

    Returns:
    x_est : torch.Tensor of shape (d,)
        The estimated robust center.
    """
    n, d = X.shape
    n_honest_points_lower_bound = n // 2
    for i in range(max_iters):
        # Compute squared Euclidean distances to current estimate
        distances = torch.norm(X - x_est, dim=1)

        # Compute weights using the exponential reweighting scheme
        weights = torch.exp(-r * distances)
        weights /= torch.sum(weights)  # Normalize weights to sum to 1

        # Identify the top n/2 highest weights
        sorted_indices = torch.argsort(weights)[-n_honest_points_lower_bound:]
        min_weight_value = weights[sorted_indices[0]]  # Smallest weight among the top n/2

        # If min_weight_value is zero or very small, set a small adaptive floor (e.g., 1/n)
        if min_weight_value < 1e-6:
            if verbose >= 20:
                print('*** ', i, min_weight_value.item(), n_honest_points_lower_bound)
            min_weight_value = 1.0 / n  # Ensures at least n/2 weights > 0
            weights[weights < min_weight_value] = min_weight_value
        # weights[sorted_indices[:f]] = 0.0
        weights /= torch.sum(weights)  # Normalize weights to sum to 1

        # Update x_est as the weighted mean
        x_new = torch.sum(weights[:, None] * X, dim=0)
        if verbose >= 5:
            print(i, weights, torch.norm(x_new - x_est).item())

        # Check for convergence
        if torch.norm(x_new - x_est) < tol:
            break

        x_est = x_new

    return x_est


def dist(true_center, mean_point):
    return np.linalg.norm(true_center - mean_point)


def distance_vs_weight():
    import numpy as np
    import matplotlib.pyplot as plt

    # Define distance range
    distances = np.linspace(0.5, 10, 100)
    print(distances)

    # Define different weighting functions
    def inverse_distance(d):
        return 1 / (d + 1e-6)  # Avoid division by zero

    def exponential_decay(d, r=0.5):
        return np.exp(-r * d)

    def gaussian_weight(d, sigma=2):
        return np.exp(-d ** 2 / (2 * sigma ** 2))

    def sigmoid_weight(d, a=1, b=5):
        return 1 / (1 + np.exp(a * (d - b)))

    def quadratic_weight(d):
        return 1 / (1 + d ** 2)

    def softmin_weight(d, alpha=1):
        return np.exp(-alpha * d) / np.sum(np.exp(-alpha * distances))

    def truncated_weight(d, tau=5):
        return np.where(d <= tau, 1, 0)

    # Compute weights for each method
    weights = {
        "Inverse Distance": inverse_distance(distances),
        "Exponential Decay": exponential_decay(distances),
        "Gaussian": gaussian_weight(distances),
        "Sigmoid": sigmoid_weight(distances),
        "Quadratic": quadratic_weight(distances),
        "Soft-Min": softmin_weight(distances),
        "Truncated": truncated_weight(distances)
    }

    # Plot the weight functions
    plt.figure(figsize=(8, 6))
    for name, w in weights.items():
        print(name, w)
        plt.plot(distances, w, label=name)

    # plt.ylim(0, 1.1)
    plt.xlabel("Distance")
    plt.ylabel("Weight")
    plt.title("Comparison of Weighting Functions")
    plt.legend()
    plt.grid()
    plt.show()


# Example usage:
if __name__ == "__main__":

    # distance_vs_weight()
    # exit()

    np.random.seed(42)
    n = 2000
    f = (n - 3) // 2  # 2 + 2f < n, so f < n/2 - 1
    # f = n // 2 - 1

    # n2 = n//2
    # f = n2//2
    case = 'krum_failure0'
    if case == 'krum_failure':
        true_center = np.asarray([3, 0])  # np.mean(honest_points, axis=0)
        print(true_center, n, f)
        # honest_points = np.random.normal(0, 1, size=(n - f, 2))  # 50 honest points (Gaussian)
        honest_points = np.random.multivariate_normal(true_center, cov=[[2, 0], [0, 1]],
                                                      size=n - f)  # 50 honest points (Gaussian)
        # byzantine_points = np.random.uniform(5, 20, size=(10, 2))  # 10 Byzantine points (outliers)
        # byzantine_points = np.random.normal(10, 1, size=(f, 2))  # 10 Byzantine points (outliers)
        byzantine_points = np.random.multivariate_normal([40, 50], cov=[[0.1, 0], [0, 0.1]],
                                                         size=f)  # 10 Byzantine points (outliers)
    else:
        d = 2
        true_center = np.asarray([30] * d)  # np.mean(honest_points, axis=0)
        # true_center[0] = 3
        print(true_center, n, f)
        # honest_points = np.random.normal(0, 1, size=(n - f, 2))  # 50 honest points (Gaussian)
        honest_points = np.random.multivariate_normal(true_center, cov=np.ones((d, d)) * 1,
                                                      size=n - f)  # 50 honest points (Gaussian)
        # byzantine_points = np.random.uniform(5, 20, size=(10, 2))  # 10 Byzantine points (outliers)
        # byzantine_points = np.random.normal(10, 1, size=(f, 2))  # 10 Byzantine points (outliers)
        byzantine_center = np.asarray([0] * d)  # np.mean(honest_points, axis=0)
        # byzantine_center[0] = 30
        byzantine_points = np.random.multivariate_normal(byzantine_center, cov=np.ones((d, d)) * 0.5,
                                                         size=f)  # 10 Byzantine points (outliers)


    # # Define the rotation angle in degrees
    # theta = np.radians(71)  # Convert degrees to radians
    #
    # # Define the 2D rotation matrix
    # R = np.array([[np.cos(theta), -np.sin(theta)],
    #               [np.sin(theta), np.cos(theta)]])
    #
    # # Assume honest_points is a NumPy array of shape (N, 2)
    # # Rotate each point
    # honest_points = honest_points @ R.T  # Apply the rotation matrix

    X = np.vstack([honest_points, byzantine_points])

    points = X

    # Initialize x_est as the coordinate-wise median (a robust starting point)
    # x_est = np.median(X, axis=0)  # you can use any initial value for x_est
    # median_point = np.median(points, axis=0)  # average of the middle two values when n = even
    sorted_points = np.sort(points, axis=0)
    x_est = sorted_points[(len(points) - 1) // 2]  # Select the lower middle value
    # x_est = (5, 6)
    exp_estimated_center = robust_center_exponential_reweighting(X, x_est, r=0.1, verbose=10)
    print("Estimated Robust Exp Center:", exp_estimated_center)

    sigmoid_estimated_center = robust_center_sigmoid_reweighting(X, x_est, r=0.1, verbose=10)
    print("Estimated Robust Sigmoid Center:", sigmoid_estimated_center)

    circle_points = honest_points
    outlier = byzantine_points
    # Compute Mean (average of all points)
    mean_point = np.mean(points, axis=0)

    # Compute Median (coordinate-wise median)
    # median_point = np.median(points, axis=0)  # average of the middle two values when n = even
    sorted_points = np.sort(points, axis=0)
    median_point = sorted_points[(len(points) - 1) // 2]  # Select the lower middle value

    dists = np.linalg.norm(points - median_point, axis=1)
    sorted_indices = np.argsort(dists)
    mean_of_median = np.mean(points[sorted_indices[:n - f]], axis=0)

    # Compute Medoid (point closest to all others in terms of total distance)
    dist_matrix = cdist(points, points, metric='euclidean')
    total_distances = np.sum(dist_matrix, axis=1)
    medoid_index = np.argmin(total_distances)
    medoid_point = points[medoid_index]

    # Compute Krum (Selects a point closest to its nearest neighbors, ignoring the farthest)
    k = len(points) - len(outlier) - 2  # k = total_n -f-2  = 19 - 9 - 2 = 8
    print(f'k:{k}, n:{n}, f:{f}')
    krum_scores = np.sum(np.sort(dist_matrix, axis=1)[:, 1:k + 1], axis=1)  # Ignore self-distance (0)
    print(krum_scores)
    krum_index = np.argmin(krum_scores)
    krum_point = points[krum_index]


    # Compute Adaptive Weighted Mean
    def adaptive_weighted_mean(points, beta=1):
        # center_guess = np.median(points, axis=0)  # Initial guess
        sorted_points = np.sort(points, axis=0)
        center_guess = sorted_points[(len(points) - 1) // 2]  # Select the lower middle value

        distances = np.linalg.norm(points - center_guess, axis=1)
        weights = np.exp(-beta * distances)  # Exponential decay for far points
        weights /= np.sum(weights)
        print(weights)
        return np.sum(points * weights[:, np.newaxis], axis=0)


    adaptive_mean = adaptive_weighted_mean(points)


    # Compute Geometric Median using Weiszfeld’s algorithm
    def geometric_median(X, eps=1e-10):
        # y = np.mean(X, axis=0)  # Initial guess
        sorted_points = np.sort(points, axis=0)
        y = sorted_points[(len(points) - 1) // 2]  # Select the lower middle value

        while True:
            distances = np.linalg.norm(X - y, axis=1)
            nonzero_distances = np.where(distances > eps, distances, eps)  # Avoid division by zero
            weights = 1 / nonzero_distances
            new_y = np.average(X, axis=0, weights=weights)
            if np.linalg.norm(y - new_y) < eps:
                return new_y
            y = new_y


    geo_median = geometric_median(points)

    import cvxpy as cp


    def geometric_median_lp(points):
        n, d = points.shape  # n: number of points, d: dimension
        y = cp.Variable(d)  # The geometric median (a d-dimensional variable)
        t = cp.Variable(n)  # Auxiliary variables for each distance

        # Constraints: ||y - x_i||_2 ≤ t_i for all i
        constraints = [cp.norm(y - points[i], 2) <= t[i] for i in range(n)]

        # Objective: Minimize the sum of all t_i
        objective = cp.Minimize(cp.sum(t))

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return y.value  # The optimal y is the geometric median


    lp_median = geometric_median_lp(points)
    print("Geometric Liner Programming Median:", lp_median)

    # Compute robust mean
    robust_mu = robust_mean_estimation(points, f)

    iterative_mean, predicted_indices = iterative_remove_point(points, f)
    print(iterative_mean, predicted_indices)

    n, d = points.shape
    # num_projections = int(np.log2(max(n, d)))
    num_projections = d
    print(f'num_projections: {num_projections}')
    proj_median = projection_median.projection_median(torch.tensor(points, dtype=float),
                                                      num_projections=num_projections).numpy()

    # Print results
    print('true center:', true_center)
    for est_name, est in [('mean', mean_point), ('median', median_point), ('mean_of_median', mean_of_median),
                          ('medoid', medoid_point),
                          ('krum', krum_point), ('geo_median', geo_median), ('adaptive_mean', adaptive_mean),
                          ('exp_weighted_mean', exp_estimated_center),
                          ('sigmoid_weighted_mean', sigmoid_estimated_center),
                          ('lp_mean', lp_median),
                          ('robust_mean', robust_mu),
                          ('iterative_mean', iterative_mean),
                          ('proj_median', proj_median)
                          ]:
        print(f'{est_name}: {est[:5]}, ||x-x*||: {dist(true_center, est):.2f}')
    # print(mean_point, median_point, medoid_point, krum_point, geo_median, adaptive_mean, estimated_center)

    # Plot results
    plt.figure(figsize=(6, 6))
    plt.scatter(circle_points[:, 0], circle_points[:, 1], label="Circle Points", color="blue", alpha=0.7)
    plt.scatter(outlier[:, 0], outlier[:, 1], label="Outlier (100,100)", color="red", marker="x", s=150)
    plt.scatter(*mean_point, color="purple", label="Mean", marker="o", s=100)
    plt.scatter(*median_point, color="green", label="Median", marker="s", s=100)
    plt.scatter(*medoid_point, color="orange", label="Medoid", marker="D", s=100)
    plt.scatter(*krum_point, color="brown", label="Krum", marker="P", s=100)
    plt.scatter(*geo_median, color="black", label="Geo_median", marker="o", s=100)
    plt.scatter(*adaptive_mean, color="yellow", label="adaptive_mean", marker="s", s=100)
    plt.scatter(*exp_estimated_center, color="green", label="weighted_exp_mean", marker="x", s=100)
    plt.scatter(*sigmoid_estimated_center, color="blue", label="weighted_sigmoid_mean", marker="+", s=100)
    plt.scatter(*proj_median, color="blue", label="projection_median", marker="o", s=100)

    # plt.xlim(-2, 105)
    # plt.ylim(-2, 105)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Comparison of Aggregation Methods")
    plt.grid()
    plt.show()
    plt.close()
