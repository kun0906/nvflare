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
import time
import torch
import numpy as np
import robust_aggregation
import matplotlib.pyplot as plt

VERBOSE = 1
SEED = 42

# Generate 15 colors from the tab20 colormap
colors = plt.get_cmap("tab20").colors[:15]
makers = ['o', 's', '*', 'v', 'p', 'h', 'x', '8', '1', '^', 'D', 'd', '+', '.']


def median_case():
    """ the case shows that Krum is better than coordinate wise Median (order median)
    """
    # N is the total number of points, where up to f are byzantine, honest = N-f = N//2 + 2
    # Krum requires that 2+2f < N (total points), so f < (N-2)/2, so we set f = (N-2)//2 - 1 = N//2 - 2
    # if N = 5, f < (N-2)/2 = 1.5, so f can be 1.  e.g., N = 5, f = 1, honest = 4

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
    # byzantine points must be appended to the end if using the first N-f points to compute true mean
    points = [torch.tensor(p) for p in np.concatenate([byzantine_points, honest_points])]
    points = torch.stack(points)
    print(points)
    N = len(points)
    weights = torch.tensor([1] * N)
    if 2 + 2 * f >= N or f + len(honest_points) != N:
        raise ValueError(f, N)
    print(f'N:{N}, f:{f}, h:{N - f}')
    # True median if only honest points were considered
    true_cw_median, clients_type = robust_aggregation.cw_median(points[:N - f], weights[:N - f])
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
    # N is the total number of points, where up to f are byzantine, honest = N-f = N//2 + 2
    # Krum requires that 2+2f < N (total points), so f < (N-2)/2, so we set f = (N-2)//2 - 1 = N//2 - 2
    # if N = 5, f < (N-2)/2 = 1.5, so f can be 1.
    # if N = 5, f = 1, honest = 4, p = (f/N)/2 = 1/10, trimmed_cnt = max(1, int(p * N)) = max(1, 0.2) = 1.
    # so in our current implementation, we still remove 1 from lower side and 1 from upper side.
    # if N = 10, f = 3, honest = 7. p = (f/N)/2 = 3/20, trimmed_cnt = max(1, int(p * N)) = 3/2 = 1,
    # lower = upper = 1
    # if N = 20, f = 8, honest = 12. p = (f/N)/2 = 8/40, trimmed_cnt = max(1, int(p * N)) = 8/40*20 = 4

    attack_type = "large_value"
    if attack_type == "large_value":
        # N = 10, f = 3, h = 7
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
    # byzantine points must be appended to the end if using the first N-f points to compute true mean
    points = [torch.tensor(p) for p in np.concatenate([honest_points, byzantine_points])]
    points = torch.stack(points)
    print(points)
    N = len(points)
    weights = torch.tensor([1] * N)
    if 2 + 2 * f >= N or f + len(honest_points) != N:
        raise ValueError(f, N)
    print(f'N:{N}, f:{f}, h:{N - f}')
    # True median if only honest points were considered
    true_cw_mean, clients_type = robust_aggregation.cw_mean(points[:N - f], weights[:N - f], verbose=VERBOSE)
    print(f'true_cw_mean: {true_cw_mean}, {clients_type}')

    p = (f / N) / 2
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
    medoid, clients_type = robust_aggregation.medoid(points, weights, trimmed_average=True, upper_trimmed_ratio=f / N,
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


def synthetic_single_case(N=100, D=500, byzantine_mu=1, f=None, show=False, random_state=42):
    """
    """
    # random_state=200
    attack_type = "large_value"
    if attack_type == "large_value":
        # N = 10, f = 3, h = 7, 2 + 2*f < N
        # honest_points = np.asarray([(0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.5, 0.6),
        #                             (0.9, 0.9), (1.0, 1.0), (1.1, 1.1),
        #                             ])
        # byzantine_points = np.asarray([(1.5, 1.5), (1.5, 1.8), (1.7, 1.5)])

        # f = (N - 2) // 2
        # if 2 + 2 * f >= N:
        #     f = f - 1
        # h = N - f
        if f is None:
            f = (N - 3) // 2

        h = N - f  # the number of honest points
        if 2 + 2 * f >= N or f + h != N:
            raise ValueError(f, N)
        print(f'N: {N}, f: {f}, h: {h}, {byzantine_mu}, seed: {random_state}')
        np.random.seed(random_state)
        # honest_points = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=h)
        # byzantine_points = np.random.multivariate_normal(mean=[byzantine_mu, byzantine_mu],
        #                                                  cov=[[1, 0], [0, 1]], size=f)
        mean = np.zeros(D)  # d-dimensional mean vector (all zeros)
        cov = np.eye(D)  # d x d identity matrix as covariance (independent variables)
        honest_points = np.random.multivariate_normal(mean=mean, cov=cov, size=h)

        mean = np.ones(D) * byzantine_mu  # d-dimensional mean vector (all zeros)
        # mean[1:] = 0
        cov = np.eye(D) * 2  # d x d identity matrix as covariance (independent variables)
        byzantine_points = np.random.multivariate_normal(mean, cov=cov, size=f)

        # mean = np.ones(D) * byzantine_mu*10  # d-dimensional mean vector (all zeros)
        # cov = np.eye(D) * 2  # d x d identity matrix as covariance (independent variables)
        # byzantine_points2 = np.random.multivariate_normal(mean, cov=cov, size=f//2)
        # byzantine_points = np.concatenate([byzantine_points, byzantine_points2], axis=0)

    elif attack_type == "sign_flipping":
        honest_points = [(0.4, .0), (0.5, 0.6), (0.6, 0.5), (0, 0.4)]
        # honest_points = [(0.4, .0), (0.5, 0.6)]
        byzantine_points = [(v[0] * (-1), v[1] * (-1)) for v in honest_points[:len(honest_points) - 1]]
        # malicious_points = [(1.5, 1.2), (1.2, 1.0), (1.0, 2.0)]
    else:
        raise NotImplementedError

    f = len(byzantine_points)
    # byzantine points must be appended to the end if using the first N-f points to compute true mean
    points = [torch.tensor(p, dtype=torch.float) for p in np.concatenate([honest_points, byzantine_points])]
    points = torch.stack(points)
    if len(points) != N:
        raise ValueError(len(points), N)
    weights = torch.tensor([1] * N)
    if 2 + 2 * f >= N or f + len(honest_points) != N:
        raise ValueError(f, N)
    # print(f"points: {points}, {weights}")
    print(f'N: {N}, f: {f}, h: {h}, D: {D}, byzantine_mu:{byzantine_mu}, seed: {random_state}')

    if show:
        # Plotting
        plt.figure(figsize=(6, 6))
        plt.scatter(honest_points[:, 0], honest_points[:, 1], color='blue', label='Honest Updates', s=100)
        plt.scatter(byzantine_points[:, 0], byzantine_points[:, 1], color='red', label='Byzantine Updates', s=100,
                    marker='x')
        # Labels and legend
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        plt.title("Honest and Byzantine Updates in $\mathbb{R}^2$")
        plt.legend()
        # plt.grid(True)
        # plt.xlim(0, 2.2)
        # plt.ylim(0, 2.2)
        plt.tight_layout()
        plt.savefig('synthetic_data.png', dpi=300)

        plt.show()

    # True median if only honest points were considered
    empirical_cw_mean, clients_type = robust_aggregation.cw_mean(points[:N - f], weights[:N - f], verbose=VERBOSE)
    method = 'empirical_cw_mean'
    SPACES = 16
    print(
        f'{method:{SPACES}s}: {[float(f"{v:.3f}") for v in empirical_cw_mean.tolist()]}, clients_type: {clients_type.tolist()}')

    results = {}

    aggregation_methods = (
        # Single value
        # 'adaptive_krum', 'krum', 'adaptive_krum+rp', 'krum+rp', 'medoid', 'median', 'mean',
        'adaptive_krum', 'krum', 'medoid',
        'median',
        'mean',
        # Merged value
        # 'adaptive_krum_avg', 'krum_avg', 'adaptive_krum+rp_avg', 'krum+rp_avg', 'medoid_avg',
        # 'adaptive_krum_avg', 'krum_avg', 'medoid_avg',
        'trimmed_mean', 'geometric_median'
    )
    for method in aggregation_methods:
        start = time.time()
        ##################################### Single value ################################################
        if method == 'adaptive_krum':
            agg_value, clients_type = robust_aggregation.adaptive_krum(points, weights,
                                                                       trimmed_average=False,
                                                                       random_projection=False, k_factor=1,
                                                                       random_state=SEED,
                                                                       verbose=VERBOSE)
        elif method == 'krum':
            agg_value, clients_type = robust_aggregation.krum(points, weights, f=f,
                                                              trimmed_average=False,
                                                              random_projection=False, k_factor=1,
                                                              random_state=SEED,
                                                              verbose=VERBOSE)
        elif method == 'adaptive_krum+rp':
            agg_value, clients_type = robust_aggregation.adaptive_krum(points, weights,
                                                                       trimmed_average=False,
                                                                       random_projection=True,
                                                                       k_factor=1,
                                                                       random_state=SEED,
                                                                       verbose=VERBOSE)
        elif method == 'krum+rp':
            agg_value, clients_type = robust_aggregation.krum(points, weights,
                                                              f=f,
                                                              trimmed_average=False,
                                                              random_projection=True,
                                                              k_factor=1,
                                                              random_state=SEED,
                                                              verbose=VERBOSE)
        elif method == 'medoid':
            p = (f / N)
            agg_value, clients_type = robust_aggregation.medoid(points, weights, trimmed_average=False,
                                                                upper_trimmed_ratio=p, verbose=VERBOSE)
        elif method == 'median':
            agg_value, clients_type = robust_aggregation.cw_median(points, weights, verbose=VERBOSE)
        elif method == 'mean':
            agg_value, clients_type = robust_aggregation.cw_mean(points, weights, verbose=VERBOSE)
        ##################################### merged value ################################################
        elif method == 'adaptive_krum_avg':
            agg_value, clients_type = robust_aggregation.adaptive_krum(points, weights, trimmed_average=True,
                                                                       random_projection=False, k_factor=1,
                                                                       random_state=SEED,
                                                                       verbose=VERBOSE)
        elif method == 'krum_avg':
            agg_value, clients_type = robust_aggregation.krum(points, weights, f=f,
                                                              trimmed_average=True,
                                                              random_projection=False, k_factor=1,
                                                              random_state=SEED,
                                                              verbose=VERBOSE)
        elif method == 'adaptive_krum+rp_avg':
            agg_value, clients_type = robust_aggregation.adaptive_krum(points, weights,
                                                                       trimmed_average=True,
                                                                       random_projection=True,
                                                                       k_factor=1,
                                                                       random_state=SEED,
                                                                       verbose=VERBOSE)
        elif method == 'krum+rp_avg':
            agg_value, clients_type = robust_aggregation.krum(points, weights, f,
                                                              trimmed_average=True,
                                                              random_projection=True,
                                                              k_factor=1,
                                                              random_state=SEED,
                                                              verbose=VERBOSE)
        elif method == 'medoid_avg':
            p = (f / N)
            agg_value, clients_type = robust_aggregation.medoid(points, weights, trimmed_average=True,
                                                                upper_trimmed_ratio=p, verbose=VERBOSE)
        elif method == 'trimmed_mean':
            p = (f / N) / 2
            # p = 0.45
            agg_value, clients_type = robust_aggregation.trimmed_mean(points, weights, trim_ratio=p, verbose=VERBOSE)
        elif method == 'geometric_median':
            agg_value, clients_type = robust_aggregation.geometric_median(points, weights, verbose=VERBOSE)
        else:
            raise NotImplementedError(method)

        end = time.time()
        time_taken = end - start
        # l2_error = torch.norm(agg_value - 0).item()
        l2_error = torch.norm(agg_value - empirical_cw_mean).item()
        results[method] = (agg_value, l2_error, time_taken, clients_type)
        print(f'{method:{SPACES}s}: {[float(f"{v:.3f}") for v in agg_value.tolist()]}, '
              f'l2: {l2_error:.5f}, time: {time_taken:.5f}, ')
        # f'clients_type: {clients_type.tolist()}')

    if show:
        # Plotting
        plt.figure(figsize=(6, 6))
        plt.scatter(honest_points[:, 0], honest_points[:, 1], color='blue', label='Honest Points', s=100)
        plt.scatter(byzantine_points[:, 0], byzantine_points[:, 1], color='red', label='Byzantine Point', s=100,
                    marker='x')
        plt.scatter([empirical_cw_mean[0]], [empirical_cw_mean[1]], color='green', label='True MEAN (Honest Only)',
                    s=150,
                    marker='*')

        for i, (method, (agg_value, l2_error, time_taken, clients_type)) in enumerate(results.items()):
            plt.scatter([agg_value[0]], [agg_value[1]], color=colors[i], label=f'{method}', s=120,
                        marker=makers[i])

        # Labels and legend
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        plt.title(f"Effect of Byzantine Attack, byzantine_location={byzantine_mu}")
        plt.legend()
        # plt.grid(True)
        # plt.xlim(0, 2.2)
        # plt.ylim(0, 2.2)
        plt.tight_layout()
        plt.savefig('synthetic_case.png', dpi=600)

        plt.show()

    return results


def synthetic_case(tunable_values, case='byzantine_location', N=None, D=None, byzantine_mu=None, f=None):
    results = {}
    for tunable_value in tunable_values:
        if case == 'byzantine_location':
            byzantine_mu = tunable_value
        elif case == 'N':
            N = tunable_value
        elif case == 'D':
            D = tunable_value
        elif case == 'f':
            f = tunable_value
        else:
            raise NotImplementedError(case)

        # get result with different seeds
        tmp_res = []
        num_repeats = 10
        print(f"\n{case}, N:{N}, D:{D}, byzantine_mu:{byzantine_mu}, f:{f}, num_repeats:{num_repeats}")
        for seed in range(0, 1000, 1000 // num_repeats):
            res = synthetic_single_case(N, D, byzantine_mu, f, random_state=seed)
            tmp_res.append(res)
        # compute avg_res for l2_error
        avg_res = {}
        res = tmp_res[0]
        for agg_method in res.keys():
            l2_error_list = []
            time_taken_list = []
            for res_ in tmp_res:
                agg_value, l2_error, time_taken, client_type = res_[agg_method]
                l2_error_list.append(l2_error)
                time_taken_list.append(time_taken)
            avg_res[agg_method] = {'l2_error': (np.mean(l2_error_list), np.std(l2_error_list)),
                                   'time_taken': (np.mean(time_taken_list), np.std(time_taken_list))
                                   }
        results[tunable_value] = avg_res

    METRICS = ['l2_error', ]  # 'time_taken'
    nrows = 1
    ncols = len(METRICS)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=None, figsize=(5 * ncols, 4))  # width, height
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])  # Convert to 2D array shape (1,1)
    axes = axes.reshape((1, -1))
    i_row = 0
    agg_methods = results[tunable_values[0]].keys()
    for j_col, METRIC in enumerate(METRICS):
        for i, agg_method in enumerate(agg_methods):
            vs = []
            for mu, avg_res in results.items():
                mu_, std_ = avg_res[agg_method][METRIC]
                vs.append((mu_, std_))

            ys, ys_errs = zip(*vs)
            xs = range(len(ys))
            print(METRIC, agg_method, vs, flush=True)
            # print(agg_method, txt_file, [f"{v[0]:.2f}/{v[1]:.2f}" for v in vs], flush=True)

            # axes[i_row, j_col].plot(xs, ys, label=label, marker=makers[i])
            # axes[i_row, j_col].errorbar(xs, ys, yerr=ys_errs, label=label, marker=makers[i], capsize=3)
            axes[i_row, j_col].plot(xs, ys, label=agg_method, color=colors[i], marker=makers[i])  # Plot the line
            axes[i_row, j_col].fill_between(xs,
                                            [y - e for y, e in zip(ys, ys_errs)],
                                            [y + e for y, e in zip(ys, ys_errs)],
                                            color=colors[i], alpha=0.3)  # label='Error Area'
            axes[i_row, j_col].set_xticks(xs)
            axes[i_row, j_col].set_xticklabels(tunable_values)
            if case == 'byzantine_location':
                xlabel = '$||\mu_{byzantine} - \mu_{honest}||_2$'
                title = f'n:{N}, d:{D}, byzantine_location:, f:{f}'
            elif case == 'N':
                xlabel = 'n'
                title = f'n:, d:{D}, byzantine_location:{byzantine_mu}, f:{f}'
            elif case == 'D':
                xlabel = 'd'
                title = f'n:{N}, d:, byzantine_location:{byzantine_mu}, f:{f}'
            elif case == 'f':
                xlabel = 'f'
                title = f'n:{N}, d:{D}, byzantine_location:{byzantine_mu}, f:'
            else:
                raise NotImplementedError(case)

            axes[i_row, j_col].set_xlabel(xlabel)
            # which is same to Byzantine $\mu$ Locations, as \mu_{honest} is 0.
            if METRIC == 'loss':
                ylabel = 'Loss'
            elif METRIC == 'l2_error':
                ylabel = '$||\hat{\mu} - \\bar{\mu}||_2$'
            elif METRIC == 'time_taken':
                ylabel = 'Time Taken'
            # elif METRIC == 'misclassified_error':
            #     ylabel = 'Misclassified Error'
            else:
                ylabel = 'Accuracy'
            axes[i_row, j_col].set_ylabel(ylabel)

            # variable = str(namespace_params['labeling_rate'])
            # axes[i_row, j_col].set_title(f'start:{start}, {variable}', fontsize=10)
            axes[i_row, j_col].legend(fontsize=6.5, loc='best')  # loc='upper right'

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')
    # plt.suptitle(f'Global Model ({JOBID}), start:{start}, {title}, {METRIC}', fontsize=10)
    plt.suptitle(f'{title}', fontsize=10)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = f'synthetic_case_{case}.png'
    print(fig_file)
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600)
    plt.show()
    plt.close()


def _krum_rp_case(n, dim, k_factor, num_repetitions=100):
    verbose = 0
    time_taken_list = []
    for i in range(num_repetitions):
        random_state = i * 100
        print(f'\nthe {i}th trial: ')
        # Example updates from clients
        # clients_updates = [
        #     np.random.randn(dim),  # Update from client 1
        #     np.random.randn(dim),  # Update from client 2
        #     np.random.randn(dim),  # Update from client 3
        #     np.random.randn(dim),  # Update from client 4
        #     np.random.randn(dim) + 10,  # Malicious update
        # ]
        # if number of clients is too small, with Random Projection will take more time.
        # Number of Byzantine clients to tolerate
        N = n
        D = dim
        f = (N - 3) // 2
        byzantine_mu = 10
        h = N - f  # the number of honest points
        if 2 + 2 * f >= N or f + h != N:
            raise ValueError(f, N)
        print(f'N: {N}, f: {f}, h: {h}, byzantine_mu:{byzantine_mu}, seed: {random_state}')
        np.random.seed(random_state)
        # honest_points = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=h)
        # byzantine_points = np.random.multivariate_normal(mean=[byzantine_mu, byzantine_mu],
        #                                                  cov=[[1, 0], [0, 1]], size=f)
        mean = np.zeros(D)  # d-dimensional mean vector (all zeros)
        cov = np.eye(D)  # d x d identity matrix as covariance (independent variables)
        honest_points = np.random.multivariate_normal(mean=mean, cov=cov, size=h)

        mean = np.ones(D) * byzantine_mu  # d-dimensional mean vector (all zeros)
        # mean[1:] = 0
        cov = np.eye(D) * 2  # d x d identity matrix as covariance (independent variables)
        byzantine_points = np.random.multivariate_normal(mean, cov=cov, size=f)

        # mean = np.ones(D) * byzantine_mu*10  # d-dimensional mean vector (all zeros)
        # cov = np.eye(D) * 2  # d x d identity matrix as covariance (independent variables)
        # byzantine_points2 = np.random.multivariate_normal(mean, cov=cov, size=f//2)
        # byzantine_points = np.concatenate([byzantine_points, byzantine_points2], axis=0)

        f = len(byzantine_points)
        # byzantine points must be appended to the end if using the first N-f points to compute true mean
        points = [torch.tensor(p, dtype=torch.float) for p in np.concatenate([honest_points, byzantine_points])]
        points = torch.stack(points)
        if len(points) != N:
            raise ValueError(len(points), N)
        weights = torch.tensor([1] * N)
        if 2 + 2 * f >= N or f + len(honest_points) != N:
            raise ValueError(f, N)
        # print(f"points: {points}, {weights}")
        print(f'N: {N}, f: {f}, h: {h}, D: {D}, byzantine_mu:{byzantine_mu}, seed: {random_state}')

        # True median if only honest points were considered
        empirical_cw_mean, clients_type = robust_aggregation.cw_mean(points[:N - f], weights[:N - f], verbose=VERBOSE)
        method = 'empirical_cw_mean'
        SPACES = 16
        print(
            f'{method:{SPACES}s}: {[float(f"{v:.3f}") for v in empirical_cw_mean.tolist()]}, clients_type: {clients_type.tolist()}')

        clients_updates = points
        trimmed_average = False
        # print('Krum...')
        print('\nadaptive Krum...')
        start = time.time()
        aggregated_update, _ = robust_aggregation.adaptive_krum(clients_updates, weights, trimmed_average,
                                                             random_projection=False, k_factor=k_factor,
                                                             verbose=verbose)
        end = time.time()
        time_taken = end - start
        l2_error = torch.norm(aggregated_update - empirical_cw_mean).item()
        print("Aggregated Update (adaptive Krum):", aggregated_update, time_taken)

        print('\nadaptive Krum with Random Projection...')
        start = time.time()
        aggregated_update2, _ = robust_aggregation.adaptive_krum(clients_updates, weights, trimmed_average,
                                                              random_projection=True, k_factor=k_factor,
                                                              verbose=verbose)
        end = time.time()
        time_taken2 = end - start
        l2_error_rp = torch.norm(aggregated_update2 - empirical_cw_mean).item()
        print("Aggregated Update (adaptive Krum) with RP:", aggregated_update2, time_taken2)

        time_taken_list.append([time_taken, time_taken2, l2_error, l2_error_rp])
        # if np.sum(aggregated_update2.numpy() - aggregated_update.numpy()) != 0:
        #     print("Different updates were aggregated")
        #     results.append(clients_updates)
        # break

    return time_taken_list


def krum_rp_case(tunable_values, case='dim', dim=1000, n=100, k_factor=10, num_repetitions=100):
    results = {}
    for tunable_value in tunable_values:
        if case == 'dim':
            dim = tunable_value
        elif case == 'k_factor':
            k_factor = tunable_value
        else:
            raise NotImplementedError(case)
        # print(f'\naccuracy: {1 - len(results) / num_repetitions}')
        time_taken_list = _krum_rp_case(n, dim, k_factor, num_repetitions)
        if case == 'dim':
            results[tunable_value] = {
                'aKrum': (np.mean([v[0] for v in time_taken_list]), np.std([v[0] for v in time_taken_list])),
                'aKrum_rp': (np.mean([v[1] for v in time_taken_list]), np.std([v[1] for v in time_taken_list]))}
        elif case == 'k_factor':
            results[tunable_value] = {
                'aKrum': (np.mean([v[2] for v in time_taken_list]), np.std([v[2] for v in time_taken_list])),
                'aKrum_rp': (np.mean([v[3] for v in time_taken_list]), np.std([v[3] for v in time_taken_list]))}
        else:
            raise NotImplementedError(case)

    colors = ['b', 'g']
    markers = ['*', 'o']
    # Create a figure and axis
    fig, ax = plt.subplots()
    for i, method in enumerate(['aKrum', 'aKrum_rp']):
        xs = range(len(tunable_values))
        ys, ys_errs = [results[t][method][0] for t in tunable_values], [results[t][method][1] for t in tunable_values]
        ax.plot(xs, ys, label=method, color=colors[i], marker=markers[i])
        ax.fill_between(xs,
                         [y - e for y, e in zip(ys, ys_errs)],
                         [y + e for y, e in zip(ys, ys_errs)],
                         color=colors[i], alpha=0.3)  # label='Error Area'
    if case == 'dim':
        ax.set_ylabel('Time Taken in Seconds')
        ax.set_xlabel(f'd')
        title = f'n:{n}, k = {k_factor}*logd'
    elif case == 'k_factor':
        ax.set_ylabel('Estimated Error: $||\hat{\mu} - \\bar{\mu}||_2$')
        ax.set_xlabel(f'k_factor')
        title = f'n:{n}, d:{dim}, k: k_factor*log{dim}'
    else:
        raise NotImplementedError(case)
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(tunable_values)))  # Numeric positions
    ax.set_xticklabels(tunable_values)  # Custom labels

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = f'synthetic_case_rp_{case}.png'
    print(fig_file)
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()


if __name__ == '__main__':
    # median_case()
    # trimmed_mean_case()

    num_repetitions = 10
    # dims = [500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 100000]
    dims = [500, 1000, 5000, 10000, 50000, 100000]
    # dims = [50, 100, 200]
    krum_rp_case(dims, case='dim', n=100, dim=None, k_factor=10, num_repetitions=num_repetitions)

    # dim = 10000
    # k_factor_max = int(dim / np.log(dim))
    # k_factors = [1, 5, 10, 20, 25, 50, 75, 100, 150, k_factor_max]
    # k_factors = [v for v in sorted(k_factors) if v <= k_factor_max]
    # print(k_factors)
    # krum_rp_case(k_factors, case='k_factor', n=100, dim=dim, k_factor=None, num_repetitions=num_repetitions)

    # # only for test
    # N = 100
    # f_max = (N - 3) // 2
    # synthetic_single_case(N=N, D=2, byzantine_mu=4, f=f_max, show=True, random_state=100)

    # N = 100
    # byzantine_mu_locations = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30][::-1]
    # f_max = (N - 3) // 2
    # synthetic_case(byzantine_mu_locations, case='byzantine_location', N=N, D=2, f=f_max)

    # Ns = [5, 10, 25, 50, 75, 100, 200, 300, 400, 500]
    # synthetic_case(Ns, case='N', D=2, byzantine_mu=10, f=None)

    # N = 100
    # f_max = (N - 3) // 2
    # Ds = [2, 5, 10, 25, 50, 100, 250, 500, 750, 1000]
    # synthetic_case(Ds, case='D', N=N, byzantine_mu=10, f=f_max)

    # # Case: different f
    # N = 100
    # # fs = [0, 5, 10, 25, 48]
    # f_max = (N - 3) // 2
    # # fs = list(range(5, f_max+1, (f_max-5)//10)) + [f_max]
    # # fs = [int(N * p) for p in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]]
    # # fs = [0] + fs + [f_max]
    # fs = [1, 5, 10, 15, 20, 25, 30, 40, 45, 48]
    # fs = [v for v in fs if v <= f_max]
    # fs = sorted(set(fs), reverse=False)
    # print(f_max, fs)
    # synthetic_case(fs, case='f', N=N, D=2, byzantine_mu=10)
