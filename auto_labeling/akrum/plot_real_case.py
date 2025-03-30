""" Real dataset: MNIST

    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    # $module load conda
    # $conda activate nvflare-3.10
    # $cd nvflare/auto_labeling
    $module load conda && conda activate nvflare-3.10 && cd nvflare/auto_labeling
    $PYTHONPATH=. python3 akrum/plot_real_case.py

    Storage path: /projects/kunyang/nvflare_py31012/nvflare

"""
import time
import torch
import numpy as np

import ragg.robust_aggregation as rag
import matplotlib.pyplot as plt
import subprocess
import os

from akrum.plot_robust_aggregation import parse_file
from ragg.utils import timer

VERBOSE = 1
SEED = 42

# Generate 15 colors from the tab20 colormap
colors = plt.get_cmap("tab20").colors[:15]
makers = ['o', 's', '*', 'v', 'p', 'h', 'x', '8', '1', '^', 'D', 'd', '+', '.']


@timer
def real_case(tunable_values, tunable_type='different_location', num_clients=5, server_epochs=2, num_repeats=1,
              D=5, f=None, byzantine_mu=None):
    print(f'tunable_type: {tunable_type}, {tunable_values}')
    results = {}
    for tunable_value in tunable_values:
        if tunable_type == 'different_location':
            byzantine_mu = tunable_value
        elif tunable_type == 'different_n':
            num_clients = tunable_value
        elif tunable_type == 'different_d':
            D = tunable_value
        elif tunable_type == 'different_f':
            f = tunable_value
        else:
            raise NotImplementedError(tunable_type)

        # print(f"\n{tunable_type}, num_clients:{num_clients}, D:{D}, byzantine_mu:{byzantine_mu}, f:{f}", flush=True)
        print(f"\n{tunable_type}, {tunable_value}", flush=True)
        aggregation_methods = (
            # Single value
            # 'adaptive_krum', 'krum', 'adaptive_krum+rp', 'krum+rp', 'medoid', 'median', 'mean',
            'adaptive_krum',
            'krum',
            'medoid',
            'median',
            # 'mean',
            # Merged value
            # 'adaptive_krum_avg', 'krum_avg', 'adaptive_krum+rp_avg', 'krum+rp_avg', 'medoid_avg',
            # 'adaptive_krum_avg', 'krum_avg', 'medoid_avg',
            # 'trimmed_mean',
            'geometric_median'
        )
        avg_res = {}
        for method in aggregation_methods:
            # Set the command with parameters
            aggregation = method
            PARAMS = (f"-r {tunable_value} -t {tunable_type} -s {server_epochs} -n {num_clients} "
                      f"-a {aggregation} -R {num_repeats}")
            # Build the final command
            cmd = f"PYTHONPATH=. python3 akrum/ragg_model_large_value.py {PARAMS}"

            # Run the command using subprocess
            try:
                # result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # print(f"\n-----------------{aggregation} with tunable_value:{tunable_value}-----------------",
                #       flush=True)
                # # print the first 10 nonempty lines to get the basic info.
                # [print(v, flush=True) for v in result.stdout.decode().split('\n')[:10] if v != '']
                # # print the last 2 nonempty lines to get the executed time
                # [print(v, flush=True) for v in result.stdout.decode().split('\n')[-2:] if v != '']
                #
                # # Save the output to a file
                # txt_file = f"log/out_{PARAMS}.txt"
                # with open(txt_file, "w") as out_f:
                #     out_f.write(result.stdout.decode())

                txt_file = f"log/out_{PARAMS}.txt"
                METRIC = 'accuracy'
                result = parse_file(txt_file, metric=METRIC)

                # just use the last epoch's result
                if METRIC == 'accuracy':
                    mu_, std_ = result[-1]['shared_acc']  # (mu, std)
                elif METRIC == 'l2_error':
                    mu_, std_ = result[-1]['l2_error']
                elif METRIC == 'time_taken':
                    mu_, std_ = result[-1]['time_taken']
                    # if agg_method == 'median_avg': continue
                # elif METRIC == 'misclassification_error':
                #     vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
                #     vs = [1 - v for v in vs]
                else:
                    raise NotImplementedError(METRIC)
                avg_res[method] = {'accuracy': (mu_, std_), 'time_taken': (mu_, std_), 'l2_error': (mu_, std_)}
            # except subprocess.CalledProcessError as e:
            #     print("Error occurred:", e.stderr.decode())
            except Exception as e:
                print("Error:", e)

        results[tunable_value] = avg_res

    METRICS = ['accuracy', ]  # 'time_taken'
    nrows = 1
    ncols = len(METRICS)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=None, figsize=(5 * ncols, 4))  # width, height
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])  # Convert to 2D array shape (1,1)
    axes = axes.reshape((1, -1))
    title = ''
    i_row = 0
    agg_methods = results[tunable_values[0]].keys()
    for j_col, METRIC in enumerate(METRICS):
        for i, agg_method in enumerate(agg_methods):
            try:
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
                if tunable_type == 'different_location':
                    xlabel = '$||\mu_{byzantine} - \mu_{honest}||_2$'
                    title = f'n:{num_clients}, d:{D}, different_location:, f:{f}'
                elif tunable_type == 'different_n':
                    xlabel = 'n'
                    title = f'n:, d:{D}, different_location:{byzantine_mu}, f:{f}'
                elif tunable_type == 'different_d':
                    xlabel = 'd'
                    title = f'n:{num_clients}, d:, different_location:{byzantine_mu}, f:{f}'
                elif tunable_type == 'different_f':
                    xlabel = 'f'
                    title = f'n:{num_clients}, d:{D}, different_location:{byzantine_mu}, f:'
                else:
                    raise NotImplementedError(tunable_type)

                axes[i_row, j_col].set_xlabel(xlabel)
                # which is same to Byzantine $\mu$ Locations, as \mu_{honest} is 0.
                if METRIC == 'loss':
                    ylabel = 'Loss'
                elif METRIC == 'l2_error':
                    ylabel = '$||\hat{\mu} - \\bar{\mu}||_2$'
                elif METRIC == 'time_taken':
                    ylabel = 'Time Taken'
                # elif METRIC == 'misclassification_error':
                #     ylabel = 'Misclassification Error'
                else:
                    ylabel = 'Accuracy'
                axes[i_row, j_col].set_ylabel(ylabel)

                # variable = str(namespace_params['labeling_rate'])
                # axes[i_row, j_col].set_title(f'start:{start}, {variable}', fontsize=10)
                axes[i_row, j_col].legend(fontsize=6.5, loc='best')  # loc='upper right'
            except Exception as e:
                print("Error:", e)
    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')
    # plt.suptitle(f'Global Model ({JOBID}), start:{start}, {title}, {METRIC}', fontsize=10)
    plt.suptitle(f'{title}', fontsize=10)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = f'real_case_{tunable_type}.png'
    print(fig_file)
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600)
    plt.show()
    plt.close()


def _krum_rp_case(n, dim, k_factor, num_repeats=100):
    verbose = 0
    time_taken_list = []
    for i in range(num_repeats):
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
        empirical_cw_mean, clients_type = rag.cw_mean(points[:N - f], weights[:N - f], verbose=VERBOSE)
        method = 'empirical_cw_mean'
        SPACES = 16
        print(
            f'{method:{SPACES}s}: {[float(f"{v:.3f}") for v in empirical_cw_mean.tolist()]}, clients_type: {clients_type.tolist()}')

        clients_updates = points
        trimmed_average = False
        # print('Krum...')
        print('\nadaptive Krum...')
        start = time.time()
        aggregated_update, _ = rag.adaptive_krum(clients_updates, weights, trimmed_average,
                                                 random_projection=False, k_factor=k_factor,
                                                 verbose=verbose)
        end = time.time()
        time_taken = end - start
        l2_error = torch.norm(aggregated_update - empirical_cw_mean).item()
        print("Aggregated Update (adaptive Krum):", aggregated_update, time_taken)

        print('\nadaptive Krum with Random Projection...')
        start = time.time()
        aggregated_update2, _ = rag.adaptive_krum(clients_updates, weights, trimmed_average,
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


def krum_rp_case(tunable_values, case='dim', dim=1000, n=100, k_factor=10, num_repeats=100):
    results = {}
    for tunable_value in tunable_values:
        if case == 'dim':
            dim = tunable_value
        elif case == 'k_factor':
            k_factor = tunable_value
        else:
            raise NotImplementedError(case)
        # print(f'\naccuracy: {1 - len(results) / num_repeats}')
        time_taken_list = _krum_rp_case(n, dim, k_factor, num_repeats)
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
    # num_repeats = 10
    # # dims = [500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 100000]
    # dims = [500, 1000, 5000, 10000, 50000, 100000]
    # # dims = [50, 100, 200]
    # krum_rp_case(dims, case='dim', n=100, dim=None, k_factor=10, num_repeats=num_repeats)

    # dim = 10000
    # k_factor_max = int(dim / np.log(dim))
    # k_factors = [1, 5, 10, 20, 25, 50, 75, 100, 150, k_factor_max]
    # k_factors = [v for v in sorted(k_factors) if v <= k_factor_max]
    # print(k_factors)
    # krum_rp_case(k_factors, case='k_factor', n=100, dim=dim, k_factor=None, num_repeats=num_repeats)

    paper_experiment_flg = True
    if not paper_experiment_flg:  # for testing
        num_clients = 5
        server_epochs = 2
        num_repeats = 1

        # Case: different mus
        mus = [1, 2]
        real_case(tunable_type='different_location', tunable_values=mus,
                  num_clients=50, server_epochs=20, num_repeats=2)

        # Case: different d
        Ds = [2, 5]
        real_case(tunable_type='different_d', tunable_values=Ds,
                  num_clients=num_clients, server_epochs=server_epochs, num_repeats=num_repeats)

        # Case: different n
        Ns = [5, 10]
        real_case(tunable_type='different_n', tunable_values=Ns,
                  num_clients=num_clients, server_epochs=server_epochs, num_repeats=num_repeats)

        # Case: different f
        fs = [0, 0.1]
        real_case(tunable_type='different_f', tunable_values=fs,
                  num_clients=num_clients, server_epochs=server_epochs, num_repeats=num_repeats)

    else:
        num_clients = 50
        server_epochs = 20
        num_repeats = 5

        # Case: different mus
        # mus = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 20][::-1]
        mus = [0.5, 1, 2, 4, 8, 20][::-1]
        real_case(tunable_type='different_location', tunable_values=mus,
                  num_clients=num_clients, server_epochs=server_epochs, num_repeats=num_repeats)

        # # Case: different d
        # Ds = [2, 5, 10, 25, 50, 100, 250, 500, 600, 768]
        # real_case(tunable_type='different_d', tunable_values=Ds,
        #           num_clients=num_clients, server_epochs=server_epochs, num_repeats=num_repeats)
        #
        # # Case: different n
        # Ns = [5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 1000]
        # real_case(tunable_type='different_n', tunable_values=Ns,
        #           num_clients=num_clients, server_epochs=server_epochs, num_repeats=num_repeats)
        #
        # # Case: different f
        # # N = 100
        # # # fs = [0, 5, 10, 25, 48]
        # # f_max = (N - 3) // 2
        # # # fs = list(range(5, f_max+1, (f_max-5)//10)) + [f_max]
        # # # fs = [int(N * p) for p in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]]
        # # # fs = [0] + fs + [f_max]
        # # fs = [1, 5, 10, 15, 20, 25, 30, 40, 45, 48]
        # # fs = [v for v in fs if v <= f_max]
        # # fs = sorted(set(fs), reverse=False)
        # # print(f_max, fs)
        # fs = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49]
        # real_case(tunable_type='different_f', tunable_values=fs,
        #           num_clients=num_clients, server_epochs=server_epochs, num_repeats=num_repeats)
