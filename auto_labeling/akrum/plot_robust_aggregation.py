"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    # $module load conda
    # $conda activate nvflare-3.10
    # $cd nvflare/auto_labeling
    $module load conda && conda activate nvflare-3.10 && cd nvflare/auto_labeling
    $PYTHONPATH=. python3 akrum/plot_robust_aggregation.py

    Storage path: /projects/kunyang/nvflare_py31012/nvflare
"""

import re
import traceback
import numpy as np
import matplotlib.pyplot as plt

#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Generate 14 distinct colors from tab20 colormap
# colors = plt.cm.tab20(np.linspace(0, 1, 14))
#
# # Example usage in a scatter plot
# for i, color in enumerate(colors):
#     plt.scatter(i, 0, color=color, label=f'Algorithm {i+1}', s=100)
#
# plt.legend()
# plt.show()

# colors = [
#     'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
#     'olive', 'cyan', 'magenta', 'gold', 'navy', 'teal'
# ]
# import matplotlib.colors as mcolors
# print(mcolors.get_named_colors_mapping().keys())  # Show available colors


ALG2COLORS = {
    'adaptive_krum_avg': 'blue',
    'krum_avg': 'green',

    # 'adaptive_krum+rp_avg': 'lightblue',
    # 'krum+rp_avg': 'lightgreen',
    'adaptive_krum+rp_avg': 'purple',
    'krum+rp_avg': 'magenta',

    'adaptive_krum': '#4682B4',  # Steel Blue (close to "median blue")
    'krum': '#3CB371',  # Medium Sea Green (close to "median green")

    'adaptive_krum+rp': '#87CEFA',  # Light Sky Blue (close to "median light blue")
    'krum+rp': '#90EE90',  # Light Green

    'medoid': 'purple',
    'medoid_avg': '#D8BFD8',  # Thistle (light purple)

    'median': 'magenta',
    'geometric_median': 'brown',

    'trimmed_mean': '#E65100',  # Deep Dark Orange (Hex)
    'mean': 'orange'
}

# makers = ['o', '+', 's', '*', 'v', '.', 'p', 'h', 'x', '8', '1', '^', 'D', 'd']
ALG2MARKERS = {
    'adaptive_krum_avg': 'o',
    'krum_avg': 'P',

    'adaptive_krum+rp_avg': 'p',
    'krum+rp_avg': '*',

    'adaptive_krum': 'v',
    'krum': '.',

    'adaptive_krum+rp': 'p',
    'krum+rp': 'h',

    'medoid': 'x',  # X
    'medoid_avg': '8',

    'median': 's',
    'geometric_median': '^',

    'trimmed_mean': 'D',
    'mean': 'd'
}


#
# method_txt_files = [
#                     # # # # # Aggregated results: single point
#                     ('adaptive_krum', f'{IN_DIR}/out_{JOBID}_{start}.out'),
#                     ('krum', f'{IN_DIR}/out_{JOBID}_{start + 1}.out'),
#                     ('adaptive_krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 2}.out'),
#                     ('krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 3}.out'),
#                     ('medoid', f'{IN_DIR}/out_{JOBID}_{start + 4}.out'),
#                     ('median', f'{IN_DIR}/out_{JOBID}_{start + 5}.out'),
#                     # ('mean', f'{IN_DIR}/out_{JOBID}_{start + 6}.out'),
#                     # ('exp_weighted_mean', f'{IN_DIR}/out_{JOBID}_{start + 7}.out'),
#
#                     # # # Aggregated results: average point
#                     ('adaptive_krum_avg', f'{IN_DIR}/out_{JOBID}_{start2}.out'),
#                     ('krum_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 1}.out'),
#                     ('adaptive_krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 2}.out'),
#                     ('krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 3}.out'),
#                     ('medoid_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 4}.out'),
#                     ('geometric_median', f'{IN_DIR}/out_{JOBID}_{start2 + 6}.out'),
#                     ('trimmed_mean', f'{IN_DIR}/out_{JOBID}_{start2 + 5}.out'),
#
#                 ]


def extract_case_info(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

    case_name = lines[2].split('/')[-1]
    return case_name


def extract_namespace(filepath):
    with open(filepath, "r") as file:
        match = re.search(r'Namespace\((.*?)\)', file.read())
        if not match:
            return None

        params = {}
        for key, value in re.findall(r'(\w+)=([\S]+)', match.group(1)):
            # Remove trailing commas and handle numeric conversion
            clean_value = value.rstrip(',')
            if clean_value.replace('.', '', 1).isdigit():  # Check for int/float
                clean_value = float(clean_value) if '.' in clean_value else int(clean_value)
            else:  # string
                clean_value = clean_value.strip("'")
            params[key] = clean_value  # Remove quotes if present

        return params


#
# def parse_file_old(txt_file, metric = ''):
#     # Read the entire file into a string
#     with open(txt_file, "r") as f:
#         text = f.read()
#
#     # Find the section that starts after the global model marker
#     global_marker = r"\*\*\*model_type: global\*\*\*"
#     global_match = re.search(global_marker, text)
#     local_marker = r"\*\*\*model_type: local\*\*\*"
#     local_match = re.search(local_marker, text)
#     results_text = text[global_match.end():-1]
#     # results_text = text[global_match.end():local_match.start()]
#
#     # Find the block for client 0.
#     # We assume the client block starts with 'client 0' and ends before 'client 1' (or the end of file if no client 1).
#     client0_match = re.search(r"client 0(.*?)(client \d+|$)", results_text, re.S)
#     if client0_match:
#         client0_block = client0_match.group(1)
#     else:
#         raise ValueError("Client 0 block not found.")
#
#     # Define a regex pattern to extract the required metrics:
#     # It matches lines like:
#     if metric == 'loss':
#         # "Epoch: 0, labeled_loss: 0.00, val_loss: 0.00, unlabeled_loss: 0.00, shared_loss: 0.14"
#         pattern = (r"Epoch:\s*(\d+),\s*labeled_loss:\s*([\d\.]+),\s*val_loss:\s*([\d\.]+),"
#                    r"\s*unlabeled_loss:\s*([\d\.]+),\s*shared_loss:\s*([\d\.]+)")
#
#     elif metric == 'l2_error':
#         # Epoch: 0, labeled_acc:0.89, val_acc:0.90, unlabeled_acc:0.01, shared_acc:0.03, time_taken:0.01, l2_error:3.51
#         # pattern = (r"Epoch:\s*(\d+).*?l2_error:\s*([\d\.]+)")
#         pattern = (r"Epoch:\s*(\d+),\s*labeled_acc:\s*([\d\.]+),\s*val_acc:\s*([\d\.]+),"
#                    r"\s*unlabeled_acc:\s*([\d\.]+).*?l2_error:\s*([\d\.]+)")
#     elif metric == 'time_taken':
#         # Epoch: 0, labeled_acc:0.89, val_acc:0.90, unlabeled_acc:0.01, shared_acc:0.03, time_taken:0.01, l2_error:3.51
#         # pattern = (r"Epoch:\s*(\d+).*?time_taken:\s*([\d\.]+)")
#         pattern = (r"Epoch:\s*(\d+),\s*labeled_acc:\s*([\d\.]+),\s*val_acc:\s*([\d\.]+),"
#                    r"\s*unlabeled_acc:\s*([\d\.]+).*?time_taken:\s*([\d\.]+)")
#     else:
#         # pattern = (r"Epoch:\s*(\d+),\s*labeled_accuracy:\s*([\d\.]+),\s*val_accuracy:\s*([\d\.]+),"
#         #                r"\s*unlabeled_accuracy:\s*([\d\.]+),\s*shared_accuracy:\s*([\d\.]+)")
#
#         # "Epoch: 0, labeled_acc: 0.00, val_acc: 0.00, unlabeled_acc: 0.00, shared_acc: 0.14"
#         pattern = (r"Epoch:\s*(\d+),\s*labeled_acc:\s*([\d\.]+),\s*val_acc:\s*([\d\.]+),"
#                    r"\s*unlabeled_acc:\s*([\d\.]+),\s*shared_acc:\s*([\d\.]+)")
#
#     # Find all matches in the client 0 block
#     matches = re.findall(pattern, client0_block)
#
#     # Extract epochs and labeled_acc values
#     results = {'epochs': [],
#                'labeled_accs': [],
#                'val_accs': [],
#                'unlabeled_accs': [],
#                'shared_accs': []}
#     for m in matches:
#         results['epochs'].append(int(m[0]))
#         results['labeled_accs'].append(float(m[1]))
#         results['val_accs'].append(float(m[2]))
#         results['unlabeled_accs'].append(float(m[3]))
#         results['shared_accs'].append(float(m[4]))
#
#     return results

def plot_robust_aggregation(start=0, METRIC='accuracy'):
    """
       single point: 'adaptive_krum' 'krum' 'adaptive_krum+rp' 'krum+rp' 'median' 'mean'

       average points: 'adaptive_krum_avg' 'krum_avg' 'adaptive_krum+rp_avg' 'krum+rp_avg'
                            'median_avg' 'trimmed_mean' 'geometric_median'


    Args:
        start:

    Returns:

    """
    global_accs = {}
    start2 = start + 6
    method_txt_files = [
        # # # # Aggregated results: single point
        ('adaptive_krum', f'{IN_DIR}/out_{JOBID}_{start}.out'),
        ('krum', f'{IN_DIR}/out_{JOBID}_{start + 1}.out'),
        # ('adaptive_krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 2}.out'),
        # ('krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 3}.out'),
        ('median', f'{IN_DIR}/out_{JOBID}_{start + 4}.out'),
        ('mean', f'{IN_DIR}/out_{JOBID}_{start + 5}.out'),
        # # ('exp_weighted_mean', f'{IN_DIR}/out_{JOBID}_{start + 6}.out'),

        # # Aggregated results: average point
        # # start2 = start + 5
        # ('adaptive_krum_avg', f'{IN_DIR}/out_{JOBID}_{start2}.out'),
        # ('krum_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 1}.out'),
        # ('adaptive_krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 2}.out'),
        # ('krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 3}.out'),
        # ('median_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 4}.out'),
        ('trimmed_mean', f'{IN_DIR}/out_{JOBID}_{start2 + 5}.out'),
        ('geometric_median', f'{IN_DIR}/out_{JOBID}_{start2 + 6}.out'),

    ]

    # Example usage
    namespace_params = extract_namespace(f'{IN_DIR}/out_{JOBID}_{start}.out')
    # if (namespace_params['server_epochs'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
    #         or namespace_params['labeling_rate'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
    #         or namespace_params['num_clients'] in []):
    #     return
    # print(namespace_params)
    if (namespace_params['server_epochs'] == SERVER_EPOCHS and namespace_params['labeling_rate'] != 0.0
            and namespace_params['num_clients'] == NUM_CLIENTS):
        pass

        print(namespace_params)
    else:
        return

    title = ', '.join(['num_clients:' + str(namespace_params['num_clients']),
                       'classes_cnt:' + str(namespace_params['server_epochs']),
                       'large_value:' + str(namespace_params['labeling_rate'])])
    for method, txt_file in method_txt_files:
        namespace_params = extract_namespace(txt_file)
        method0 = namespace_params['aggregation_method']
        if method != method0:
            print(f'{method} != {method0}, ', txt_file, flush=True)
            continue
        try:
            results = parse_file(txt_file, metric=METRIC)
            global_accs[(method, txt_file)] = results
        except Exception as e:
            print(e, method0, txt_file, flush=True)
            # traceback.print_exc()

    plt.close()

    aggregation_methods = list(global_accs.keys())
    makers = ['o', '+', 's', '*', 'v', '.', 'p', 'h', 'x', '8', '1', '^', 'D', 'd']
    for i in range(len(aggregation_methods)):
        agg_method, txt_file = aggregation_methods[i]
        label = agg_method
        ys = global_accs[(agg_method, txt_file)]['shared_accs']  # [:10]
        if METRIC == 'misclassification_error':
            ys = [1 - v for v in ys]
        xs = range(len(ys))
        print(agg_method, txt_file, ys, flush=True)
        plt.plot(xs, ys, label=label, marker=makers[i])
    plt.xlabel('Server Epochs')
    if METRIC == 'loss':
        plt.ylabel('Loss')
    elif METRIC == 'l2_error':
        plt.ylabel('L_2 Error')
    elif METRIC == 'time_taken':
        plt.ylabel('Time Taken')
    elif METRIC == 'misclassification_error':
        plt.ylabel('Misclassification Error')
    else:
        plt.ylabel('Accuracy')
    plt.title(f'Global Model ({JOBID}), start:{start}, {title}', fontsize=10)
    plt.legend(fontsize=6.5, loc='lower right')

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = '../global_cnn.png'
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()
    plt.close()


def parse_file(txt_file, metric=''):
    # Read the entire file into a string
    with open(txt_file, "r") as f:
        text = f.read()

    # Find the section that starts after the global model marker
    global_marker = r"Final\*\*\*model_type: global\*\*\*"
    global_match = re.search(global_marker, text)
    local_marker = r"Final\*\*\*model_type: local\*\*\*"
    local_match = re.search(local_marker, text)
    results_text = text[global_match.end():-1]
    # results_text = text[global_match.end():local_match.start()]

    # Find the block for client 0.
    # We assume the client block starts with 'client 0' and ends before 'client 1' (or the end of file if no client 1).
    client0_match = re.search(r"\*client_0(.*?)(\*client_\d+|$)", results_text, re.S)
    if client0_match:
        client0_block = client0_match.group(1)
    else:
        raise ValueError("Client 0 block not found.")

    pattern = re.compile(
        r"server_epoch_(\d+),\s*"
        r"train_acc:([\d.]+)-([\d.]+),\s*"
        r"val_acc:([\d.]+)-([\d.]+),\s*"
        r"test_acc:([\d.]+)-([\d.]+),\s*"
        r"shared_acc:([\d.]+)-([\d.]+),\s*"
        r"l2_error:((?:[-+]?\d*\.\d+(?:[eE][-+]?\d+)?)|(?:[-+]?\d+(?:[eE][-+]?\d+)?))-"
        r"((?:[-+]?\d*\.\d+(?:[eE][-+]?\d+)?)|(?:[-+]?\d+(?:[eE][-+]?\d+)?)),\s*"
        r"time_taken:([\d.]+)-([\d.]+)"
    )

    results = []
    for match in pattern.finditer(client0_block):
        results.append({
            "server_epoch": int(match.group(1)),
            "train_acc": (float(match.group(2)), float(match.group(3))),
            "val_acc": (float(match.group(4)), float(match.group(5))),
            "test_acc": (float(match.group(6)), float(match.group(7))),
            "shared_acc": (float(match.group(8)), float(match.group(9))),
            "l2_error": (float(match.group(10)), float(match.group(11))),
            "time_taken": (float(match.group(12)), float(match.group(13))),
        })
    return results


def plot_robust_aggregation_all():
    """
       single point: 'adaptive_krum' 'krum' 'adaptive_krum+rp' 'krum+rp' 'median' 'mean'

       average points: 'adaptive_krum_avg' 'krum_avg' 'adaptive_krum+rp_avg' 'krum+rp_avg'
                            'median_avg' 'trimmed_mean' 'geometric_median'


    Args:
        start:

    Returns:

    """
    plt.close()

    fig, axes = plt.subplots(nrows=3, ncols=8, sharey=None, figsize=(30, 15))  # width, height
    # axes = axes.reshape((1, -1))

    j_col = 0
    for start in range(0, 110, 14):
        i_row = 0
        for METRIC in ['accuracy', 'l2_error', 'time_taken']:  # ['accuracy', 'l2_error', 'time_taken']:
            try:
                print(f'\nstart: {start}, {METRIC}')
                global_accs = {}
                start2 = start + 7
                method_txt_files = [
                    # # # # # Aggregated results: single point
                    ('adaptive_krum', f'{IN_DIR}/out_{JOBID}_{start}.out'),
                    ('krum', f'{IN_DIR}/out_{JOBID}_{start + 1}.out'),
                    ('adaptive_krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 2}.out'),
                    ('krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 3}.out'),
                    ('medoid', f'{IN_DIR}/out_{JOBID}_{start + 4}.out'),
                    ('median', f'{IN_DIR}/out_{JOBID}_{start + 5}.out'),
                    # ('mean', f'{IN_DIR}/out_{JOBID}_{start + 6}.out'),
                    # ('exp_weighted_mean', f'{IN_DIR}/out_{JOBID}_{start + 7}.out'),

                    # # # Aggregated results: average point
                    ('adaptive_krum_avg', f'{IN_DIR}/out_{JOBID}_{start2}.out'),
                    ('krum_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 1}.out'),
                    ('adaptive_krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 2}.out'),
                    ('krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 3}.out'),
                    ('medoid_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 4}.out'),
                    ('geometric_median', f'{IN_DIR}/out_{JOBID}_{start2 + 6}.out'),
                    ('trimmed_mean', f'{IN_DIR}/out_{JOBID}_{start2 + 5}.out'),

                ]
                # if AVG_FLG:
                #     method_txt_files= method_txt_files[7:]
                # else:
                #     method_txt_files = method_txt_files[:7]
                case_name = extract_case_info(f'{IN_DIR}/out_{JOBID}_{start}.out')
                print(case_name)
                # Example usage
                namespace_params = extract_namespace(f'{IN_DIR}/out_{JOBID}_{start}.out')
                # if (namespace_params['server_epochs'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['labeling_rate'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['num_clients'] in []):
                #     return
                # print(namespace_params)
                if (namespace_params['server_epochs'] == SERVER_EPOCHS and namespace_params['labeling_rate'] != -1.0
                        and namespace_params['num_clients'] == NUM_CLIENTS):
                    pass

                    print(namespace_params)
                    tunable_type = namespace_params['tunable_type']
                else:
                    print(namespace_params, flush=True)
                    return

                title = ', '.join(['num_clients:' + str(namespace_params['num_clients']),
                                   'classes_cnt:' + str(namespace_params['server_epochs']),
                                   'large_value:' + str(namespace_params['labeling_rate'])])
                for method, txt_file in method_txt_files:
                    namespace_params = extract_namespace(txt_file)
                    method0 = namespace_params['aggregation_method']
                    if method != method0:
                        print(f'{method} != {method0}, ', txt_file, flush=True)
                        continue
                    try:
                        results = parse_file(txt_file, metric=METRIC)
                        global_accs[(method, txt_file)] = results
                    except Exception as e:
                        print(e, method0, txt_file, flush=True)
                        # traceback.print_exc()

                aggregation_methods = list(global_accs.keys())
                makers = ['o', '+', 's', '*', 'v', '.', 'p', 'h', 'x', '8', '1', '^', 'D', 'd']
                for i in range(len(aggregation_methods)):
                    agg_method, txt_file = aggregation_methods[i]
                    label = agg_method
                    if METRIC == 'accuracy':
                        vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
                    elif METRIC == 'l2_error':
                        vs = [vs['l2_error'] for vs in global_accs[(agg_method, txt_file)]]
                    elif METRIC == 'time_taken':
                        vs = [vs['time_taken'] for vs in global_accs[(agg_method, txt_file)]]
                        if agg_method == 'median_avg': continue
                    # elif METRIC == 'misclassification_error':
                    #     vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
                    #     vs = [1 - v for v in vs]
                    else:
                        raise NotImplementedError(METRIC)
                    ys, ys_errs = zip(*vs)
                    xs = range(len(ys))
                    print(agg_method, txt_file, ys, flush=True)
                    # print(agg_method, txt_file, [f"{v[0]:.2f}/{v[1]:.2f}" for v in vs], flush=True)

                    # axes[i_row, j_col].plot(xs, ys, label=label, marker=makers[i])
                    # axes[i_row, j_col].errorbar(xs, ys, yerr=ys_errs, label=label, marker=makers[i], capsize=3)
                    axes[i_row, j_col].plot(xs, ys, label=label,
                                            marker=ALG2MARKERS[agg_method],
                                            color=ALG2COLORS[agg_method])  # Plot the line
                    axes[i_row, j_col].fill_between(xs,
                                                    [y - e for y, e in zip(ys, ys_errs)],
                                                    [y + e for y, e in zip(ys, ys_errs)],
                                                    alpha=0.3)  # label='Error Area', color='blue'
                plt.xlabel('Server Epochs')
                if METRIC == 'loss':
                    ylabel = 'Loss'
                elif METRIC == 'l2_error':
                    ylabel = 'L_2 Error'
                elif METRIC == 'time_taken':
                    ylabel = 'Time Taken'
                # elif METRIC == 'misclassification_error':
                #     ylabel = 'Misclassification Error'
                else:
                    ylabel = 'Accuracy'
                axes[i_row, j_col].set_ylabel(ylabel)

                variable = str(namespace_params['labeling_rate'])
                axes[i_row, j_col].set_title(f'start:{start}, {variable}', fontsize=10)
                axes[i_row, j_col].legend(fontsize=6.5, loc='lower right')

            except Exception as e:
                print(e)
            i_row += 1
        j_col += 1

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')
    # plt.suptitle(f'Global Model ({JOBID}), start:{start}, {title}, {METRIC}', fontsize=10)
    plt.suptitle(f'Global Model ({JOBID}), {title}\n{case_name}', fontsize=10)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = f'global_{JOBID}_all_{tunable_type}.png'
    print(fig_file)
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600)
    plt.show()
    plt.close()


def plot_paper_results(metric='accuracy', fig_type='', algorithm_list=[]):
    """
           single point: 'adaptive_krum' 'krum' 'adaptive_krum+rp' 'krum+rp' 'median' 'mean'

           average points: 'adaptive_krum_avg' 'krum_avg' 'adaptive_krum+rp_avg' 'krum+rp_avg'
                                'median_avg' 'trimmed_mean' 'geometric_median'


        Args:
            start:

        Returns:

        """

    j_col = 0
    all_results = {}
    Xs = []
    for start in range(0, 14 * 7, 14):
        i_row = 0
        all_results[start] = {}
        for METRIC in [metric]:  # ['accuracy', 'l2_error', 'time_taken']:
            all_results[start][METRIC] = {}
            try:
                print(f'\nstart: {start}, {METRIC}')
                global_accs = {}
                start2 = start + 7
                method_txt_files = [
                    # # # # # Aggregated results: single point
                    ('adaptive_krum', f'{IN_DIR}/out_{JOBID}_{start}.out'),
                    ('krum', f'{IN_DIR}/out_{JOBID}_{start + 1}.out'),
                    ('adaptive_krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 2}.out'),
                    ('krum+rp', f'{IN_DIR}/out_{JOBID}_{start + 3}.out'),
                    ('medoid', f'{IN_DIR}/out_{JOBID}_{start + 4}.out'),
                    ('median', f'{IN_DIR}/out_{JOBID}_{start + 5}.out'),
                    ('mean', f'{IN_DIR}/out_{JOBID}_{start + 6}.out'),
                    # ('exp_weighted_mean', f'{IN_DIR}/out_{JOBID}_{start + 7}.out'),

                    # # # Aggregated results: average point
                    ('adaptive_krum_avg', f'{IN_DIR}/out_{JOBID}_{start2}.out'),
                    ('krum_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 1}.out'),
                    ('adaptive_krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 2}.out'),
                    ('krum+rp_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 3}.out'),
                    # ('medoid_avg', f'{IN_DIR}/out_{JOBID}_{start2 + 4}.out'),
                    ('geometric_median', f'{IN_DIR}/out_{JOBID}_{start2 + 6}.out'),
                    ('trimmed_mean', f'{IN_DIR}/out_{JOBID}_{start2 + 5}.out'),

                ]
                # if AVG_FLG:
                #     method_txt_files= method_txt_files[7:]
                # else:
                #     method_txt_files = method_txt_files[:7]
                case_name = extract_case_info(f'{IN_DIR}/out_{JOBID}_{start}.out')
                print(case_name)
                # Example usage
                namespace_params = extract_namespace(f'{IN_DIR}/out_{JOBID}_{start}.out')
                # if (namespace_params['server_epochs'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['labeling_rate'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['num_clients'] in []):
                #     return
                # print(namespace_params)
                if (namespace_params['server_epochs'] == SERVER_EPOCHS and namespace_params['labeling_rate'] != -1.0
                        and namespace_params['num_clients'] == NUM_CLIENTS):
                    pass

                    print(namespace_params)
                    tunable_type = namespace_params['tunable_type']
                else:
                    print(namespace_params, flush=True)
                    return

                v = namespace_params['labeling_rate']
                if v not in set(Xs):
                    Xs.append(v)
                title = ', '.join(['num_clients:' + str(namespace_params['num_clients']),
                                   'server_epochs:' + str(namespace_params['server_epochs']),
                                   'n_repeats:' + str(namespace_params['num_repeats'])])

                for method in algorithm_list:
                    txt_file = [txt_file for m, txt_file in method_txt_files if m == method][0]
                    namespace_params = extract_namespace(txt_file)
                    method0 = namespace_params['aggregation_method']
                    if method != method0:
                        print(f'{method} != {method0}, ', txt_file, flush=True)
                        continue
                    try:
                        results = parse_file(txt_file, metric=METRIC)
                        global_accs[(method, txt_file)] = results
                    except Exception as e:
                        print(e, method0, txt_file, flush=True)
                        # traceback.print_exc()

                aggregation_methods = list(global_accs.keys())
                for i in range(len(aggregation_methods)):
                    agg_method, txt_file = aggregation_methods[i]
                    label = agg_method
                    if METRIC == 'accuracy':
                        vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
                    elif METRIC == 'l2_error':
                        vs = [vs['l2_error'] for vs in global_accs[(agg_method, txt_file)]]
                    elif METRIC == 'time_taken':
                        vs = [vs['time_taken'] for vs in global_accs[(agg_method, txt_file)]]
                        if agg_method == 'median_avg': continue
                    # elif METRIC == 'misclassification_error':
                    #     vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
                    #     vs = [1 - v for v in vs]
                    else:
                        raise NotImplementedError(METRIC)
                    ys, ys_errs = zip(*vs)
                    xs = range(len(ys))
                    print(agg_method, txt_file, ys, flush=True)

                    all_results[start][METRIC][agg_method] = (ys, ys_errs, txt_file)

            except Exception as e:
                traceback.print_exc()
            i_row += 1

        j_col += 1

    ####################### plot the results
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=None, figsize=(8, 6))  # width, height
    axes = np.array([axes]).reshape((1, -1))
    # axes = axes.reshape((1, -1))

    print(f'\n')
    # METRICS = ['accuracy', 'l2_error', 'time_taken']
    METRIC = metric
    start = 0
    # print(agg_method, txt_file, [f"{v[0]:.2f}/{v[1]:.2f}" for v in vs], flush=True)
    i_row = 0
    j_col = 0
    for i, agg_method in enumerate(algorithm_list):  # all_results[start][METRIC].keys()
        # axes[i_row, j_col].plot(xs, ys, label=label, marker=makers[i])
        # axes[i_row, j_col].errorbar(xs, ys, yerr=ys_errs, label=label, marker=makers[i], capsize=3)
        # xs = list(all_results.keys())
        xs = []
        ys = []
        ys_errs = []
        for start_ in all_results.keys():
            try:
                ys_, ys_errs_, txt_file_ = all_results[start_][METRIC][agg_method]
                xs.append(start_)
                ys.append(ys_[-1])  # for each result, we only use the last epoch's result for this plot
                ys_errs.append(ys_errs_[-1])
            except Exception as e:
                traceback.print_exc()
        print(METRIC, agg_method, xs, ys, ys_errs, flush=True)
        label = agg_method
        axes[i_row, j_col].plot(xs, ys, label=label,
                                marker=ALG2MARKERS[agg_method],
                                color=ALG2COLORS[agg_method])  # Plot the line
        axes[i_row, j_col].fill_between(xs,
                                        [y - e for y, e in zip(ys, ys_errs)],
                                        [y + e for y, e in zip(ys, ys_errs)],
                                        alpha=0.3)  # label='Error Area', color='blue'

        if tunable_type == 'different_n':
            XLabel = 'N'
            Xs = [int(v) for v in Xs]
        elif tunable_type == 'different_f':
            XLabel = 'f'
        elif tunable_type == 'different_d':
            XLabel = 'D'
            Xs = [int(v) for v in Xs]
        elif tunable_type == 'different_mu':
            XLabel = 'Location'
        elif tunable_type == 'different_var':
            XLabel = 'Variance'
        else:
            raise NotImplementedError(tunable_type)
        plt.xlabel(XLabel, fontsize=10)
        max_len = min(len(Xs), len(xs))
        Xs, xs = Xs[:max_len], xs[:max_len]
        axes[i_row, j_col].set_xticks(xs)
        xs_labels = Xs
        axes[i_row, j_col].set_xticklabels(xs_labels)

        if METRIC == 'loss':
            ylabel = 'Loss'
        elif METRIC == 'l2_error':
            ylabel = 'L_2 Error'
        elif METRIC == 'time_taken':
            ylabel = 'Time Taken'
        # elif METRIC == 'misclassification_error':
        #     ylabel = 'Misclassification Error'
        else:
            ylabel = 'Accuracy'
        axes[i_row, j_col].set_ylabel(ylabel)

        # variable = str(namespace_params['labeling_rate'])
        # axes[i_row, j_col].set_title(f'start:{start}, {variable}', fontsize=10)
        axes[i_row, j_col].legend(fontsize=6.5, loc='lower right')

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')
    # plt.suptitle(f'Global Model ({JOBID}), start:{start}, {title}, {METRIC}', fontsize=10)
    plt.suptitle(f'Global Model ({JOBID}), {title}\n{case_name}', fontsize=10)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    # fig_suffix= "{XLabel}_{metric}_{fig_type}"
    fig_file = f'global_{JOBID}_{XLabel}_{metric}_{fig_type}.png'
    print(fig_file)
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600)
    plt.show()
    plt.close()


if __name__ == '__main__':
    SERVER_EPOCHS = 100
    NUM_CLIENTS = 50
    IN_DIR = 'log'
    # IN_DIR = '/projects/kunyang/nvflare_py31012/nvflare/log'
    # AVG_FLG = False
    ############################################################################################

    # Model large value, dimension = 5
    # JOBID = 278953  # with different byzantine locations \mu

    # JOBID = 279070  # with different f

    # JOBID = 279273  # with different n, number of clients
    #
    # JOBID = 279933  # with different variance, 279730
    # JOBID = 280039  # with different variance, 279730
    #
    # JOBID = 280164  # with different reduced dimensions
    #
    # JOBID = 280707  # with different reduced dimensions
    # JOBID = 280808  # with different reduced dimensions
    # JOBID = 281108  # with different reduced dimensions

    #################################################### different D
    JOBID = 281632  # with different reduced dimensions     log/

    #################################################### different n
    # JOBID = 281763  # with different n, num_repeats=10    log/

    # JOBID = 281991  # with different reduced dimensions     log/

    #################################################### different f
    # JOBID = 282178  # with different reduced dimensions, different f
    #
    # JOBID = 282380  # with different reduced dimensions log/
    #
    # JOBID = 282547  # with different reduced dimensions /projects/kunyang/nvflare_py31012/nvflare

    # plot_robust_aggregation()
    # JOBID = 256611  # it works, log_large_values_20250214 with fixed large values
    # JOBID = 271247  # 266353 #266233 #265651 #265426 #265364 #265338 # 265030
    # METRIC = 'loss'
    # METRIC = 'misclassification_error'  # or misclassification Rate
    # METRIC = "l2_error"  # 'accuracy'  # l2_error, time_taken

    # for start in range(0, 100, 14):
    #     try:
    #         print(f'\nstart: {start}')
    #         plot_robust_aggregation(start, METRIC)
    #     except Exception as e:
    #         print(e)

    # plot_robust_aggregation_all()

    # plot the results for latex paper
    algorithm_list = ['adaptive_krum_avg', 'krum_avg',
                      # 'adaptive_krum+rp_avg', 'krum+rp_avg',

                      # 'adaptive_krum', 'krum',
                      # 'adaptive_krum+rp', 'krum+rp',

                      'medoid',

                      'median',
                      'geometric_median',

                      'trimmed_mean',
                      'mean'
                      ]
    plot_paper_results(metric='accuracy', fig_type='all',
                       algorithm_list=algorithm_list)  # accuracy, l2_error, time_taken

    # plot the results for latex paper
    algorithm_list = ['adaptive_krum_avg', 'krum_avg',
                      'adaptive_krum+rp_avg', 'krum+rp_avg',

                      'adaptive_krum', 'krum',
                      'adaptive_krum+rp', 'krum+rp',
                      ]
    plot_paper_results(metric='accuracy', fig_type='with+wo-rp',
                       algorithm_list=algorithm_list)  # accuracy, l2_error, time_taken
