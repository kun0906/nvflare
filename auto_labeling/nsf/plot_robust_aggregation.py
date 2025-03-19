import re
import traceback

import matplotlib.pyplot as plt


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
        # ('refined_krum', f'log/output_{JOBID}_{start}.out'),
        ('adaptive_krum', f'log/output_{JOBID}_{start}.out'),
        ('krum', f'log/output_{JOBID}_{start + 1}.out'),
        # ('median', f'log/output_{JOBID}_{start + 2}.out'),
        ('mean', f'log/output_{JOBID}_{start + 3}.out'),
        # ('exp_weighted_mean', f'log/output_{JOBID}_{start + 6}.out'),

        # # Aggregated results: average point
        # # start2 = start + 5
        # ('adaptive_krum_avg', f'log/output_{JOBID}_{start2}.out'),
        # ('krum_avg', f'log/output_{JOBID}_{start2 + 1}.out'),
        # ('adaptive_krum+rp_avg', f'log/output_{JOBID}_{start2 + 2}.out'),
        # ('krum+rp_avg', f'log/output_{JOBID}_{start2 + 3}.out'),
        # ('median_avg', f'log/output_{JOBID}_{start2 + 4}.out'),
        # ('trimmed_mean', f'log/output_{JOBID}_{start2 + 5}.out'),
        # ('geometric_median', f'log/output_{JOBID}_{start2 + 6}.out'),

    ]

    # Example usage
    namespace_params = extract_namespace(f'log/output_{JOBID}_{start}.out')
    print(namespace_params, flush=True)
    # if (namespace_params['server_epochs'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
    #         or namespace_params['labeling_rate'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
    #         or namespace_params['num_clients'] in []):
    #     return
    # print(namespace_params)
    if (namespace_params['server_epochs'] == 10 and namespace_params['labeling_rate'] != 0.0
            and namespace_params['num_clients'] == 100):
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

    fig, ax = plt.subplots()
    FONTSIZE = 10
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
        elif METRIC == 'misclassification_error':
            vs = [vs['shared_acc'] for vs in global_accs[(agg_method, txt_file)]]
            vs = [(1 - v[0], v[1]) for v in vs] # (\mu, \std)
        else:
            raise NotImplementedError(METRIC)
        ys, ys_errs = zip(*vs)
        xs = range(len(ys))
        print(agg_method, txt_file, ys, flush=True)

        # plt.plot(xs, ys, label=label, marker=makers[i])
        # axes[i_row, j_col].plot(xs, ys, label=label, marker=makers[i])
        # axes[i_row, j_col].errorbar(xs, ys, yerr=ys_errs, label=label, marker=makers[i], capsize=3)
        plt.plot(xs, ys, label=label, marker=makers[i])  # Plot the line
        # axes[i_row, j_col].fill_between(xs,
        #                                 [y - e for y, e in zip(ys, ys_errs)],
        #                                 [y + e for y, e in zip(ys, ys_errs)],
        #                                 color='blue', alpha=0.3)  # label='Error Area'
    plt.xlabel('Epochs', fontsize=FONTSIZE)
    if len(xs) > 50:
        xs_labels = [1] + [v + 1 for v in xs if (v + 1) % 20 == 0]
    else:
        xs_labels = [v + 1 for v in xs]
    ax.set_xticks(xs_labels)
    ax.set_xticklabels(xs_labels)
    if METRIC == 'loss':
        plt.ylabel('Loss', fontsize=FONTSIZE)
    elif METRIC == 'l2_error':
        plt.ylabel('L_2 Error', fontsize=FONTSIZE)
    elif METRIC == 'time_taken':
        plt.ylabel('Time Taken', fontsize=FONTSIZE)
    elif METRIC == 'misclassification_error':
        plt.ylabel('Misclassification Error', fontsize=FONTSIZE)
    else:
        plt.ylabel('Accuracy')
    # plt.title(f'Global Model ({JOBID}), start:{start}, {title}', fontsize=10)
    plt.legend(fontsize=FONTSIZE, loc='best')

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

    fig, axes = plt.subplots(nrows=3, ncols=5, sharey=None, figsize=(15, 10))  # width, height
    # axes = axes.reshape((1, -1))

    j_col = 0
    for start in range(0, 110, 4):
        i_row = 0
        for METRIC in ['accuracy', 'l2_error', 'time_taken']:
            try:
                print(f'\nstart: {start}, {METRIC}')
                global_accs = {}
                start2 = start + 7
                method_txt_files = [
                    # # # # # Aggregated results: single point
                    ('adaptive_krum', f'log/output_{JOBID}_{start}.out'),
                    ('krum', f'log/output_{JOBID}_{start + 1}.out'),
                    ('median', f'log/output_{JOBID}_{start + 2}.out'),
                    ('mean', f'log/output_{JOBID}_{start + 3}.out'),
                    # ('exp_weighted_mean', f'log/output_{JOBID}_{start + 7}.out'),

                    # # # Aggregated results: average point
                    # ('adaptive_krum_avg', f'log/output_{JOBID}_{start2}.out'),
                    # ('krum_avg', f'log/output_{JOBID}_{start2 + 1}.out'),
                    # ('adaptive_krum+rp_avg', f'log/output_{JOBID}_{start2 + 2}.out'),
                    # ('krum+rp_avg', f'log/output_{JOBID}_{start2 + 3}.out'),
                    # ('medoid_avg', f'log/output_{JOBID}_{start2 + 4}.out'),
                    # ('trimmed_mean', f'log/output_{JOBID}_{start2 + 5}.out'),
                    # ('geometric_median', f'log/output_{JOBID}_{start2 + 6}.out'),

                ]
                case_name = extract_case_info(f'log/output_{JOBID}_{start}.out')
                print(case_name, flush=True)
                # Example usage
                namespace_params = extract_namespace(f'log/output_{JOBID}_{start}.out')
                # if (namespace_params['server_epochs'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['labeling_rate'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
                #         or namespace_params['num_clients'] in []):
                #     return
                # print(namespace_params)
                if (namespace_params['server_epochs'] == 10 and namespace_params['labeling_rate'] != 0.0
                        and namespace_params['num_clients'] == 20):
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
                    axes[i_row, j_col].plot(xs, ys, label=label, marker=makers[i])  # Plot the line
                    # axes[i_row, j_col].fill_between(xs,
                    #                                 [y - e for y, e in zip(ys, ys_errs)],
                    #                                 [y + e for y, e in zip(ys, ys_errs)],
                    #                                 color='blue', alpha=0.3)  # label='Error Area'
                plt.xlabel('Epochs', fontsize=10)
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

                FONTSIZE = 20
                axes[i_row, j_col].set_ylabel(ylabel, fontsize=FONTSIZE)

                variable = str(namespace_params['labeling_rate'])
                axes[i_row, j_col].set_title(f'start:{start}, {variable}', fontsize=FONTSIZE)
                axes[i_row, j_col].legend(fontsize=6.5, loc='lower right', fontsie=FONTSIZE)

            except Exception as e:
                print(e)
            i_row += 1
        j_col += 1

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')
    # plt.suptitle(f'Global Model ({JOBID}), start:{start}, {title}, {METRIC}', fontsize=10)
    plt.suptitle(f'Global Model ({JOBID}), {title}\n{case_name}', fontsize=FONTSIZE)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = f'global_{JOBID}.png'
    print(fig_file)
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # plot_robust_aggregation()

    ######################### Results 20250313 ###############################################
    # JOBID = 256611  # it works, log_large_values_20250214 with fixed large values

    # # Random noise injection with alpha=10, client_epochs=20, batch_size=512, epoch=10, num_clients=20, f=8
    # JOBID = 273327  # for Model Poisoning Attacks, random noise injection to model updates

    # # Random noise injection with alpha=10, client_epochs=20, batch_size=512, epoch=100, num_clients=50, f=23
    # JOBID = 274003  # for Model Poisoning Attacks, random noise injection to model updates

    # # # Large Value with alpha=10, client_epochs=20, batch_size=512, epoch=10, num_clients=20, f=8
    # JOBID = 273345  # for Model Poisoning Attacks, Large values to model updates

    # # # Large Value with alpha=10, client_epochs=20, batch_size=512, epoch=100, num_clients=50, f=23
    JOBID = 274024  # for Model Poisoning Attacks, Large values to model updates

    ### Label flipping

    # # Flip labels with alpha=10, client_epochs=1, batch_size=512, epoch=10, num_clients=20, f=8
    # JOBID = 273678  # for Data Poisoning Attacks, flip labels for Byzantine clients

    # # Flip labels with alpha=10, client_epochs=1, batch_size=512, epoch=100, num_clients=50, f=23
    # JOBID = 273720  # for Data Poisoning Attacks, flip labels for Byzantine clients

    ######################################################################################

    # JOBID = 273742
    # JOBID = 273327
    JOBID = 274219      # Sentiment140 with 30% data
    JOBID = 274249  # Sentiment140 with 10% data with different batch sizes
    JOBID = 274271  # Sentiment140 with 10% data with different alpha
    JOBID = 274320  # Sentiment140 with 10% data with different alpha
    # METRIC = 'loss'
    # METRIC = 'misclassification_error'  # or misclassification Rate
    # METRIC = "l2_error"  # 'accuracy'  # l2_error, time_taken
    METRIC = 'accuracy'
    for start in range(0, 100, 4):
        try:
            print(f'\nstart: {start}')
            plot_robust_aggregation(start, METRIC)
        except Exception as e:
            print(e)

    plot_robust_aggregation_all()
