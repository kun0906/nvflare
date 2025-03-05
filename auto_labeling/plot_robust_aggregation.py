import re
import traceback

import matplotlib.pyplot as plt


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


def parse_file(txt_file):
    # Read the entire file into a string
    with open(txt_file, "r") as f:
        text = f.read()

    # Find the section that starts after the global model marker
    global_marker = r"\*\*\*model_type: global\*\*\*"
    global_match = re.search(global_marker, text)
    local_marker = r"\*\*\*model_type: local\*\*\*"
    local_match = re.search(local_marker, text)
    results_text = text[global_match.end():-1]
    # results_text = text[global_match.end():local_match.start()]

    # Find the block for client 0.
    # We assume the client block starts with 'client 0' and ends before 'client 1' (or the end of file if no client 1).
    client0_match = re.search(r"client 0(.*?)(client \d+|$)", results_text, re.S)
    if client0_match:
        client0_block = client0_match.group(1)
    else:
        raise ValueError("Client 0 block not found.")

    # Define a regex pattern to extract the required metrics:
    # It matches lines like:
    if METRIC == 'loss':
        # "Epoch: 0, labeled_loss: 0.00, val_loss: 0.00, unlabeled_loss: 0.00, shared_loss: 0.14"
        pattern = (r"Epoch:\s*(\d+),\s*labeled_loss:\s*([\d\.]+),\s*val_loss:\s*([\d\.]+),"
                   r"\s*unlabeled_loss:\s*([\d\.]+),\s*shared_loss:\s*([\d\.]+)")
    else:
        # pattern = (r"Epoch:\s*(\d+),\s*labeled_accuracy:\s*([\d\.]+),\s*val_accuracy:\s*([\d\.]+),"
        #                r"\s*unlabeled_accuracy:\s*([\d\.]+),\s*shared_accuracy:\s*([\d\.]+)")

        # "Epoch: 0, labeled_acc: 0.00, val_acc: 0.00, unlabeled_acc: 0.00, shared_acc: 0.14"
        pattern = (r"Epoch:\s*(\d+),\s*labeled_acc:\s*([\d\.]+),\s*val_acc:\s*([\d\.]+),"
                   r"\s*unlabeled_acc:\s*([\d\.]+),\s*shared_acc:\s*([\d\.]+)")

    # Find all matches in the client 0 block
    matches = re.findall(pattern, client0_block)

    # Extract epochs and labeled_acc values
    results = {'epochs': [],
               'labeled_accs': [],
               'val_accs': [],
               'unlabeled_accs': [],
               'shared_accs': []}
    for m in matches:
        results['epochs'].append(int(m[0]))
        results['labeled_accs'].append(float(m[1]))
        results['val_accs'].append(float(m[2]))
        results['unlabeled_accs'].append(float(m[3]))
        results['shared_accs'].append(float(m[4]))

    return results


def plot_robust_aggregation(start=0):
    """
       single point: 'refined_krum' 'krum' 'refined_krum+rp' 'krum+rp' 'median' 'mean'

       average points: 'refined_krum_avg' 'krum_avg' 'refined_krum+rp_avg' 'krum+rp_avg'
                            'median_avg' 'trimmed_mean' 'geometric_median'


    Args:
        start:

    Returns:

    """
    global_accs = {}
    start2 = start + 6
    method_txt_files = [
        # Aggregated results: single point
        ('refined_krum', f'log/output_{JOBID}_{start}.out'),
        ('krum', f'log/output_{JOBID}_{start + 1}.out'),
        ('refined_krum+rp', f'log/output_{JOBID}_{start + 2}.out'),
        ('krum+rp', f'log/output_{JOBID}_{start + 3}.out'),
        ('median', f'log/output_{JOBID}_{start + 4}.out'),
        ('mean', f'log/output_{JOBID}_{start + 5}.out'),
        # ('exp_weighted_mean', f'log/output_{JOBID}_{start + 6}.out'),

        # # Aggregated results: average point
        # # start2 = start + 5
        # ('refined_krum_avg', f'log/output_{JOBID}_{start2}.out'),
        # ('krum_avg', f'log/output_{JOBID}_{start2 + 1}.out'),
        # ('refined_krum+rp_avg', f'log/output_{JOBID}_{start2 + 2}.out'),
        # ('krum+rp_avg', f'log/output_{JOBID}_{start2 + 3}.out'),
        # ('median_avg', f'log/output_{JOBID}_{start2 + 4}.out'),
        # ('trimmed_mean', f'log/output_{JOBID}_{start2 + 5}.out'),
        # ('geometric_median', f'log/output_{JOBID}_{start2 + 6}.out'),

    ]

    # Example usage
    namespace_params = extract_namespace(f'log/output_{JOBID}_{start}.out')
    # if (namespace_params['server_epochs'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
    #         or namespace_params['labeling_rate'] in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.8, 9.0, 10.0]
    #         or namespace_params['num_clients'] in []):
    #     return
    # print(namespace_params)
    if (namespace_params['server_epochs'] == 5 and namespace_params['labeling_rate'] != 0.0
            and namespace_params['num_clients'] == 10):
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
            print(f'{method} != {method0}, ', txt_file)
            continue
        try:
            results = parse_file(txt_file)
            global_accs[method] = results
        except Exception as e:
            print(e, method0, txt_file, flush=True)
            # traceback.print_exc()

    plt.close()

    aggregation_methods = list(global_accs.keys())
    makers = ['o', '+', 's', '*', 'v', '.', 'p', 'h', 'x', '8', '1', '^', 'D', 'd']
    for i in range(len(aggregation_methods)):
        agg_method = aggregation_methods[i]
        label = agg_method
        ys = global_accs[agg_method]['shared_accs']  # [:10]
        if METRIC == 'misclassified_error':
            ys = [1 - v for v in ys]
        xs = range(len(ys))
        print(agg_method, ys)
        plt.plot(xs, ys, label=label, marker=makers[i])
    plt.xlabel('Server Epochs')
    if METRIC == 'loss':
        plt.ylabel('Loss')
    elif METRIC == 'misclassified_error':
        plt.ylabel('Misclassified Error')
    else:
        plt.ylabel('Accuracy')
    plt.title(f'Global Model ({JOBID}), start:{start}, {title}', fontsize=10)
    plt.legend(fontsize=6.5, loc='lower right')

    # attacker_ratio = NUM_MALICIOUS_CLIENTS / (NUM_HONEST_CLIENTS + NUM_MALICIOUS_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_MALICIOUS_CLIENTS}_accuracy.png')
    fig_file = 'global_cnn.png'
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # plot_robust_aggregation()
    # JOBID = 256611  # it works, log_large_values_20250214 with fixed large values
    JOBID = 267766  # 266353 #266233 #265651 #265426 #265364 #265338 # 265030
    # METRIC = 'loss'
    # METRIC = 'misclassified_error'  # or misclassification Rate
    METRIC = 'accuracy'  # or misclassification Rate
    for start in range(0, 100, 13):
        try:
            print(f'\nstart: {start}')
            plot_robust_aggregation(start)
        except Exception as e:
            print(e)
