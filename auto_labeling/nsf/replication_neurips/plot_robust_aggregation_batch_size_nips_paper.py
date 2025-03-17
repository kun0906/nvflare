import re

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
            params[key] = clean_value  # Remove quotes if present

        return params

#
# def parse_file(txt_file):
#     # Read the entire file into a string
#     with open(txt_file, "r") as f:
#         text = f.read()
#
#     # Find the section that starts after the global model marker
#     global_marker = r"\*\*\*model_type: global\*\*\*"
#     global_match = re.search(global_marker, text)
#     local_marker = r"\*\*\*model_type: local\*\*\*"
#     local_match = re.search(local_marker, text)
#
#     results_text = text[global_match.end():local_match.start()]
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
#     if METRIC == 'loss':
#         # "Epoch: 0, labeled_loss: 0.00, val_loss: 0.00, unlabeled_loss: 0.00, shared_loss: 0.14"
#         pattern = (r"Epoch:\s*(\d+),\s*labeled_loss:\s*([\d\.]+),\s*val_loss:\s*([\d\.]+),"
#                    r"\s*unlabeled_loss:\s*([\d\.]+),\s*shared_loss:\s*([\d\.]+)")
#     else:
#         pattern = (r"Epoch:\s*(\d+),\s*labeled_accuracy:\s*([\d\.]+),\s*val_accuracy:\s*([\d\.]+),"
#                    r"\s*unlabeled_accuracy:\s*([\d\.]+),\s*shared_accuracy:\s*([\d\.]+)")
#         # "Epoch: 0, labeled_acc: 0.00, val_acc: 0.00, unlabeled_acc: 0.00, shared_acc: 0.14"
#         # pattern = (r"Epoch:\s*(\d+),\s*labeled_acc:\s*([\d\.]+),\s*val_acc:\s*([\d\.]+),"
#         #            r"\s*unlabeled_acc:\s*([\d\.]+),\s*shared_acc:\s*([\d\.]+)")
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


def parse_file(txt_file):
    # Read the entire file into a string
    with open(txt_file, "r") as f:
        text = f.read()

    # Find the section that starts after the global model marker
    global_marker = r"\*\*\*model_type: global\*\*\*"
    global_match = re.search(global_marker, text)
    local_marker = r"\*\*\*model_type: local\*\*\*"
    local_match = re.search(local_marker, text)

    results_text = text[global_match.end():local_match.start()]

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
        pattern = (r"Epoch:\s*(\d+),\s*labeled_accuracy:\s*([\d\.]+),\s*val_accuracy:\s*([\d\.]+),"
                   r"\s*unlabeled_accuracy:\s*([\d\.]+),\s*shared_accuracy:\s*([\d\.]+)")
        # "Epoch: 0, labeled_acc: 0.00, val_acc: 0.00, unlabeled_acc: 0.00, shared_acc: 0.14"
        # pattern = (r"Epoch:\s*(\d+),\s*labeled_acc:\s*([\d\.]+),\s*val_acc:\s*([\d\.]+),"
        #            r"\s*unlabeled_acc:\s*([\d\.]+),\s*shared_acc:\s*([\d\.]+)")

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


def plot_robust_aggregation():

    # JOBID_no_malicious = 265364  # with 0% malicious clients
    # JOBID_with_malicious = 265338  # with 45% malicious clients
    # starts = [0, 4, 8, 12, 16] #  the correpsoding epochs = [20, 50, 80, 100]
    # epochs = [3, 20, 50, 80, 100]


    # JOBID_no_malicious = 265412  # with 0% malicious clients
    # JOBID_with_malicious = 265651  # with 45% malicious clients
    # starts = [4, 8, 12, 16, 20, 24] #  the correpsoding BATCH_SIZE = [20, 50, 80, 100]
    # BATCH_SIZE = [2, 3, 20, 50, 80, 100]

    JOBID_no_malicious = 272544  # with 0% malicious clients
    JOBID_with_malicious = 272544  # with 45% malicious clients
    starts = [4]  # the correpsoding BATCH_SIZE = [20, 50, 80, 100]
    # BATCH_SIZE = [2, 3, 20, 50, 80, 100]
    # BATCH_SIZE = [2, 3, 20, 50, 80, 100]
    BATCH_SIZE = range(20)

    all_results = {}
    for start in starts:
        global_accs = {}
        method_txt_files = [
            # # ('adaptive_krum', f'log/output_{JOBID_no_malicious}_{start}.out'),
            # ('krum (0% byz)', f'log/output_{JOBID_no_malicious}_{start + 1}.out'),
            # # ('median', f'log/output_{JOBID}_{start + 2}.out'),
            # ('mean (0% byz)', f'log/output_{JOBID_no_malicious}_{start + 3}.out'),

            ('adaptive_krum (45% byz)', f'log/output_{JOBID_with_malicious}_{start}.out'),
            ('krum (45% byz)', f'log/output_{JOBID_with_malicious}_{start + 1}.out'),
            ('median (45% byz)', f'log/output_{JOBID_with_malicious}_{start + 2}.out'),
            ('mean (45% byz)', f'log/output_{JOBID_with_malicious}_{start + 3}.out'),
        ]

        for method, txt_file in method_txt_files:
            results = parse_file(txt_file)
            global_accs[method] = results

        all_results[start] = global_accs

    aggregation_methods = list(global_accs.keys())
    makers = ['o', '+', 's', '*']
    colors = ['']
    for i in range(len(aggregation_methods)):
        agg_method = aggregation_methods[i]
        label = agg_method
        ys = global_accs[agg_method]['shared_accs'] #[:10]
        # # choose sever_epoch = 10 for each method
        # sever_epoch = 100
        # ys = [res[agg_method]['shared_accs'][sever_epoch] for res in all_results.values()]
        if METRIC == 'misclassification_error':
            ys = [1-v for v in ys]
        print(agg_method, [float(f'{v:.2f}') for v in ys])
        xs = range(len(ys))
        # xs = BATCH_SIZE
        plt.plot(xs, ys, label=label, marker=makers[i])
    plt.xlabel('Batch size')
    if METRIC == 'loss':
        plt.ylabel('Loss')
    elif METRIC == 'misclassification_error':
        plt.ylabel('Misclassification Error')
    else:
        plt.ylabel('Accuracy')
    title = f'{JOBID_no_malicious} + {JOBID_with_malicious}'
    plt.title(f'Global Model, start:{start}, {title}', fontsize=10)
    plt.legend(fontsize=6.5, loc='lower right')

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = '../../global_cnn.png'
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # METRIC = 'misclassification_error'  # or misclassification Rate
    # plot_robust_aggregation()


    # plot_robust_aggregation()
    # JOBID = 256611  # it works, log_large_values_20250214 with fixed large values
    JOBID = 265290 # 265030
    METRIC = 'loss'
    METRIC = 'misclassification_error'  # or misclassification Rate
    for start in range(0, 100, 4):
        try:
            print(f'\nstart: {start}')
            plot_robust_aggregation(start)
        except Exception as e:
            print(e)
