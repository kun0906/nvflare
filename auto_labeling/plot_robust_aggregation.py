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


def plot_robust_aggregation(start=0):
    global_accs = {}
    JOBID = 256405  # it works

    method_txt_files = [
        ('refined_krum', f'log/output_{JOBID}_{start}.out'),
        ('krum', f'log/output_{JOBID}_{start + 1}.out'),
        ('median', f'log/output_{JOBID}_{start + 2}.out'),
        ('mean', f'log/output_{JOBID}_{start + 3}.out'),
    ]

    # Example usage
    namespace_params = extract_namespace(f'log/output_{JOBID}_{start}.out')
    if namespace_params['server_epochs'] != 5:
        return

    title = ', '.join(['benign_clients:' + str(namespace_params['benign_clients']),
                       'classes_cnt:' + str(namespace_params['server_epochs']),
                       'large_value:' + str(namespace_params['labeling_rate'])])
    for method, txt_file in method_txt_files:
        results = parse_file(txt_file)
        global_accs[method] = results

    plt.close()

    aggregation_methods = list(global_accs.keys())
    makers = ['o', '+', 's', '*']
    for i in range(len(aggregation_methods)):
        agg_method = aggregation_methods[i]
        label = agg_method
        ys = global_accs[agg_method]['shared_accs']
        xs = range(len(ys))
        plt.plot(xs, ys, label=label, marker=makers[i])
    plt.xlabel('Server Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Global CNN, start:{start}, {title}', fontsize=10)
    plt.legend(fontsize=6.5, loc='lower right')

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_BENIGN_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_BENIGN_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = 'global_cnn.png'
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()
    plt.close()


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


if __name__ == '__main__':
    # plot_robust_aggregation()
    for start in range(0, 300, 4):
        try:
            print(f'\nstart: {start}')
            plot_robust_aggregation(start)
        except Exception as e:
            print(e)
