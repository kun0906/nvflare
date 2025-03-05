"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    # $module load conda
    # $conda activate nvflare-3.10
    # $cd nvflare/auto_labeling
    $module load conda && conda activate nvflare-3.10 && cd nvflare/auto_labeling
    $PYTHONPATH=. python3 fl_cnn_robust_aggregation_label_flipping.py

    Storage path: /projects/kunyang/nvflare_py31012/nvflare


    https://github.com/LPD-EPFL/byzfl
    how to generate honest clients dataset
    # Distribute data among clients using non-IID Dirichlet distribution
data_distributor = DataDistributor({
    "data_distribution_name": "dirichlet_niid",
    "distribution_parameter": 0.5,
    "nb_honest": nb_honest_clients,
    "data_loader": train_loader,
    "batch_size": batch_size,
})
client_dataloaders = data_distributor.split_data()

"""

import os
import pickle

import numpy as np
import torch
import argparse

from utils import dirichlet_split

print(f'current directory: {os.path.abspath(os.getcwd())}')
print(f'current file: {__file__}')

# Check if GPU is available and use it
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Set print options for 2 decimal places
torch.set_printoptions(precision=2, sci_mode=False)

seed = 42  # Set any integer seed
np.random.seed(seed)

torch.manual_seed(seed)  # CPU
torch.cuda.manual_seed(seed)  # GPU (if available)
torch.cuda.manual_seed_all(seed)  # Multi-GPU

# Ensures deterministic behavior in CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Print level
VERBOSE = 10


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedCNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--labeling_rate', type=float, required=False, default=5,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-s', '--server_epochs', type=int, required=False, default=5,
                        help="The number of server epochs (integer).")
    parser.add_argument('-n', '--num_clients', type=int, required=False, default=6,
                        help="The number of total clients.")
    parser.add_argument('-a', '--aggregation_method', type=str, required=False, default='median_avg',
                        help="aggregation method.")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
LABELING_RATE = 0.8
BIG_NUMBER = int(args.labeling_rate)
# SERVER_EPOCHS = args.server_epochs
SERVER_EPOCHS = args.server_epochs
IID_CLASSES_CNT = 5
NUM_CLIENTS = args.num_clients
NUM_MALICIOUS_CLIENTS = (NUM_CLIENTS - 2) // 2 - 1  # 2 + 2f < n for Krum, so f < (n-2)/2, not equal to (n-2)/2
# ns = [5, 10, 20, 50, 100, 1000], f = [(n-2)//2-1 for n in ns]=[0, 3, 8, 23, 48, 498], n-f=[5, 7, 12, 27, 52, 502]
NUM_HONEST_CLIENTS = NUM_CLIENTS - NUM_MALICIOUS_CLIENTS  # n - f
AGGREGATION_METHOD = args.aggregation_method
# aggregation_method = 'mean'  # refined_krum, krum, median, mean
print(args)
print(f'NUM_CLIENTS: {NUM_CLIENTS}, in which NUM_HONEST_CLIENTS: {NUM_HONEST_CLIENTS} and '
      f'NUM_MALICIOUS_CLIENTS: {NUM_MALICIOUS_CLIENTS}')

from base import *


@timer
def gen_client_data(data_dir='data/MNIST/clients', out_dir='.'):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    train_dataset = datasets.MNIST(root="./data", train=True, transform=None, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)
    X_test = test_dataset.data
    y_test = test_dataset.targets
    mask = np.full(len(y_test), False)
    for l in LABELS:
        mask_ = y_test == l
        mask[mask_] = True
    X_test, y_test = X_test[mask], y_test[mask]
    # preprocessing X_test
    X_test = normalize(X_test.numpy())
    y_test = y_test.numpy()
    shared_data = {"X": torch.tensor(X_test).float().to(DEVICE), 'y': torch.tensor(y_test).to(DEVICE)}

    X = train_dataset.data  # Tensor of shape (60000, 28, 28)
    y = train_dataset.targets  # Tensor of shape (60000,)
    mask = np.full(len(y), False)
    for l in LABELS:
        mask_ = y == l
        mask[mask_] = True
    X, y = X[mask], y[mask]
    X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()
    num_samples = len(y)

    random_state = 42
    torch.manual_seed(random_state)
    indices = torch.randperm(num_samples)  # Randomly shuffle
    # step = int(num_samples / NUM_HONEST_CLIENTS)
    # step = 50  # for debugging
    # non_iid_cnt0 = 0  # # make sure that non_iid_cnt is always less than iid_cnt
    # non_iid_cnt1 = 0

    Xs, Ys = dirichlet_split(X, y, num_clients=NUM_CLIENTS, alpha=1.0)
    # Xs, Ys = [X[:]] * NUM_CLIENTS, [y[:]]*NUM_CLIENTS   # if each client has all the data
    print([collections.Counter(y_) for y_ in Ys])
    ########################################### Benign Clients #############################################
    for c in range(NUM_HONEST_CLIENTS):
        client_type = 'honest'
        print(f"\n*** client_{c}: {client_type}...")
        # X_c = X[indices[c * step:(c + 1) * step]]
        # y_c = y[indices[c * step:(c + 1) * step]]
        # np.random.seed(c)  # change seed
        # # if c % 4 == 0 and non_iid_cnt0 < NUM_HONEST_CLIENTS // 4:  # 1/4 of honest clients has part of classes
        # # if c == 0:
        # #     non_iid_cnt0 += 1  # make sure that non_iid_cnt is always less than iid_cnt
        # #     mask_c = np.full(len(y_c), False)
        # #     # for l in [0, 1, 2, 3, 4]:
        # #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT, replace=False):
        # #         mask_ = y_c == l
        # #         mask_c[mask_] = True
        # #     # mask_c = (y_c != (c%10))  # excluding one class for each client
        # # elif c == 1:
        # #     # elif c % 4 == 1 and non_iid_cnt1 < NUM_HONEST_CLIENTS // 4:  # 1/4 of honest clients has part of classes
        # #     non_iid_cnt1 += 1
        # #     mask_c = np.full(len(y_c), False)
        # #     # for l in [5, 6, 7, 8, 9]:
        # #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT*2, replace=False):
        # #         mask_ = y_c == l
        # #         mask_c[mask_] = True
        # if c < NUM_HONEST_CLIENTS // 3:
        #     # elif c % 4 == 1 and non_iid_cnt1 < NUM_HONEST_CLIENTS // 4:  # 1/4 of honest clients has part of classes
        #     non_iid_cnt1 += 1
        #     mask_c = np.full(len(y_c), False)
        #     # for l in [5, 6, 7, 8, 9]:
        #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT * 2, replace=False):
        #         mask_ = y_c == l
        #         mask_c[mask_] = True
        # else:  # 2/4 of honest clients has IID distributions
        #     mask_c = np.full(len(y_c), True)
        # X_c = X_c[mask_c]
        # y_c = y_c[mask_c]

        X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution

        # might be used in server
        # train_info = {"client_type": client_type, "cnn": {}, 'client_id': c}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=1 - LABELING_RATE,
                                                       shuffle=True, random_state=random_state)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                      random_state=random_state)
        train_mask = np.full(num_samples_client, False)
        val_mask = np.full(num_samples_client, False)
        test_mask = np.full(num_samples_client, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).float().to(DEVICE), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        print(f'client_{c} data:', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)

    ########################################### Byzantine Clients #############################################
    indices = torch.randperm(num_samples)  # Randomly shuffle
    for c in range(NUM_HONEST_CLIENTS, NUM_HONEST_CLIENTS + NUM_MALICIOUS_CLIENTS, 1):
        client_type = 'attacker'
        print(f"\n*** client_{c}: {client_type}...")
        # X_c = X[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]
        # y_c = y[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]

        X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution

        mask_c = np.full(len(y_c), False)
        for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=BIG_NUMBER, replace=False):
            mask_ = y_c == l
            mask_c[mask_] = True
        X_c = X_c[mask_c]
        y_c = y_c[mask_c]

        # might be used in server
        # train_info = {"client_type": client_type, "cnn": {}, 'client_id': c}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=1 - LABELING_RATE,
                                                       shuffle=True, random_state=random_state)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                      random_state=random_state)
        train_mask = np.full(num_samples_client, False)
        val_mask = np.full(num_samples_client, False)
        test_mask = np.full(num_samples_client, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        # y_c[train_mask] = (NUM_CLASSES - 1) - y_c[train_mask]  # flip label
        # y_c[val_mask] = (NUM_CLASSES - 1) - y_c[val_mask]  # flip label
        y_c[train_mask] = torch.tensor([(NUM_CLASSES - 1) - v if v % 1 == 0 else v for v in y_c[train_mask]])
        y_c[val_mask] = torch.tensor([(NUM_CLASSES - 1) - v if v % 1 == 0 else v for v in y_c[val_mask]])  # flip label

        # train_info['NUM_MALICIOUS_CLIENTS'] = NUM_MALICIOUS_CLIENTS
        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).to(DEVICE).float(), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        print(f'client_{c} data:', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)


def clients_training(data_dir, epoch, global_cnn):
    clients_cnns = {}
    clients_info = {
        "NUM_CLASSES": NUM_CLASSES, "NUM_HONEST_CLIENTS": NUM_HONEST_CLIENTS,
        "NUM_MALICIOUS_CLIENTS": NUM_MALICIOUS_CLIENTS, "VERBOSE": VERBOSE,
        'DEVICE': DEVICE
    }  # extra information (e.g., number of samples) of clients that can be used in aggregation
    history = {}
    ########################################### Benign Clients #############################################
    for c in range(NUM_HONEST_CLIENTS):
        client_type = 'honest'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        # might be used in server
        train_info = {"client_type": client_type, "cnn": {}, 'client_id': c, 'server_epoch': epoch,
                      'DEVICE': DEVICE}

        data_file = f'{data_dir}/{c}.pth'
        with open(data_file, 'rb') as f:
            local_data = torch.load(f)
        num_samples_client = len(local_data['y'].tolist())
        label_cnts = collections.Counter(local_data['y'].tolist())
        clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
        print(f'client_{c} data:', label_cnts)
        print_data(local_data)

        print('Train CNN...')
        # local_cnn = CNN(input_dim=input_dim, hidden_dim=hidden_dim_cnn, output_dim=num_classes)
        local_cnn = CNN(num_classes=NUM_CLASSES)
        train_cnn(local_cnn, global_cnn, local_data, train_info)
        # w = w0 - \eta * \namba_w, so delta_w = w0 - w
        delta_w = {key: global_cnn.state_dict()[key] - local_cnn.state_dict()[key] for key in global_cnn.state_dict()}
        clients_cnns[c] = delta_w
        delta_dist = sum([torch.norm(local_cnn.state_dict()[key].cpu() - global_cnn.state_dict()[key].cpu()) for key
                          in global_cnn.state_dict()])
        print(f'dist(local, global): {delta_dist}')

        print('Evaluate CNNs...')
        evaluate(local_cnn, local_data, global_cnn,
                 test_type='Client data', client_id=c, train_info=train_info)
        evaluate_shared_test(local_cnn, local_data, global_cnn,
                             test_type='Shared test data', client_id=c, train_info=train_info)

        history[c] = train_info

    ########################################### Byzantine Clients #############################################
    for c in range(NUM_HONEST_CLIENTS, NUM_HONEST_CLIENTS + NUM_MALICIOUS_CLIENTS, 1):
        client_type = 'attacker'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        # might be used in server
        train_info = {"client_type": client_type, "cnn": {}, 'client_id': c, 'server_epoch': epoch,
                      'DEVICE': DEVICE}

        data_file = f'{data_dir}/{c}.pth'
        with open(data_file, 'rb') as f:
            local_data = torch.load(f)
        num_samples_client = len(local_data['y'].tolist())
        label_cnts = collections.Counter(local_data['y'].tolist())
        clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
        print(f'client_{c} data:', label_cnts)
        print_data(local_data)

        local_cnn = CNN(num_classes=NUM_CLASSES).to(DEVICE)
        # byzantine_method = 'adaptive_large_value'
        # if byzantine_method == 'last_global_model':
        #     local_cnn.load_state_dict(global_cnn.state_dict())
        # elif byzantine_method == 'flip_sign':
        #     local_cnn.load_state_dict(-1 * global_cnn.state_dict())
        # elif byzantine_method == 'mean':  # assign mean to each parameter
        #     for param in local_cnn.parameters():
        #         param.data = param.data / 2  # not work
        # elif byzantine_method == 'zero':  # assign 0 to each parameter
        #     for param in local_cnn.parameters():
        #         param.data.fill_(0.0)  # Assign big number to each parameter
        # elif byzantine_method == 'adaptive_large_value':
        #     new_state_dict = {}
        #     for key, param in global_cnn.state_dict().items():
        #         new_state_dict[key] = param * BIG_NUMBER
        #     local_cnn.load_state_dict(new_state_dict)
        # else:  # assign large values
        #     # Assign fixed large values to all parameters
        #     # BIG_NUMBER = 1.0  # if epoch % 5 == 0 else -1e3  # Example: Set all weights and biases to 1,000,000
        #     for param in local_cnn.parameters():
        #         param.data.fill_(BIG_NUMBER)  # Assign big number to each parameter
        train_cnn(local_cnn, global_cnn, local_data, train_info)
        delta_w = {key: (global_cnn.state_dict()[key] - local_cnn.state_dict()[key]) for key
                   in global_cnn.state_dict()}
        clients_cnns[c] = delta_w

        print('Evaluate CNNs...')
        evaluate(local_cnn, local_data, global_cnn,
                 test_type='Client data', client_id=c, train_info=train_info)
        evaluate_shared_test(local_cnn, local_data, global_cnn,
                             test_type='Shared test data', client_id=c, train_info=train_info)

        history[c] = train_info

    return clients_cnns, clients_info, history


@timer
def main():
    print(f"\n*************************** Generate Clients Data ******************************")
    data_dir = (f'data/MNIST/label_flipping/h_{NUM_HONEST_CLIENTS}-b_{NUM_MALICIOUS_CLIENTS}'
                f'-{IID_CLASSES_CNT}-{LABELING_RATE}-{BIG_NUMBER}-{AGGREGATION_METHOD}')
    data_out_dir = data_dir
    # data_out_dir = f'/projects/kunyang/nvflare_py31012/nvflare/{data_dir}'
    gen_client_data(data_dir, data_out_dir)

    print(f"\n***************************** Global Models *************************************")
    global_cnn = CNN(num_classes=NUM_CLASSES)
    global_cnn = global_cnn.to(DEVICE)
    print(global_cnn)

    histories = {'clients': [], 'server': [],
                 "IN_DIR": data_out_dir, "AGGREGATION_METHOD": AGGREGATION_METHOD,
                 "LABELING_RATE": LABELING_RATE, "SERVER_EPOCHS": SERVER_EPOCHS,
                 "NUM_CLASSES": NUM_CLASSES, "NUM_HONEST_CLIENTS": NUM_HONEST_CLIENTS,
                 "NUM_MALICIOUS_CLIENTS": NUM_MALICIOUS_CLIENTS, "VERBOSE": VERBOSE,
                 'DEVICE': DEVICE
                 }
    for server_epoch in range(SERVER_EPOCHS):
        print(f"\n*************** Server Epoch: {server_epoch}/{SERVER_EPOCHS}, Client Training *****************")
        clients_cnns, clients_info, history = clients_training(data_out_dir, server_epoch, global_cnn)
        histories['clients'].append(history)

        print(f"\n*************** Server Epoch: {server_epoch}/{SERVER_EPOCHS}, Server Aggregation **************")
        aggregate_cnns(clients_cnns, clients_info, global_cnn, AGGREGATION_METHOD, histories, server_epoch)

    prefix = f'-n_{SERVER_EPOCHS}'
    history_file = f'{IN_DIR}/histories_{prefix}.pth'
    print(f'saving histories to {history_file}')
    with open(history_file, 'wb') as f:
        pickle.dump(histories, f)
    torch.save(histories, history_file)

    try:
        print_histories(histories)
    except Exception as e:
        print('Exception: ', e)
    # print_histories_server(histories['server'])

    # Delete all the generated data
    shutil.rmtree(data_out_dir)


if __name__ == '__main__':
    IN_DIR = 'fl/mnist'
    LABELS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    # LABELS = {0, 1}
    NUM_CLASSES = len(LABELS)
    print(f'IN_DIR: {IN_DIR}, AGGREGATION_METHOD: {AGGREGATION_METHOD}, LABELING_RATE: {LABELING_RATE}, '
          f'NUM_HONEST_CLIENTS: {NUM_HONEST_CLIENTS}, NUM_MALICIOUS_CLIENTS: {NUM_MALICIOUS_CLIENTS}, '
          f'NUM_CLASSES: {NUM_CLASSES}, where classes: {LABELS}')
    main()
