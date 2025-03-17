"""
1. HPC Instructions:
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    # $module load conda
    # $conda activate nvflare-3.10
    # $cd nvflare/auto_labeling
    $module load conda && conda activate nvflare-3.10 && cd nvflare/auto_labeling
    $PYTHONPATH=. python3 nsf/fl_cnn_robust_aggregation_random_noise_model.py

    Storage path: /projects/kunyang/nvflare_py31012/nvflare

2. Data distributions for federated learning
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

Author: kun88.yang@gmail.com
"""
import argparse
import pickle
import shutil
import random

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets
from dataclasses import dataclass, field

from base import *
from utils import dirichlet_split

print(f'current directory: {os.path.abspath(os.getcwd())}')
print(f'current file: {__file__}')

# Set print options for 2 decimal places
torch.set_printoptions(precision=2, sci_mode=False)

# Set random seed for reproducibility
SEED = 42
# Set a random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)  # CPU
torch.cuda.manual_seed(SEED)  # GPU (if available)
torch.cuda.manual_seed_all(SEED)  # Multi-GPU

# Ensure deterministic behavior in CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if GPU is available and use it
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedCNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--labeling_rate', type=float, required=False, default=100,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-s', '--server_epochs', type=int, required=False, default=2,
                        help="The number of server epochs (integer).")
    parser.add_argument('-n', '--num_clients', type=int, required=False, default=5,
                        help="The number of total clients.")
    parser.add_argument('-a', '--aggregation_method', type=str, required=False, default='krum+rp',
                        help="aggregation method.")
    parser.add_argument('-v', '--verbose', type=int, required=False, default=10,
                        help="verbose mode.")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


@dataclass
class CONFIG:
    def __init__(self):
        self.SEED = None
        self.TRAIN_VAL_SEED = 42
        self.DEVICE = None
        self.VERBOSE = None
        self.LABELING_RATE = None
        self.BIG_NUMBER = None
        self.SERVER_EPOCHS = None
        self.CLIENT_EPOCHS = 20
        self.BATCH_SIZE = 512
        self.IID_CLASSES_CNT = 5
        self.NUM_CLIENTS = None
        self.NUM_BYZANTINE_CLIENTS = None
        self.NUM_HONEST_CLIENTS = None
        self.AGGREGATION_METHOD = None
        self.LABELS = set()
        self.NUM_CLASSES = None

    def __str__(self):
        return str(self.__dict__)  # Prints attributes as a dictionary

    def __repr__(self):
        return f"CONFIG({self.__dict__})"  # More detailed representation


def get_configuration(train_val_seed):
    # Parse command-line arguments
    args = parse_arguments()

    CFG = CONFIG()
    CFG.SEED = SEED  # Control data split across clients, such as NonIID
    CFG.TRAIN_VAL_SEED = train_val_seed  # Control the train and val data split on each client
    CFG.DEVICE = DEVICE
    # Print level
    CFG.VERBOSE = args.verbose
    CFG.LABELING_RATE = 0.8
    CFG.BIG_NUMBER = args.labeling_rate
    CFG.BATCH_SIZE = int(CFG.BIG_NUMBER)
    CFG.SERVER_EPOCHS = args.server_epochs
    # CFG.IID_CLASSES_CNT = 5
    CFG.NUM_CLIENTS = args.num_clients
    # 2 + 2f < n for Krum, so f < (n-2)/2, not equal to (n-2)/2
    if CFG.NUM_CLIENTS < 5:  # if n == 4, f will be 0
        raise ValueError(f"NUM_CLIENTS ({CFG.NUM_CLIENTS}) must be >= 5 as we require 2 + 2f < n.")
    CFG.NUM_BYZANTINE_CLIENTS = int((CFG.NUM_CLIENTS - 3) / 2)
    if 2 + 2 * CFG.NUM_BYZANTINE_CLIENTS == CFG.NUM_CLIENTS:  # 2 + 2f < n
        CFG.NUM_BYZANTINE_CLIENTS -= 1
    CFG.NUM_HONEST_CLIENTS = CFG.NUM_CLIENTS - CFG.NUM_BYZANTINE_CLIENTS  # n - f
    CFG.AGGREGATION_METHOD = args.aggregation_method  # adaptive_krum, krum, median, mean
    print(args)
    print(f'NUM_CLIENTS: {CFG.NUM_CLIENTS}, in which NUM_HONEST_CLIENTS: {CFG.NUM_HONEST_CLIENTS} and '
          f'NUM_BYZANTINE_CLIENTS: {CFG.NUM_BYZANTINE_CLIENTS}')

    CFG.LABELS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    CFG.NUM_CLASSES = len(CFG.LABELS)
    print(CFG)
    return CFG


@timer
def gen_client_data(data_dir='data/MNIST/clients', out_dir='.', CFG=None):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    train_dataset = datasets.MNIST(root="./data", train=True, transform=None, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)
    X_test = test_dataset.data
    y_test = test_dataset.targets
    mask = np.full(len(y_test), False)
    for l in CFG.LABELS:
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
    for l in CFG.LABELS:
        mask_ = y == l
        mask[mask_] = True
    X, y = X[mask], y[mask]
    X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()
    num_samples = len(y)

    # random_state = 42
    # torch.manual_seed(random_state)
    # indices = torch.randperm(num_samples)  # Randomly shuffle
    # step = int(num_samples / NUM_HONEST_CLIENTS)
    # step = 50  # for debugging
    # non_iid_cnt0 = 0  # # make sure that non_iid_cnt is always less than iid_cnt
    # non_iid_cnt1 = 0
    # m = len(y)//2

    # Xs1, Ys1 = dirichlet_split(X[:m, :], y[:m], num_clients=CFG.NUM_CLIENTS // 2, alpha=0.5, random_state=SEED)
    # Xs2, Ys2 = dirichlet_split(X[m:], y[m:], num_clients=CFG.NUM_CLIENTS - CFG.NUM_CLIENTS // 2, alpha=100,
    #                            random_state=SEED)
    # Xs, Ys = Xs1 + Xs2, Ys1 + Ys2

    Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=10, random_state=SEED)
    # Shuffle the lists X and y in the same order
    combined = list(zip(Xs, Ys))  # Combine the lists into pairs of (X[i], y[i])
    random.shuffle(combined)  # Shuffle the combined list
    # Unzip the shuffled list back into X and y
    Xs, Ys = zip(*combined)
    # Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.BIG_NUMBER, random_state=SEED)
    # Xs, Ys = [X[:]] * NUM_CLIENTS, [y[:]]*NUM_CLIENTS   # if each client has all the data
    total_size = 0
    for j, y_ in enumerate(Ys):
        vs = collections.Counter(y_.tolist())
        vs = dict(sorted(vs.items(), key=lambda x: x[0], reverse=False))
        print(f"client {j}'s data size: {len(y_)}, total classes: {len(vs)}, in which {vs}")
        total_size += len(y_)
    print(f"total size: {total_size}")
    ########################################### Benign Clients #############################################
    for c in range(CFG.NUM_HONEST_CLIENTS):
        client_type = 'Honest'
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
        #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT, replace=False):
        #         mask_ = y_c == l
        #         mask_c[mask_] = True
        # if c <= NUM_HONEST_CLIENTS//BIG_NUMBER:
        #     mask_c = np.full(len(y_c), False)
        #     # for l in [5, 6, 7, 8, 9]:
        #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT, replace=False):
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
        train_indices, test_indices = train_test_split(indices_sub, test_size=1 - CFG.LABELING_RATE,
                                                       shuffle=True, random_state=SEED)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                      random_state=CFG.TRAIN_VAL_SEED)
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
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        print(f'client_{c} data ({len(label_cnts)}): ', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)

    ########################################### Byzantine Clients #############################################
    indices = torch.randperm(num_samples)  # Randomly shuffle
    for c in range(CFG.NUM_HONEST_CLIENTS, CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'Byzantine'
        print(f"\n*** client_{c}: {client_type}...")
        # X_c = X[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]
        # y_c = y[indices[(c - NUM_HONEST_CLIENTS) * step:((c - NUM_HONEST_CLIENTS) + 1) * step]]

        X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution

        # mask_c = np.full(len(y_c), False)
        # for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=CFG.BIG_NUMBER, replace=False):
        #     mask_ = y_c == l
        #     mask_c[mask_] = True
        # X_c = X_c[mask_c]
        # y_c = y_c[mask_c]

        # might be used in server
        # train_info = {"client_type": client_type, "cnn": {}, 'client_id': c}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=1 - CFG.LABELING_RATE,
                                                       shuffle=True, random_state=SEED)  # test set unchanged
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                      random_state=CFG.TRAIN_VAL_SEED)
        train_mask = np.full(num_samples_client, False)
        val_mask = np.full(num_samples_client, False)
        test_mask = np.full(num_samples_client, False)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        # y_c[train_mask] = (NUM_CLASSES - 1) - y_c[train_mask]  # flip label
        # y_c[val_mask] = (NUM_CLASSES - 1) - y_c[val_mask]  # flip label

        # train_info['NUM_BYZANTINE_CLIENTS'] = NUM_BYZANTINE_CLIENTS
        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).to(DEVICE).float(), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        print(f'client_{c} data ({len(label_cnts)}): ', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)


# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, std=0.2):
    noise = torch.randn(image.size()) * std + mean  # Generate Gaussian noise
    noisy_image = image + noise  # Add noise
    noisy_image = torch.clamp(noisy_image, 0, 1)  # Keep pixel values in [0,1]
    return noisy_image


def add_salt_and_pepper_noise(image, prob=0.02):
    np_image = image.numpy()  # Convert to NumPy array
    noisy_image = np_image.copy()

    # Generate mask for salt and pepper noise
    salt_pepper = np.random.rand(*np_image.shape)
    noisy_image[salt_pepper < prob / 2] = 0  # Black pixels
    noisy_image[salt_pepper > 1 - prob / 2] = 1  # White pixels

    return torch.tensor(noisy_image)


def add_speckle_noise(image, std=0.2):
    noise = torch.randn(image.size()) * image * std  # Noise proportional to pixel values
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)


def clients_training(data_dir, epoch, global_cnn, CFG):
    clients_cnns = {}
    clients_info = {}  # extra information (e.g., number of samples) of clients that can be used in aggregation
    history = {}
    ########################################## Benign Clients #############################################
    for c in range(CFG.NUM_HONEST_CLIENTS):
        client_type = 'Honest'
        print(f"\n*** server_epoch:{epoch}, client_{c}: {client_type}... ***")
        # might be used in server
        train_info = {"client_type": client_type, "cnn": {}, 'client_id': c, 'server_epoch': epoch,
                      'DEVICE': DEVICE, 'CFG': CFG}

        data_file = f'{data_dir}/{c}.pth'
        with open(data_file, 'rb') as f:
            local_data = torch.load(f)
        num_samples_client = len(local_data['y'].tolist())
        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
        print(f'client_{c} data ({len(label_cnts)}):', label_cnts)
        print_data(local_data)

        print('Train CNN...')
        # local_cnn = CNN(input_dim=input_dim, hidden_dim=hidden_dim_cnn, output_dim=num_classes)
        local_cnn = CNN(num_classes=CFG.NUM_CLASSES)
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
    for c in range(CFG.NUM_HONEST_CLIENTS, CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'Byzantine'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        # might be used in server
        train_info = {"client_type": client_type, "cnn": {}, 'client_id': c, 'server_epoch': epoch,
                      'DEVICE': DEVICE, 'CFG': CFG}

        data_file = f'{data_dir}/{c}.pth'
        with open(data_file, 'rb') as f:
            local_data = torch.load(f)
        num_samples_client = len(local_data['y'].tolist())
        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
        print(f'client_{c} data:', label_cnts)
        print_data(local_data)

        local_cnn = CNN(num_classes=CFG.NUM_CLASSES).to(DEVICE)
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
        # train_cnn(local_cnn, global_cnn, local_data, train_info)

        # # Inject noise to malicious clients' CNNs
        # new_state_dict = {}
        # for key, param in global_cnn.state_dict().items():
        #     noise = torch.normal(0, CFG.BIG_NUMBER, size=param.shape).to(DEVICE)
        #     new_state_dict[key] = param + noise

        # only inject noise to partial dimensions of model parameters.
        model = CNN(num_classes=CFG.NUM_CLASSES)
        model.load_state_dict(global_cnn.state_dict())
        ps = parameters_to_vector(model.parameters()).detach().to(DEVICE)
        # # Randomly select the indices
        # cnt = max(1, int(CFG.BIG_NUMBER * len(ps)))
        # print(f'{cnt} parameters ({CFG.BIG_NUMBER*100}%) are changed.')

        # selected_indices = random.sample(range(len(ps)), cnt)
        # noise = torch.normal(0, 10, size=(cnt, )).to(DEVICE)
        # ps[selected_indices] = ps[selected_indices] + noise

        noise = torch.normal(0, 10, size=ps.shape).to(DEVICE)
        ps = ps + noise

        vector_to_parameters(ps, model.parameters())  # in_place
        new_state_dict = model.state_dict()

        local_cnn.load_state_dict(new_state_dict)
        # w = w0 - \eta * \namba_w, so delta_w = w0 - w, only send update difference to the server
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
    all_histories = {}
    NUM_REPEATS = 1
    for train_val_seed in range(0, 1000, 1000 // NUM_REPEATS):
        print('\n')
        CFG = get_configuration(train_val_seed)
        print(f"\n*************************** Generate Clients Data ******************************")
        data_dir = (f'data/MNIST/random_noise_model/h_{CFG.NUM_HONEST_CLIENTS}-b_{CFG.NUM_BYZANTINE_CLIENTS}'
                    f'-{CFG.IID_CLASSES_CNT}-{CFG.LABELING_RATE}-{CFG.BIG_NUMBER}-{CFG.AGGREGATION_METHOD}'
                    f'/{CFG.TRAIN_VAL_SEED}')
        data_out_dir = data_dir
        data_out_dir = f'/projects/kunyang/nvflare_py31012/nvflare/{data_dir}'
        CFG.data_out_dir = data_out_dir
        gen_client_data(data_dir, data_out_dir, CFG)

        print(f"\n***************************** Global Models *************************************")
        global_cnn = CNN(num_classes=CFG.NUM_CLASSES)
        global_cnn = global_cnn.to(DEVICE)
        print(global_cnn)

        histories = {'clients': [], 'server': [], 'CFG': CFG}
        for server_epoch in range(CFG.SERVER_EPOCHS):
            print(f"\n*************** Server Epoch: {server_epoch}/{CFG.SERVER_EPOCHS}, Client Training *************")
            clients_cnns, clients_info, history = clients_training(data_out_dir, server_epoch, global_cnn, CFG)
            histories['clients'].append(history)

            print(f"\n*************** Server Epoch: {server_epoch}/{CFG.SERVER_EPOCHS}, Server Aggregation **********")
            aggregate_cnns(clients_cnns, clients_info, global_cnn, CFG.AGGREGATION_METHOD, histories, server_epoch)

        prefix = f'-n_{CFG.SERVER_EPOCHS}'
        history_file = f'{CFG.data_out_dir}/histories_{prefix}.pth'
        print(f'saving histories to {history_file}')
        # with open(history_file, 'wb') as f:
        #     pickle.dump(histories, f)
        torch.save(histories, history_file)

        try:
            print_histories(histories)
        except Exception as e:
            print('Exception: ', e)
        # print_histories_server(histories['server'])

        # Delete all the generated data
        shutil.rmtree(data_out_dir)

        # save all results
        all_histories[CFG.TRAIN_VAL_SEED] = histories
        history_file = f'{os.path.dirname(CFG.data_out_dir)}/all_histories_{NUM_REPEATS}.pth'
        print(f'saving all histories to {history_file}')
        # with open(history_file, 'wb') as f:
        #     pickle.dump(all_histories, f)
        torch.save(all_histories, history_file)

    # history_file = 'data/MNIST/sign_flipping/h_12-b_8-5-0.8-0.1-krum_avg/all_histories_3.pth'
    # history_file = 'all_histories_5.pth'
    # all_histories = torch.load(history_file)
    print_all(all_histories)


if __name__ == '__main__':
    main()
