"""
1. HPC Instructions:
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    # $module load conda
    # $conda activate nvflare-3.10
    # $cd nvflare/auto_labeling
    $module load conda && conda activate nvflare-3.10 && cd nvflare/auto_labeling
    $PYTHONPATH=. python3 fl_cnn_robust_aggregation_label_flipping.py
    $PYTHONPATH=.:nsf python3 nsf/fl_cnn_robust_aggregation_label_flipping.py

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
import gzip
import shutil
import random
import traceback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import pandas as pd

import ragg
from ragg.base import *
from ragg.utils import dirichlet_split

from functools import partial

reduce_dim_flg = False
# Conditional model selection using partial
if reduce_dim_flg:
    reduced_dim = 10
    # Define a partial function for FNN
    FNN = partial(ragg.base.FNN, input_dim=reduced_dim)
else:
    FNN = ragg.base.FNN
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
    parser = argparse.ArgumentParser(description="FedFNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--labeling_rate', type=float, required=False, default=100,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-s', '--server_epochs', type=int, required=False, default=2,
                        help="The number of server epochs (integer).")
    parser.add_argument('-n', '--num_clients', type=int, required=False, default=5,
                        help="The number of total clients.")
    parser.add_argument('-a', '--aggregation_method', type=str, required=False, default='krum_avg',
                        help="aggregation method.")
    parser.add_argument('-v', '--verbose', type=int, required=False, default=5,
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
        self.CLIENT_EPOCHS = 5
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
    # CFG.BATCH_SIZE = int(CFG.BIG_NUMBER)
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

    CFG.LABELS = {0, 1}
    CFG.NUM_CLASSES = len(CFG.LABELS)
    CFG.CNN = FNN
    print(CFG)
    return CFG

#
#
# class FNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(FNN, self).__init__()
#         # self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # From 1 channel to 16 channels
#         # self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
#         #
#         # self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
#         # self.conv21 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         #
#         # self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
#         # self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         #
#         # self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Adjust the dimensions after the convolution layers
#         # self.fc2 = nn.Linear(512, num_classes)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#
#         # # self.sigmoid = nn.Sigmoid()
#         #
#         # self.transform = nn.Sequential(
#         #     nn.Linear(28 * 28 + num_classes, 784),
#         #     nn.LeakyReLU(0.2),
#         # )
#
#         self.fc11 = nn.Linear(768, 32)
#         self.fc21 = nn.Linear(32, 16)
#         self.fc22 = nn.Linear(16, 8)
#         self.fc33 = nn.Linear(8, num_classes)
#
#     def forward(self, x):
#         model_type = 'mlp'
#
#         x = self.leaky_relu(self.fc11(x))
#         x = self.leaky_relu(self.fc21(x))
#         x = self.leaky_relu(self.fc22(x))
#         x = self.fc33(x)
#
#         return x
#


# @timer
# def aggregate_cnns(clients_cnns, clients_info, global_cnn, aggregation_method, histories, epoch):
#     print('*aggregate cnn...')
#     CFG = histories['CFG']
#     # flatten all the parameters into a long vector
#     # clients_updates = [client_state_dict.cpu() for client_state_dict in clients_cnns.values()]
#
#     # Concatenate all parameter tensors into one vector.
#     # Note: The order here is the iteration order of the OrderedDict, which
#     # may not match the order of model.parameters().
#     # vector_from_state = torch.cat([param.view(-1) for param in state.values()])
#     # flatten_clients_updates = [torch.cat([param.view(-1).cpu() for param in client_state_dict.values()]) for
#     #                            client_state_dict in clients_cnns.values()]
#     tmp_models = []
#     for client_state_dict in clients_cnns.values():
#         model = FNN(num_classes=CFG.NUM_CLASSES)
#         model.load_state_dict(client_state_dict)
#         tmp_models.append(model)
#     flatten_clients_updates = [parameters_to_vector(md.parameters()).detach().cpu() for md in tmp_models]
#
#     # for v in flatten_clients_updates:
#     #     # print(v.tolist())
#     #     print_histgram(v, bins=10, value_type='update')
#
#     flatten_clients_updates = torch.stack(flatten_clients_updates)
#     print(f'each update shape: {flatten_clients_updates[1].shape}')
#     # for debugging
#     if CFG.VERBOSE >= 30:
#         for i, update in enumerate(flatten_clients_updates):
#             print(f'client_{i}:', end='  ')
#             print_histgram(update, bins=5, value_type='params')
#
#     min_value = min([torch.min(v).item() for v in flatten_clients_updates[: CFG.NUM_HONEST_CLIENTS]])
#     max_value = max([torch.max(v).item() for v in flatten_clients_updates[: CFG.NUM_HONEST_CLIENTS]])
#
#     # each client extra information (such as, number of samples)
#     # client_weights will affect median and krum, so be careful to weights
#     # if assign byzantine clients with very large weights (e.g., 1e6),
#     # then median will choose byzantine client's parameters.
#     clients_weights = torch.tensor([1] * len(flatten_clients_updates))  # default as 1
#     # clients_weights = torch.tensor([vs['size'] for vs in clients_info.values()])
#     start = time.time()
#     if aggregation_method == 'adaptive_krum':
#         aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(flatten_clients_updates,
#                                                                                 clients_weights,
#                                                                                 trimmed_average=False,
#                                                                                 random_projection=False,
#                                                                                 verbose=CFG.VERBOSE)
#     elif aggregation_method == 'adaptive_krum_avg':
#         aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(flatten_clients_updates,
#                                                                                 clients_weights,
#                                                                                 trimmed_average=True,
#                                                                                 random_projection=False,
#                                                                                 verbose=CFG.VERBOSE)
#
#     elif aggregation_method == 'adaptive_krum+rp':  # adaptive_krum + random projection
#         aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(
#             flatten_clients_updates, clients_weights, trimmed_average=False, random_projection=True,
#             random_state=CFG.TRAIN_VAL_SEED, verbose=CFG.VERBOSE)
#     elif aggregation_method == 'adaptive_krum+rp_avg':  # adaptive_krum + random projection
#         aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(
#             flatten_clients_updates, clients_weights, trimmed_average=True, random_projection=True,
#             random_state=CFG.TRAIN_VAL_SEED, verbose=CFG.VERBOSE)
#     elif aggregation_method == 'krum':
#         # train_info = list(histories['clients'][-1].values())[-1]
#         # f = train_info['NUM_BYZANTINE_CLIENTS']
#         f = CFG.NUM_BYZANTINE_CLIENTS
#         # client_type = train_info['client_type']
#         aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
#                                                                        trimmed_average=False, random_projection=False,
#                                                                        verbose=CFG.VERBOSE)
#     elif aggregation_method == 'krum_avg':
#         # train_info = list(histories['clients'][-1].values())[-1]
#         # f = train_info['NUM_BYZANTINE_CLIENTS']
#         f = CFG.NUM_BYZANTINE_CLIENTS
#         # client_type = train_info['client_type']
#         aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
#                                                                        trimmed_average=True, random_projection=False,
#                                                                        verbose=CFG.VERBOSE)
#     elif aggregation_method == 'krum+rp':
#         # train_info = list(histories['clients'][-1].values())[-1]
#         # f = train_info['NUM_BYZANTINE_CLIENTS']
#         f = CFG.NUM_BYZANTINE_CLIENTS
#         # client_type = train_info['client_type']
#         aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates,
#                                                                        clients_weights, f,
#                                                                        trimmed_average=False,
#                                                                        random_projection=True,
#                                                                        random_state=CFG.TRAIN_VAL_SEED,
#                                                                        verbose=CFG.VERBOSE)
#     elif aggregation_method == 'krum+rp_avg':
#         # train_info = list(histories['clients'][-1].values())[-1]
#         # f = train_info['NUM_BYZANTINE_CLIENTS']
#         f = CFG.NUM_BYZANTINE_CLIENTS
#         # client_type = train_info['client_type']
#         aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates,
#                                                                        clients_weights, f,
#                                                                        trimmed_average=True,
#                                                                        random_projection=True,
#                                                                        random_state=CFG.TRAIN_VAL_SEED,
#                                                                        verbose=CFG.VERBOSE)
#     elif aggregation_method == 'median':
#         p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
#         p = p / 2  # top p/2 and bottom p/2 are removed
#         aggregated_update, clients_type_pred = robust_aggregation.cw_median(flatten_clients_updates, clients_weights,
#                                                                             verbose=CFG.VERBOSE)
#
#     elif aggregation_method == 'medoid':
#         p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
#         aggregated_update, clients_type_pred = robust_aggregation.medoid(flatten_clients_updates,
#                                                                          clients_weights,
#                                                                          trimmed_average=False,
#                                                                          upper_trimmed_ratio=p,
#                                                                          verbose=CFG.VERBOSE)
#
#     elif aggregation_method == 'medoid_avg':
#         p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
#         aggregated_update, clients_type_pred = robust_aggregation.medoid(flatten_clients_updates,
#                                                                          clients_weights,
#                                                                          trimmed_average=True,
#                                                                          upper_trimmed_ratio=p,
#                                                                          verbose=CFG.VERBOSE)
#
#     elif aggregation_method == 'geometric_median':
#         aggregated_update, clients_type_pred = robust_aggregation.geometric_median(flatten_clients_updates,
#                                                                                    clients_weights,
#                                                                                    max_iters=100, tol=1e-6,
#                                                                                    verbose=CFG.VERBOSE)
#
#     # elif aggregation_method == 'exp_weighted_mean':
#     #     clients_type_pred = None
#     #     aggregated_update = exp_weighted_mean.robust_center_exponential_reweighting_tensor(
#     #         torch.stack(flatten_clients_updates), x_est=flatten_clients_updates[-1],
#     #         r=0.1, max_iters=100, tol=1e-6, verbose=CFG.VERBOSE)
#     elif aggregation_method == 'trimmed_mean':
#         p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
#         p = p / 2  # top p/2 and bottom p/2 are removed
#         aggregated_update, clients_type_pred = robust_aggregation.trimmed_mean(flatten_clients_updates, clients_weights,
#                                                                                trim_ratio=p,
#                                                                                verbose=CFG.VERBOSE)
#     else:
#         # empirical mean
#         aggregated_update, clients_type_pred = robust_aggregation.cw_mean(flatten_clients_updates, clients_weights,
#                                                                           verbose=CFG.VERBOSE)
#     end = time.time()
#     time_taken = end - start
#     n = len(flatten_clients_updates)
#     f = CFG.NUM_BYZANTINE_CLIENTS
#     # # weight average
#     # update = 0.0
#     # weight = 0.0
#     # for j in range(n-f):  # note here is k+1 because we want to add all values before k+1
#     #     update += flatten_clients_updates[j] * clients_weights[j]
#     #     weight += clients_weights[j]
#     # empirical_mean = update / weight
#     empirical_mean = torch.sum(flatten_clients_updates[:n - f] *
#                                clients_weights[:n - f, None], dim=0) / torch.sum(clients_weights[:n - f])
#     l2_error = torch.norm(empirical_mean - aggregated_update, p=2).item()  # l2 norm
#     histories['server'].append({"time_taken": time_taken, 'l2_error': l2_error})
#     # f'clients_weights: {clients_weights.numpy()},
#     print(f'{aggregation_method}, clients_type: {clients_type_pred}, '
#           f'client_updates: min: {min_value:.2f}, max: {max_value:.2f}, '
#           f'time taken: {time_taken:.4f}s, l2_error: {l2_error:.2f}')
#
#     # Update the global model with the aggregated parameters
#     # w = w0 - (delta_w), where delta_w = \eta*\namba_w
#     aggregated_update = parameters_to_vector(global_cnn.parameters()).detach().cpu() - aggregated_update
#     aggregated_update = aggregated_update.to(CFG.DEVICE)
#     vector_to_parameters(aggregated_update, global_cnn.parameters())  # in_place
#     # global_cnn.load_state_dict(aggregated_update)


@timer
def gen_client_data(data_dir='data/Sentiment140', out_dir='.', CFG=None):
    os.makedirs(out_dir, exist_ok=True)

    # Total rows in the dataset
    total_rows = 1600000  # Replace with the actual number of rows in your CSV file
    # Number of rows to sample
    sample_size = int(0.1*total_rows)
    # Randomly choose rows to read
    skip = np.random.choice(total_rows, total_rows - sample_size, replace=False)
    # in_file = 'data/Sentiment140/training.1600000.processed.noemoticon.csv_bert.csv'
    in_file = 'data/Sentiment140/training.1600000.processed.noemoticon.csv_bert.csv_pca_100.csv'
    df = pd.read_csv(in_file, dtype=float, header=None, skiprows=skip.tolist())
    X, y = torch.tensor(df.iloc[:, 0:-1].values), torch.tensor(df.iloc[:, -1].values, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        shuffle=True, random_state=42)
    # Initialize scaler and fit ONLY on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mask = np.full(len(y_test), False)
    for l in CFG.LABELS:
        mask_ = y_test == l
        mask[mask_] = True
    X_test, y_test = X_test[mask], y_test[mask]
    # preprocessing X_test
    # X_test = normalize(X_test.numpy())
    y_test = y_test.numpy()
    shared_data = {"X": torch.tensor(X_test).float().to(DEVICE), 'y': torch.tensor(y_test).to(DEVICE)}

    X, y = X_train, y_train
    mask = np.full(len(y), False)
    for l in CFG.LABELS:
        mask_ = y == l
        mask[mask_] = True
    X, y = X[mask], y[mask]
    # X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()
    num_samples = len(y)
    dim = X.shape[1]

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

    Xs, Ys = dirichlet_split(X, y, num_clients=CFG.NUM_CLIENTS, alpha=CFG.BIG_NUMBER, random_state=SEED)
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
        #     for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=IID_CLASSES_CNT * 2, replace=False):
        #         mask_ = y_c == l
        #         mask_c[mask_] = True
        # else:  # 2/4 of honest clients has IID distributions
        #     mask_c = np.full(len(y_c), True)
        # X_c = X_c[mask_c]
        # y_c = y_c[mask_c]

        X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution

        # might be used in server
        # train_info = {"client_type": client_type, "FNN": {}, 'client_id': c}
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

        out_file = f'{out_dir}/{c}.pt.gz'
        # torch.save(local_data, out_file)
        with gzip.open(out_file, 'wb') as f:
            torch.save(local_data, f)

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

        # mask_c = np.full(len(y_c), False)
        # for l in np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=5, replace=False):
        #     mask_ = y_c == l
        #     mask_c[mask_] = True
        # y_c[mask_c] = (CFG.NUM_CLASSES - 1)-y_c[mask_c]     # # flip label

        # might be used in server
        # train_info = {"client_type": client_type, "FNN": {}, 'client_id': c}
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
        y_c[train_mask] = (CFG.NUM_CLASSES - 1) - y_c[train_mask]  # flip label
        y_c[val_mask] = (CFG.NUM_CLASSES - 1) - y_c[val_mask]  # flip label
        # y_c[train_mask] = torch.tensor([(CFG.NUM_CLASSES - 1) - v if v % 1 == 0 else v for v in y_c[train_mask]])
        # y_c[val_mask] = torch.tensor(
        #     [(CFG.NUM_CLASSES - 1) - v if v % 1 == 0 else v for v in y_c[val_mask]])  # flip label

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

        out_file = f'{out_dir}/{c}.pt.gz'
        # torch.save(local_data, out_file)
        with gzip.open(out_file, 'wb') as f:
            torch.save(local_data, f)

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


def clients_training(data_dir, epoch, global_fnn, CFG):
    clients_fnns = {}
    clients_info = {}  # extra information (e.g., number of samples) of clients that can be used in aggregation
    history = {}
    ########################################## Benign Clients #############################################
    for c in range(CFG.NUM_HONEST_CLIENTS):
        client_type = 'Honest'
        print(f"\n*** server_epoch:{epoch}, client_{c}: {client_type}... ***")
        # might be used in server
        train_info = {"client_type": client_type, "fnn": {}, 'client_id': c, 'server_epoch': epoch,
                      'DEVICE': DEVICE, 'CFG': CFG}

        data_file = f'{data_dir}/{c}.pt.gz'
        with gzip.open(data_file, 'rb') as f:
            local_data = torch.load(f)
        num_samples_client = len(local_data['y'].tolist())
        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
        print(f'client_{c} data ({len(label_cnts)}):', label_cnts)
        print_data(local_data)

        print('Train FNN...')
        # local_fnn = FNN(input_dim=input_dim, hidden_dim=hidden_dim_fnn, output_dim=num_classes)
        local_fnn = FNN(num_classes=CFG.NUM_CLASSES)
        train_cnn(local_fnn, global_fnn, local_data, train_info)
        # w = w0 - \eta * \namba_w, so delta_w = w0 - w
        delta_w = {key: global_fnn.state_dict()[key] - local_fnn.state_dict()[key] for key in global_fnn.state_dict()}
        clients_fnns[c] = delta_w
        delta_dist = sum([torch.norm(local_fnn.state_dict()[key].cpu() - global_fnn.state_dict()[key].cpu()) for key
                          in global_fnn.state_dict()])
        print(f'dist(local, global): {delta_dist}')

        print('Evaluate FNNs...')
        evaluate(local_fnn, local_data, global_fnn,
                 test_type='Client data', client_id=c, train_info=train_info)
        evaluate_shared_test(local_fnn, local_data, global_fnn,
                             test_type='Shared test data', client_id=c, train_info=train_info)

        history[c] = train_info

    ########################################### Byzantine Clients #############################################
    for c in range(CFG.NUM_HONEST_CLIENTS, CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'Byzantine'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        # might be used in server
        train_info = {"client_type": client_type, "fnn": {}, 'client_id': c, 'server_epoch': epoch,
                      'DEVICE': DEVICE, 'CFG': CFG}

        data_file = f'{data_dir}/{c}.pt.gz'
        with gzip.open(data_file, 'rb') as f:
            local_data = torch.load(f)
        num_samples_client = len(local_data['y'].tolist())
        label_cnts = collections.Counter(local_data['y'].tolist())
        label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
        clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
        print(f'client_{c} data:', label_cnts)
        print_data(local_data)

        # local_fnn = FNN(num_classes=CFG.NUM_CLASSES).to(DEVICE)
        # byzantine_method = 'adaptive_large_value'
        # if byzantine_method == 'last_global_model':
        #     local_fnn.load_state_dict(global_fnn.state_dict())
        # elif byzantine_method == 'flip_sign':
        #     local_fnn.load_state_dict(-1 * global_fnn.state_dict())
        # elif byzantine_method == 'mean':  # assign mean to each parameter
        #     for param in local_fnn.parameters():
        #         param.data = param.data / 2  # not work
        # elif byzantine_method == 'zero':  # assign 0 to each parameter
        #     for param in local_fnn.parameters():
        #         param.data.fill_(0.0)  # Assign big number to each parameter
        # elif byzantine_method == 'adaptive_large_value':
        #     new_state_dict = {}
        #     for key, param in global_fnn.state_dict().items():
        #         new_state_dict[key] = param * BIG_NUMBER
        #     local_fnn.load_state_dict(new_state_dict)
        # else:  # assign large values
        #     # Assign fixed large values to all parameters
        #     # BIG_NUMBER = 1.0  # if epoch % 5 == 0 else -1e3  # Example: Set all weights and biases to 1,000,000
        #     for param in local_fnn.parameters():
        #         param.data.fill_(BIG_NUMBER)  # Assign big number to each parameter
        # train_fnn(local_fnn, global_fnn, local_data, train_info)

        # # Inject noise to malicious clients' FNNs
        # new_state_dict = {}
        # for key, param in global_fnn.state_dict().items():
        #     noise = torch.normal(0, CFG.BIG_NUMBER, size=param.shape).to(DEVICE)
        #     new_state_dict[key] = param + noise

        # local_fnn = FNN(input_dim=input_dim, hidden_dim=hidden_dim_fnn, output_dim=num_classes)
        local_fnn = FNN(num_classes=CFG.NUM_CLASSES)
        train_cnn(local_fnn, global_fnn, local_data, train_info)
        # w = w0 - \eta * \namba_w, so delta_w = w0 - w
        delta_w = {key: global_fnn.state_dict()[key] - local_fnn.state_dict()[key] for key in global_fnn.state_dict()}
        clients_fnns[c] = delta_w
        delta_dist = sum([torch.norm(local_fnn.state_dict()[key].cpu() - global_fnn.state_dict()[key].cpu()) for key
                          in global_fnn.state_dict()])
        print(f'dist(local, global): {delta_dist}')

        print('Evaluate FNNs...')
        evaluate(local_fnn, local_data, global_fnn,
                 test_type='Client data', client_id=c, train_info=train_info)
        evaluate_shared_test(local_fnn, local_data, global_fnn,
                             test_type='Shared test data', client_id=c, train_info=train_info)

        history[c] = train_info

    return clients_fnns, clients_info, history


@timer
def main():
    try:
        all_histories = {}
        NUM_REPEATS = 1
        for train_val_seed in range(0, 1000, 1000 // NUM_REPEATS):
            print('\n')
            CFG = get_configuration(train_val_seed)
            print(f"\n*************************** Generate Clients Data ******************************")
            data_dir = (f'data/Sentiment140/label_flipping/h_{CFG.NUM_HONEST_CLIENTS}-b_{CFG.NUM_BYZANTINE_CLIENTS}'
                        f'-{CFG.IID_CLASSES_CNT}-{CFG.LABELING_RATE}-{CFG.BIG_NUMBER}-{CFG.AGGREGATION_METHOD}'
                        f'/{CFG.TRAIN_VAL_SEED}')
            data_out_dir = data_dir
            data_out_dir = f'/projects/kunyang/nvflare_py31012/nvflare/{data_dir}'
            CFG.data_out_dir = data_out_dir
            gen_client_data(data_dir, data_out_dir, CFG)

            print(f"\n***************************** Global Models *************************************")
            global_fnn = FNN(num_classes=CFG.NUM_CLASSES)
            global_fnn = global_fnn.to(DEVICE)
            print(global_fnn)

            histories = {'clients': [], 'server': [], 'CFG': CFG}
            for server_epoch in range(CFG.SERVER_EPOCHS):
                print(f"\n*************** Server Epoch: {server_epoch}/{CFG.SERVER_EPOCHS}, Client Training *************")
                clients_fnns, clients_info, history = clients_training(data_out_dir, server_epoch, global_fnn, CFG)
                histories['clients'].append(history)

                print(f"\n*************** Server Epoch: {server_epoch}/{CFG.SERVER_EPOCHS}, Server Aggregation **********")
                aggregate_cnns(clients_fnns, clients_info, global_fnn, CFG.AGGREGATION_METHOD, histories, server_epoch)

            # prefix = f'-n_{CFG.SERVER_EPOCHS}'
            # history_file = f'{CFG.data_out_dir}/histories_{prefix}.pt.gz'
            # print(f'saving histories to {history_file}')
            # # with open(history_file, 'wb') as f:
            # #     pickle.dump(histories, f)
            # torch.save(histories, history_file)

            # try:
            #     print_histories(histories)
            # except Exception as e:
            #     print('Exception: ', e)
            # # print_histories_server(histories['server'])

            # Delete all the generated data
            shutil.rmtree(data_out_dir)

            # # save all results
            all_histories[CFG.TRAIN_VAL_SEED] = histories
            # history_file = f'{os.path.dirname(CFG.data_out_dir)}/all_histories_{NUM_REPEATS}.pt.gz'
            # print(f'saving all histories to {history_file}')
            # # with open(history_file, 'wb') as f:
            # #     pickle.dump(all_histories, f)
            # torch.save(all_histories, history_file)

        # history_file = 'data/Sentiment140/sign_flipping/h_12-b_8-5-0.8-0.1-krum_avg/all_histories_3.pt.gz'
        # history_file = 'all_histories_5.pt.gz'
        # all_histories = torch.load(history_file)
        print_all(all_histories)
    except Exception as e:
        traceback.print_exc()
        shutil.rmtree(data_out_dir)


if __name__ == '__main__':
    main()
