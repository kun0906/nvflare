"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    # $module load conda
    # $conda activate nvflare-3.10
    # $cd nvflare/auto_labeling
    $module load conda && conda activate nvflare-3.10 && cd nvflare/auto_labeling
    $PYTHONPATH=. python3 fl_cnn_robust_aggregation_random_noise_model_nips_paper.py

    Byzantine (malicious or faulty) clients.

    Storage path: /projects/kunyang/nvflare_py31012/nvflare
"""

import argparse
import collections
import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets

import robust_aggregation
from utils import timer

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

VERBOSE = 5


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedCNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--labeling_rate', type=float, required=False, default=3,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=5,
                        help="The number of server epochs (integer).")
    parser.add_argument('-b', '--honest_clients', type=int, required=False, default=2,
                        help="The number of honest clients.")
    parser.add_argument('-a', '--aggregation_method', type=str, required=False, default='mean',
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
# BIG_NUMBER = args.labeling_rate
# SERVER_EPOCHS = args.server_epochs
SERVER_EPOCHS = 10000
IID_CLASSES_CNT = args.server_epochs
# NUM_HONEST_CLIENTS = args.honest_clients
# NUM_BYZANTINE_CLIENTS = NUM_HONEST_CLIENTS - 1
# the total number of clients is 20, in which 33% of them is malicious clients, i.e., f = int(0.33*20) = 6.
TOTAL_CLIENTS = 20
ATTACK_METHOD = 'omniscient'
if ATTACK_METHOD == 'omniscient':   # case 2
    NUM_BYZANTINE_CLIENTS = 0 # int(0.45 * TOTAL_CLIENTS)  # 0
else:   # case 1: gaussian noise
    NUM_BYZANTINE_CLIENTS = 0  # int(0.33 * TOTAL_CLIENTS)  # 0
NUM_HONEST_CLIENTS = TOTAL_CLIENTS - NUM_BYZANTINE_CLIENTS
AGGREGATION_METHOD = args.aggregation_method
# aggregation_method = 'mean'  # adaptive_krum, krum, median, mean
EPOCHS_CLIENT = 1  # number of epochs of each client used
print(args)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # torch.tensor(X, dtype=torch.float32)
        self.y = y  # torch.tensor(y, dtype=torch.long)  # For classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # From 1 channel to 16 channels
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv21 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Adjust the dimensions after the convolution layers
        self.fc2 = nn.Linear(512, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # # self.sigmoid = nn.Sigmoid()
        #
        # self.transform = nn.Sequential(
        #     nn.Linear(28 * 28 + num_classes, 784),
        #     nn.LeakyReLU(0.2),
        # )

        self.fc11 = nn.Linear(57, 32)
        self.fc21 = nn.Linear(32, 16)
        self.fc22 = nn.Linear(16, 8)
        self.fc33 = nn.Linear(8, num_classes)

    def forward(self, x):

        model_type = 'mlp'
        if model_type == 'mlp':
            x = self.leaky_relu(self.fc11(x))
            x = self.leaky_relu(self.fc21(x))
            x = self.leaky_relu(self.fc22(x))
            x = self.fc33(x)
        else:
            x = x.view(x.shape[0], 1, 28, 28)  # (N, 1, 28, 28)

            # Ensure input has the correct shape (batch_size, 1, 28, 28)
            x = self.leaky_relu(self.conv1(x))
            # x = self.leaky_relu(self.conv11(x))

            x = self.leaky_relu(self.conv2(x))
            # x = self.leaky_relu(self.conv21(x))

            x = self.leaky_relu(self.conv3(x))
            # x = self.leaky_relu(self.conv31(x))

            x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
            x = self.leaky_relu(self.fc1(x))
            # x = F.softmax(self.fc2(x), dim=1)
            x = self.fc2(x)
        return x


@timer
def train_cnn(local_cnn, global_cnn, local_data, train_info={}):
    """
        1. Use gans to generated data for each class
        2. Use the generated data + local data to train local cnn with initial parameters of global_cnn
        3. Send local cnn'parameters to server.
    Args:
        local_cnn:
        gan:
        global_cnn:
        local_data:

    Returns:

    """

    # X, y, train_mask, val_mask, test_mask = train_info['cnn']['data']
    X, y = local_data['X'], local_data['y']
    train_mask, val_mask, test_mask = local_data['train_mask'], local_data['val_mask'], local_data['test_mask']
    tmp = X.cpu().numpy().flatten()
    print(f'X: min: {min(tmp)}, max: {max(tmp)}')

    X_train, y_train = X[train_mask], y[train_mask]
    print('Compute classes weights...')
    # Get indices of y
    y_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    indices = torch.tensor(y_indices).to(DEVICE)  # total labeled data
    new_y = y[indices]
    labeled_cnt = collections.Counter(new_y.tolist())
    print('labeled_y: ', labeled_cnt.items(), flush=True)
    s = sum(labeled_cnt.values())
    labeled_classes_weights = {k: s / v for k, v in labeled_cnt.items()}
    s2 = sum(labeled_classes_weights.values())
    labeled_classes_weights = {k: w / s2 for k, w in labeled_classes_weights.items()}  # normalize weights
    # data['labeled_classes_weights'] = labeled_classes_weights
    # print('labeled_y (train)', collections.Counter(new_y.tolist()), ', old_y:', ct.items(),
    #       f'\nlabeled_classes_weights ({sum(labeled_classes_weights.values())})',
    #       {k: float(f"{v:.2f}") for k, v in labeled_classes_weights.items()})

    # only train smaller model
    losses = []
    val_losses = []
    best = {'epoch': -1, 'val_accuracy': -1.0, 'val_accs': [], 'val_losses': [], 'train_accs': [], 'train_losses': []}
    val_cnt = 0
    pre_val_loss = 0
    # # here, you need make sure weight aligned with class order.
    class_weight = torch.tensor(list(labeled_classes_weights.values()), dtype=torch.float).to(DEVICE)
    print(f'class_weight: {class_weight}, sum(class_weight) = {sum(class_weight)}')

    local_cnn = local_cnn.to(DEVICE)
    local_cnn.load_state_dict(global_cnn.state_dict())  # Initialize client_gm with the parameters of global_model
    # optimizer = optim.Adam(local_cnn.parameters(), lr=0.005)
    # optimizer = optim.Adam(local_cnn.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    optimizer = optim.SGD(local_cnn.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(local_cnn.parameters(), lr=0.001, weight_decay=5e-4)
    # criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean').to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=BIG_NUMBER, shuffle=True)
    for epoch in range(EPOCHS_CLIENT):
        local_cnn.train()  #
        model_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # your local personal model
            outputs = local_cnn(X_batch)
            # Loss calculation: Only for labeled nodes
            loss_ = criterion(outputs, y_batch)
            loss_.backward()

            # # Print gradients for each parameter
            # print("Gradients for model parameters:")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad}")
            #     else:
            #         print(f"{name}: No gradient (likely frozen or unused)")

            optimizer.step()

            scheduler.step()  # adjust learning rate

            model_loss += loss_.item()

            # from the paper, Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
            # During round t, the parameter server broadcasts its parameter vector x t ∈ R d to all the workers.
            # Each correct worker p computes an estimate V p t = G(x t , ξ p t ) of the gradient ∇Q(x t ) of the
            # cost function Q, where ξ p t is a random variable representing, e.g.,
            # the sample (or a mini-batch of samples) drawn from the dataset. so here for each epoch,
            # only one batch is used to compute the gradient and update the model
            break


        losses.append(model_loss)

        val_loss = model_loss
        if epoch % 100 == 0:
            print(f"train_cnn epoch: {epoch}, local_cnn train loss: {model_loss:.4f}, "
                  f"val_loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]}")

        # if stop_training:
        #     local_cnn.stop_training = True
        #     print(f'Early Stopping. Epoch: {epoch}, Loss: {model_loss:.4f}')
        #     break

    # exit()
    # train_info['cnn'] = {'graph_data': graph_data, "losses": losses}
    # print('***best at epoch: ', best['epoch'], ' best val_accuracy: ', best['val_accuracy'])
    # local_cnn.load_state_dict(best['model'])

    # if train_info['server_epoch'] % 10 == 0:
    #     X_train, y_train = graph_data.x[graph_data.train_mask], graph_data.y[graph_data.train_mask]
    #     X_val, y_val = graph_data.x[graph_data.val_mask], graph_data.y[graph_data.val_mask]
    #     X_test, y_test = graph_data.x[graph_data.test_mask], graph_data.y[graph_data.test_mask]
    #     # test on the generated data
    #     dim = X_train.shape[1]
    #     X_gen_test = np.zeros((0, dim))
    #     y_gen_test = np.zeros((0,), dtype=int)
    #     for l, vs in generated_data.items():
    #         X_gen_test = np.concatenate((X_gen_test, vs['X']), axis=0)
    #         y_gen_test = np.concatenate((y_gen_test, vs['y']))
    #
    #
    #     evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_gen_test, y_gen_test, verbose=10)

    show = False
    if show:
        client_id = train_info['client_id']
        server_epoch = train_info['server_epoch']
        fig_file = f'{IN_DIR}/{client_id}/server_epoch_{server_epoch}.png'
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = axes.reshape((1, 2))
        val_losses = best['val_losses']
        train_losses = best['train_losses']
        axes[0, 0].plot(range(len(losses)), train_losses, label='Training Loss', marker='+')  # in early_stopping
        axes[0, 0].plot(range(len(val_losses)), val_losses, label='Validating Loss', marker='o')  # in early_stopping
        # axes[0, 0].plot(range(len(losses)), losses, label='Training Loss', marker='o')  # in cnn_train
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        best_val_acc = best['val_accuracy']
        train_acc = best['train_accuracy']
        epoch = best['epoch']
        axes[0, 0].set_title(f'Best_Val_Acc: {best_val_acc:.2f}, train: {train_acc:.2f} at Epoch: {epoch}')
        axes[0, 0].legend(fontsize='small')

        train_accs = best['train_accs']
        val_accs = best['val_accs']
        axes[0, 1].plot(range(len(val_accs)), train_accs, label='Training Accuracy', marker='+')  # in early_stopping
        axes[0, 1].plot(range(len(val_accs)), val_accs, label='Validating Accuracy', marker='o')  # in early_stopping
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        best_val_acc = best['val_accuracy']
        epoch = best['epoch']
        axes[0, 1].set_title(f'Best_Val_Acc: {best_val_acc:.2f},  train: {train_acc:.2f}  at Epoch: {epoch}')
        axes[0, 1].legend(fontsize='small')

        # plt.suptitle(title)
        # plt.grid()
        plt.tight_layout()
        plt.savefig(fig_file)
        # plt.show()
        plt.clf()
        plt.close(fig)

    return None


#
# def aggregate_cnns_layers(clients_cnns, clients_info, global_cnn, aggregation_method, histories, epoch):
#     print('*aggregate cnn...')
#
#     # Initialize the aggregated state_dict for the global model
#     global_state_dict = {key: torch.zeros_like(value).to(DEVICE) for key, value in global_cnn.state_dict().items()}
#
#     # Aggregate parameters for each layer
#     for key in global_state_dict:
#         # print(f'global_state_dict: {key}')
#         clients_updates = [client_state_dict[key].cpu() for client_state_dict in clients_cnns.values()]
#         min_value = min([torch.min(v).item() for v in clients_updates[: NUM_HONEST_CLIENTS]])
#         max_value = max([torch.max(v).item() for v in clients_updates[: NUM_HONEST_CLIENTS]])
#         # each client extra information (such as, number of samples)
#         # client_weights will affect median and krum, so be careful to weights
#         # if assign byzantine clients with very large weights (e.g., 1e6),
#         # then median will choose byzantine client's parameters.
#         clients_weights = torch.tensor([1] * len(clients_updates))  # default as 1
#         # clients_weights = torch.tensor([vs['size'] for vs in clients_info.values()])
#         if aggregation_method == 'adaptive_krum':
#             aggregated_update, clients_type_pred = adaptive_krum(clients_updates, clients_weights, trimmed_average=False)
#         elif aggregation_method == 'krum':
#             train_info = list(histories['clients'][-1].values())[-1]
#             f = train_info['NUM_BYZANTINE_CLIENTS']
#             # client_type = train_info['client_type']
#             aggregated_update, clients_type_pred = krum(clients_updates, clients_weights, f, trimmed_average=False)
#         elif aggregation_method == 'median':
#             aggregated_update, clients_type_pred = median(torch.stack(clients_updates, dim=0), clients_weights, dim=0)
#         else:
#             aggregated_update, clients_type_pred = mean(clients_updates, clients_weights)
#         print(f'{aggregation_method}, {key}, clients_type: {clients_type_pred}, '
#               f'client_updates: min: {min_value}, max: {max_value}')  # f'clients_weights: {clients_weights.numpy()},
#         global_state_dict[key] = aggregated_update.to(DEVICE)
#
#     # Update the global model with the aggregated parameters
#     global_cnn.load_state_dict(global_state_dict)
#

@timer
def aggregate_cnns(clients_cnns, clients_info, global_cnn, aggregation_method, histories, epoch):
    print('*aggregate cnn...')
    # flatten all the parameters into a long vector
    # clients_updates = [client_state_dict.cpu() for client_state_dict in clients_cnns.values()]

    # Concatenate all parameter tensors into one vector.
    # Note: The order here is the iteration order of the OrderedDict, which
    # may not match the order of model.parameters().
    # vector_from_state = torch.cat([param.view(-1) for param in state.values()])
    # flatten_clients_updates = [torch.cat([param.view(-1).cpu() for param in client_state_dict.values()]) for
    #                            client_state_dict in clients_cnns.values()]
    tmp_models = []
    for client_state_dict in clients_cnns.values():
        model = CNN(num_classes=NUM_CLASSES)
        model.load_state_dict(client_state_dict)
        tmp_models.append(model)
    flatten_clients_updates = [parameters_to_vector(md.parameters()).detach().cpu() for md in tmp_models]
    # for debugging
    if VERBOSE >= 30:
        for i, update in enumerate(flatten_clients_updates):
            print(f'client_{i}:', end='  ')
            print_histgram(update, bins=5, value_type='params')

    min_value = min([torch.min(v).item() for v in flatten_clients_updates[: NUM_HONEST_CLIENTS]])
    max_value = max([torch.max(v).item() for v in flatten_clients_updates[: NUM_HONEST_CLIENTS]])

    trimmed_average = False
    # each client extra information (such as, number of samples)
    # client_weights will affect median and krum, so be careful to weights
    # if assign byzantine clients with very large weights (e.g., 1e6),
    # then median will choose byzantine client's parameters.
    clients_weights = torch.tensor([1] * len(flatten_clients_updates))  # default as 1
    # clients_weights = torch.tensor([vs['size'] for vs in clients_info.values()])
    if aggregation_method == 'adaptive_krum':
        aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(flatten_clients_updates, clients_weights,
                                                                               trimmed_average, verbose=VERBOSE)
    elif aggregation_method == 'krum':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        f = NUM_BYZANTINE_CLIENTS
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
                                                                       trimmed_average, verbose=VERBOSE)
    elif aggregation_method == 'median':
        p = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
        p = p / 2  # top p/2 and bottom p/2 are removed
        aggregated_update, clients_type_pred = robust_aggregation.median(flatten_clients_updates, clients_weights,
                                                                         trimmed_average, p=p,
                                                                         verbose=VERBOSE)
    else:
        p = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
        p = p / 2  # top p/2 and bottom p/2 are removed
        aggregated_update, clients_type_pred = robust_aggregation.mean(flatten_clients_updates, clients_weights,
                                                                       trimmed_average, p=p,
                                                                       verbose=VERBOSE)
    print(f'{aggregation_method}, clients_type: {clients_type_pred}, '
          f'client_updates: min: {min_value}, max: {max_value}')  # f'clients_weights: {clients_weights.numpy()},

    # Update the global model with the aggregated parameters
    # w = w0 - (delta_w), where delta_w = \eta*\namba_w
    aggregated_update = parameters_to_vector(global_cnn.parameters()).detach().cpu() - aggregated_update
    aggregated_update = aggregated_update.to(DEVICE)
    vector_to_parameters(aggregated_update, global_cnn.parameters())  # in_place
    # global_cnn.load_state_dict(aggregated_update)


@timer
def evaluate(local_cnn, local_data, global_cnn, test_type='test', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    print('---------------------------------------------------------------')
    for model_type, model in [('global', global_cnn), ('local', local_cnn)]:
        # At time t, global model has not been updated yet, however, local_cnn is updated.
        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'***Testing {model_type} model on {test_type}...')
        cnn = model
        cnn = cnn.to(DEVICE)

        criterion = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)

        # X, Y, train_mask, val_mask, test_mask = train_info['cnn']['data']
        X, Y = local_data['X'], local_data['y']
        train_mask, val_mask, test_mask = local_data['train_mask'], local_data['val_mask'], local_data['test_mask']

        # for debug purpose
        for data_type, mask_ in [('train', train_mask),
                                 ('val', val_mask),
                                 ('test', test_mask)]:
            # Calculate accuracy for the labeled data
            # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
            # print(f'labeled_indices {len(labeled_indices)}')

            X_, y_ = X[mask_], Y[mask_]
            cnn.eval()
            with torch.no_grad():
                output = cnn(X_)
                loss = criterion(output, y_)
                _, predicted_labels = torch.max(output, dim=1)

            train_info[f'{model_type}_{data_type}_loss'] = loss.item()

            y_ = y_.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()
            print(collections.Counter(y_.tolist()))

            # Total samples and number of classes
            total_samples = len(y_)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_.tolist()).items()}
            # sample_weight = [class_weights[y_0.item()] for y_0 in y]
            sample_weight = [1 for y_0 in y_]
            # print(f'class_weights: {class_weights}')

            accuracy = accuracy_score(y_, y_pred, sample_weight=sample_weight)

            train_info[f'{model_type}_{data_type}_accuracy'] = accuracy
            print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y_.tolist())}, "
                  f"loss: {loss:.4f}")
            conf_matrix = confusion_matrix(y_, y_pred, sample_weight=sample_weight)
            conf_matrix = conf_matrix.astype(int)
            train_info[f'{model_type}_{data_type}_cm'] = conf_matrix
            print("Confusion Matrix:\n", conf_matrix)

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


@timer
def evaluate_shared_test(local_cnn, local_data, global_cnn,
                         test_type='shared_test_data', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    print('---------------------------------------------------------------')
    shared_data = local_data['shared_data']

    # shared_test
    X_test, y_test = shared_data['X'].to(DEVICE), shared_data['y'].to(DEVICE)
    print(f'X_test: {X_test.size()}, {collections.Counter(y_test.tolist())}')
    tmp = X_test.cpu().numpy().flatten()
    print(f'X_test: [min: {min(tmp)}, max: {max(tmp)}]')
    criterion = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)

    for model_type, model in [('global', global_cnn), ('local', local_cnn)]:
        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'***Testing {model_type} model on {test_type}...')
        # evaluate the data
        # cnn = local_cnn
        cnn = model
        cnn.to(DEVICE)
        cnn.eval()
        with torch.no_grad():
            output = cnn(X_test)
            loss = criterion(output, y_test)
            _, predicted_labels = torch.max(output, dim=1)

            train_info[f'{model_type}_shared_loss'] = loss.item()

            # only on test set
            print('Evaluate on shared test data...')
            # Calculate accuracy for the labeled data
            # NUM_CLASSES = 10
            # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
            # print(f'labeled_indices {len(labeled_indices)}')
            true_labels = y_test

            y = true_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()

            # Total samples and number of classes
            total_samples = len(y)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y]
            sample_weight = [1 for y_0 in y]
            # print(f'class_weights: {class_weights}')

            accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
            train_info[f'{model_type}_shared_accuracy'] = accuracy
            print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}, "
                  f"loss: {loss:.4f}")

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
            conf_matrix = conf_matrix.astype(int)
            train_info[f'{model_type}_shared_cm'] = conf_matrix
            print("Confusion Matrix:")
            print(conf_matrix)

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


def print_histories(histories):
    num_server_epoches = len(histories)
    num_clients = len(histories[0])
    print('num_server_epoches:', num_server_epoches, ' num_clients:', num_clients)

    for model_type in ['global', 'local']:
        print(f'\n***model_type: {model_type}***')
        ncols = 2
        nrows, r = divmod(num_clients, ncols)
        nrows = nrows if r == 0 else nrows + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 15))  # width, height
        for c in range(num_clients):
            i, j = divmod(c, ncols)
            print(f"\nclient {c}")
            train_accs = []  # train
            val_accs = []
            unlabeled_accs = []  # test
            shared_accs = []
            client_type = None
            try:
                for evaluation_metric in ['accuracy']:  # 'loss',
                    for s in range(num_server_epoches):
                        client = histories[s][c]
                        client_type = client["client_type"]
                        train_acc = client[f'{model_type}_train_{evaluation_metric}']
                        val_acc = client[f'{model_type}_val_{evaluation_metric}']
                        test_acc = client[f'{model_type}_test_{evaluation_metric}']
                        shared_acc = client[f'{model_type}_shared_{evaluation_metric}']
                        train_accs.append(train_acc)
                        val_accs.append(val_acc)
                        unlabeled_accs.append(test_acc)
                        shared_accs.append(shared_acc)
                        print(f'\t\tEpoch: {s}, labeled_{evaluation_metric}:{train_acc:.2f}, '
                              f'val_{evaluation_metric}:{val_acc:.2f}, '
                              f'unlabeled_{evaluation_metric}:{test_acc:.2f}, '
                              f'shared_{evaluation_metric}:{shared_acc:.2f}')
            except Exception as e:
                print(f'\t\tException: {e}')
            # Training and validation loss on the first subplot
            if evaluation_metric == 'accuracy':
                label = 'acc'
            else:
                label = 'loss'
            axes[i, j].plot(range(len(train_accs)), train_accs, label=f'labeled_{label}', marker='o')
            axes[i, j].plot(range(len(val_accs)), val_accs, label=f'val_{label}', marker='o')
            axes[i, j].plot(range(len(unlabeled_accs)), unlabeled_accs, label=f'unlabeled_{label}', marker='+')
            axes[i, j].plot(range(len(shared_accs)), shared_accs, label=f'shared_{label}', marker='s')
            axes[i, j].set_xlabel('Server Epochs')
            if evaluation_metric == 'accuracy':
                axes[i, j].set_ylabel('Accuracy')
            else:
                axes[i, j].set_ylabel('Loss')
            axes[i, j].set_title(f'Client_{c}: {client_type}')
            axes[i, j].legend(fontsize=6.5)

        malicious_ratio = NUM_BYZANTINE_CLIENTS / (NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS)
        title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
                 f':{malicious_ratio:.2f}-{LABELING_RATE:.2f}')
        plt.suptitle(title)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
                    f'{SERVER_EPOCHS}_{NUM_HONEST_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_{evaluation_metric}.png')
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300)
        plt.show()
        plt.close(fig)


def print_data(local_data):
    # print('Local_data: ')
    X, y = local_data['X'], local_data['y']
    tmp = X.cpu().numpy().flatten()
    print(f'X: {X.shape} [min: {min(tmp)}, max:{max(tmp)}], y: '
          f'{collections.Counter(y.tolist())}, in which, ')
    train_mask = local_data['train_mask']
    val_mask = local_data['val_mask']
    test_mask = local_data['test_mask']
    X_train, y_train = local_data['X'][train_mask], local_data['y'][train_mask]
    X_val, y_val = local_data['X'][val_mask], local_data['y'][val_mask]
    X_test, y_test = local_data['X'][test_mask], local_data['y'][test_mask]
    print(f'\tX_train: {X_train.shape}, y_train: '
          f'{collections.Counter(y_train.tolist())}')
    print(f'\tX_val: {X_val.shape}, y_val: '
          f'{collections.Counter(y_val.tolist())}')
    print(f'\tX_test: {X_test.shape}, y_test: '
          f'{collections.Counter(y_test.tolist())}')


def print_histgram(new_probs, bins=5, value_type='probs'):
    print(f'***Print histgram of {value_type}, min:{min(new_probs)}, max: {max(new_probs)}***')
    # # Convert the probabilities to numpy for histogram calculation
    # new_probs = new_probs.detach().cpu().numpy()
    # Compute histogram
    hist, bin_edges = torch.histogram(new_probs, bins=bins)
    # Print histogram
    for i in range(len(hist)):
        print(f"\tBin {i}: {value_type} Range ({bin_edges[i]}, {bin_edges[i + 1]}), Frequency: {hist[i]}")


def print_histories_server(histories_server):
    num_server_epoches = len(histories_server)

    for model_type in ['global']:
        print(f'\n***model_type: {model_type}***')
        ncols = 2
        nrows, r = divmod(4, ncols)
        nrows = nrows if r == 0 else nrows + 1
        fig, axes = plt.subplots(nrows, ncols)
        for clf_idx, clf_name in enumerate(['Random Forest']):
            i, j = divmod(clf_idx, ncols)
            print(f"\n {clf_name}")
            train_accs = []  # train
            val_accs = []
            unlabeled_accs = []  # test
            shared_accs = []
            for s in range(num_server_epoches):
                ml_info = histories_server[s][clf_name]
                #  ml_info[clf_name] = {test_type: {'accuracy': accuracy, 'cm': cm}}
                train_acc = ml_info['train']['accuracy']
                val_acc = ml_info['val']['accuracy']
                test_acc = ml_info['test']['accuracy']
                shared_acc = ml_info['shared_test']['accuracy']

                train_accs.append(train_acc)
                val_accs.append(val_acc)
                unlabeled_accs.append(test_acc)
                shared_accs.append(shared_acc)
                print(f'\t\tEpoch: {s}, labeled_acc:{train_acc:.2f}, val_acc:{val_acc:.2f}, '
                      f'unlabeled_acc:{test_acc:.2f}, '
                      f'shared_acc:{shared_acc:.2f}')

            # Training and validation loss on the first subplot
            axes[i, j].plot(range(len(train_accs)), train_accs, label='labeled_acc', marker='o')
            axes[i, j].plot(range(len(val_accs)), val_accs, label='val_acc', marker='o')
            axes[i, j].plot(range(len(unlabeled_accs)), unlabeled_accs, label='unlabeled_acc', marker='+')
            axes[i, j].plot(range(len(shared_accs)), shared_accs, label='shared_acc', marker='s')
            axes[i, j].set_xlabel('Server Epochs')
            axes[i, j].set_ylabel('Accuracy')
            axes[i, j].set_title(f'{clf_name}')
            axes[i, j].legend(fontsize=5)

        if model_type == 'global':
            title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}' + '}$' + f':{LABELING_RATE}'
        else:
            title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' + f':{LABELING_RATE}'
        plt.suptitle(title)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig_file = f'{IN_DIR}/{LABELING_RATE}/{model_type}_accuracy.png'
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300)
        plt.show()
        plt.close(fig)


def plot_imgs(X, X_noise):
    # Plot original and noisy images
    n_cols = 10
    fig, ax = plt.subplots(10, n_cols, figsize=(15, 15))  # Bigger figure

    for i in range(5):
        for j in range(n_cols):
            ax[i, j].imshow(X[i].squeeze(), cmap="gray")
            # ax[0].set_title("Original Image")
            ax[i, j].axis("off")  # Hide axes

    for i in range(5, 10, 1):
        for j in range(n_cols):
            ax[i, j].imshow(X_noise[i - 5].squeeze(), cmap="gray")
            # ax[0].set_title("Noisy Image")
            ax[i, j].axis("off")  # Hide axes

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Add spacing
    plt.show()


def normalize(X):
    # return (X / 255.0 - 0.5) * 2  # [-1, 1]   # for MNIST
    return (X / 255.0 - 0.5) * 2  # [-1, 1]


from numpy.random import dirichlet


def dirichlet_split(X, y, num_clients, alpha=0.5):
    """Splits dataset using Dirichlet distribution for non-IID allocation.

    alpha: > 0
        how class samples are divided among clients.
        Small alpha (e.g., 0.1) → Highly Non-IID
            Each client receives data dominated by a few classes.
            Some clients may not have samples from certain classes.

        Large alpha (e.g., 10) → More IID-like
            Each client receives a more balanced mix of all classes.
            The distribution approaches uniformity as alpha increases.

        alpha = 1 → Mildly Non-IID
            Classes are somewhat skewed, but each client still has a mix of multiple classes.

    """
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    X_splits, y_splits = [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]

    for c, indices in class_indices.items():
        np.random.shuffle(indices)
        proportions = dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(indices)).astype(int)

        start = 0
        for client, num_samples in enumerate(proportions):
            X_splits[client].extend(X[indices[start:start + num_samples]])
            y_splits[client].extend(y[indices[start:start + num_samples]])
            start += num_samples

    return [np.array(X_s) for X_s in X_splits], [np.array(y_s) for y_s in y_splits]


@timer
def gen_client_spambase_data(data_dir='data/spambase', out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(data_dir, 'spambase.data'), dtype=float, header=None)
    X, y = torch.tensor(df.iloc[:, 0:-1].values), torch.tensor(df.iloc[:, -1].values, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        shuffle=True, random_state=42)
    # Initialize scaler and fit ONLY on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mask = np.full(len(y_test), False)
    for l in LABELS:
        mask_ = y_test == l
        mask[mask_] = True
    X_test, y_test = X_test[mask], y_test[mask]
    # preprocessing X_test
    # X_test = normalize(X_test.numpy())
    y_test = y_test.numpy()
    shared_data = {"X": torch.tensor(X_test).float().to(DEVICE), 'y': torch.tensor(y_test).to(DEVICE)}

    X, y = X_train, y_train
    mask = np.full(len(y), False)
    for l in LABELS:
        mask_ = y == l
        mask[mask_] = True
    X, y = X[mask], y[mask]
    # X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()
    num_samples = len(y)
    dim = X.shape[1]

    random_state = 42
    torch.manual_seed(random_state)
    indices = torch.randperm(num_samples)  # Randomly shuffle
    step = int(num_samples / NUM_HONEST_CLIENTS)
    # step = 50  # for debugging
    non_iid_cnt0 = 0  # # make sure that non_iid_cnt is always less than iid_cnt
    non_iid_cnt1 = 0

    Xs, Ys = dirichlet_split(X, y, num_clients=NUM_HONEST_CLIENTS, alpha=0.5)
    print([collections.Counter(y_) for y_ in Ys])
    # exit(0)
    ########################################### Benign Clients #############################################
    for c in range(NUM_HONEST_CLIENTS):
        client_type = 'Honest'
        print(f"\n*** client_{c}: {client_type}...")
        X_c = X[:, ]
        y_c = y[:, ]
        # # X_c = X[indices[c * step:(c + 1) * step]]
        # # y_c = y[indices[c * step:(c + 1) * step]]
        # # np.random.seed(c)  # change seed

        # X_c, y_c = Xs[c], Ys[c]  # using dirichlet distribution
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
        print(f'client_{c} data ({len(label_cnts.keys())}):', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)

    ########################################### Malicious Clients #############################################
    indices = torch.randperm(num_samples)  # Randomly shuffle
    for c in range(NUM_HONEST_CLIENTS, NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'malicious'
        print(f"\n*** client_{c}: {client_type}...")

        if ATTACK_METHOD == 'omniscient':
            # each Byzantine worker computes an estimate of the gradient over the whole dataset (yielding a very
            # accurate estimate of the gradient), and proposes the opposite vector, scaled to a large length.
            # We refer to this behavior as omniscient.
            X_c = X[:, ]
            y_c = y[:, ]
        else:
            # The Gaussian Byzantine workers: Byzantine workers do not compute an estimator of the gradient and send a random vector,
            # drawn from a Gaussian distribution of which we could set the variance high enough (200) to break averaging strategies.
            X_c = torch.zeros(X.shape, dtype=float).to(DEVICE)
            y_c = torch.zeros(y.shape, dtype=int).to(DEVICE)

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

        # train_info['NUM_BYZANTINE_CLIENTS'] = NUM_BYZANTINE_CLIENTS
        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).to(DEVICE).float(), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}

        label_cnts = collections.Counter(local_data['y'].tolist())
        print(f'client_{c} data ({len(label_cnts.keys())}):', label_cnts)
        print_data(local_data)

        out_file = f'{out_dir}/{c}.pth'
        torch.save(local_data, out_file)


def clients_training(data_dir, epoch, global_cnn):
    clients_cnns = {}
    clients_info = {}  # extra information (e.g., number of samples) of clients that can be used in aggregation
    history = {}
    ########################################### Benign Clients #############################################
    for c in range(NUM_HONEST_CLIENTS):
        client_type = 'Honest'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        # might be used in server
        train_info = {"client_type": client_type, "cnn": {}, 'client_id': c, 'server_epoch': epoch}

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

    ########################################### Malicious Clients #############################################
    for c in range(NUM_HONEST_CLIENTS, NUM_HONEST_CLIENTS + NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'malicious'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        # might be used in server
        train_info = {"client_type": client_type, "cnn": {}, 'client_id': c, 'server_epoch': epoch}

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
        # train_cnn(local_cnn, global_cnn, local_data, train_info)

        # from paper: Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
        if ATTACK_METHOD == 'omniscient':
            # each Byzantine worker computes an estimate of the gradient over the whole dataset (yielding a very
            # accurate estimate of the gradient), and proposes the opposite vector, scaled to a large length.
            # We refer to this behavior as omniscient.
            scale_factor = 2.0
            train_cnn(local_cnn, global_cnn, local_data, train_info)
            delta_w = {key: -1 * scale_factor * (global_cnn.state_dict()[key] - local_cnn.state_dict()[key]) for key
                       in global_cnn.state_dict()}
        else:
            # The Gaussian Byzantine workers: Byzantine workers do not compute an estimator of the gradient and send a random vector,
            # drawn from a Gaussian distribution of which we could set the variance high enough (200) to break averaging strategies.
            new_state_dict = {}
            for key, param in global_cnn.state_dict().items():
                noise = torch.normal(0, 200, size=param.shape).to(DEVICE)
                new_state_dict[key] = noise
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

    torch.manual_seed(seed)  # CPU
    return clients_cnns, clients_info, history


@timer
def main():
    print(f"\n*************************** Generate Clients Data ******************************")
    dataset = 'spambase'
    if dataset == 'spambase':
        data_dir = '../data/spambase'
        sub_dir = (f'data/spambase/random_noise/h_{NUM_HONEST_CLIENTS}-b_{NUM_BYZANTINE_CLIENTS}'
                   f'-{IID_CLASSES_CNT}-{LABELING_RATE}-{BIG_NUMBER}-{AGGREGATION_METHOD}')
        data_out_dir = data_dir
        data_out_dir = f'/projects/kunyang/nvflare_py31012/nvflare/{sub_dir}'
        print(data_out_dir)
        gen_client_spambase_data(data_dir=data_dir, out_dir=data_out_dir)  # for spambase dataset
    else:
        data_dir = (f'data/MNIST/random_noise/h_{NUM_HONEST_CLIENTS}-b_{NUM_BYZANTINE_CLIENTS}'
                    f'-{IID_CLASSES_CNT}-{LABELING_RATE}-{BIG_NUMBER}-{AGGREGATION_METHOD}')
        print(data_dir)
        data_out_dir = data_dir
        data_out_dir = f'/projects/kunyang/nvflare_py31012/nvflare/{data_dir}'
        # gen_client_data(data_dir=data_dir, out_dir=data_out_dir)  # for MNIST dataset

    print(f"\n***************************** Global Models *************************************")
    global_cnn = CNN(num_classes=NUM_CLASSES)
    global_cnn = global_cnn.to(DEVICE)
    print(global_cnn)

    histories = {'clients': [], 'server': []}
    for server_epoch in range(SERVER_EPOCHS):
        print(f"\n*************** Server Epoch: {server_epoch}/{SERVER_EPOCHS}, Client Training *****************")
        clients_cnns, clients_info, history = clients_training(data_out_dir, server_epoch, global_cnn)
        histories['clients'].append(history)

        print(f"\n*************** Server Epoch: {server_epoch}/{SERVER_EPOCHS}, Server Aggregation **************")
        aggregate_cnns(clients_cnns, clients_info, global_cnn, AGGREGATION_METHOD, histories, server_epoch)

    # prefix = f'-n_{SERVER_EPOCHS}'
    # history_file = f'{IN_DIR}/histories_{prefix}.pth'
    # print(f'saving histories to {history_file}')
    # with open(history_file, 'wb') as f:
    #     pickle.dump(histories, f)
    # torch.save(histories, history_file)

    try:
        print_histories(histories['clients'])
    except Exception as e:
        print('Exception: ', e)
    # print_histories_server(histories['server'])

    # Delete all the generated data
    # shutil.rmtree(data_out_dir)


if __name__ == '__main__':
    IN_DIR = '../data/spambase'
    LABELS = {0, 1}
    NUM_CLASSES = len(LABELS)
    print(f'IN_DIR: {IN_DIR}, AGGREGATION_METHOD: {AGGREGATION_METHOD}, LABELING_RATE: {LABELING_RATE}, '
          f'NUM_HONEST_CLIENTS: {NUM_HONEST_CLIENTS}, NUM_BYZANTINE_CLIENTS: {NUM_BYZANTINE_CLIENTS}, '
          f'NUM_CLASSES: {NUM_CLASSES}, where classes: {LABELS}')
    main()
