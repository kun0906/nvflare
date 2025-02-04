"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare/auto_labeling
    $PYTHONPATH=. python3 fl_cnn_robust_aggregation.py

    Storage path: /projects/kunyang/nvflare_py31012/nvflare
"""

import argparse
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torchvision import datasets

from krum import refined_krum, krum
from utils import timer

print(os.path.abspath(os.getcwd()))
print(__file__)

# Check if GPU is available and use it
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Set print options for 2 decimal places
torch.set_printoptions(precision=2, sci_mode=False)


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedCNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--labeling_rate', type=float, required=False, default=0.,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=2,
                        help="The number of epochs (integer).")
    parser.add_argument('-l', '--hidden_dimension', type=int, required=False, default=2,
                        help="The hidden dimension of CNN.")
    parser.add_argument('-p', '--patience', type=str, required=False, default='refined_krum',
                        help="The patience.")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
LABELING_RATE = args.labeling_rate
SERVER_EPOCHS = args.server_epochs
NUM_BENIGN_CLIENTS = args.hidden_dimension
NUM_BYZANTINE_CLIENTS = int(NUM_BENIGN_CLIENTS / 2) 
AGGREGRATION_METHOD = args.patience
# aggregation_method = 'mean'  # refined_krum, krum, median, mean
print(args)


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

    def forward(self, x):
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
    epochs_client = 101
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
    optimizer = optim.Adam(local_cnn.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    # optimizer = torch.optim.AdamW(local_cnn.parameters(), lr=0.001, weight_decay=5e-4)
    # criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean').to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    # node_features = node_features.view(node_features.shape[0], 1, 28, 28)
    for epoch in range(epochs_client):
        local_cnn.train()  #
        # epoch_model_loss = 0
        # _model_loss, _model_distill_loss = 0, 0
        # epoch_gan_loss = 0
        # _gan_recon_loss, _gan_kl_loss = 0, 0
        # graph_data.to(DEVICE)
        # data_size, data_dim = graph_data.x.shape
        # your local personal model
        outputs = local_cnn(X_train)
        # Loss calculation: Only for labeled nodes
        model_loss = criterion(outputs, y_train)
        # if epoch > 0 and model_loss.item() < 1e-6:
        #     break
        optimizer.zero_grad()
        model_loss.backward()

        # # Print gradients for each parameter
        # print("Gradients for model parameters:")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad}")
        #     else:
        #         print(f"{name}: No gradient (likely frozen or unused)")

        optimizer.step()

        scheduler.step()  # adjust learning rate

        losses.append(model_loss.item())

        # if epoch % 10 == 0:
        #     print(f'epoch: {epoch}, model_loss: {model_loss.item()}')
        #     evaluate_train(local_cnn, graph_data, len(local_data['y']), generated_size, epoch, local_data)
        # # X_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
        # val_loss, pre_val_loss, val_cnt, stop_training = early_stopping(local_cnn, graph_data, None, epoch,
        #                                                                 pre_val_loss, val_cnt, criterion,
        #                                                                 patience=epochs_client,
        #                                                                 best=best)
        # val_losses.append(val_loss.item())
        val_loss = model_loss
        if epoch % 100 == 0:
            print(f"train_cnn epoch: {epoch}, local_cnn train loss: {model_loss.item():.4f}, "
                  f"val_loss: {val_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]}")

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


def evaluate_train(cnn, graph_data, gen_start, generated_size, epoch, local_data):
    print(f'\n***epoch: ', epoch)

    cnn.eval()
    train_mask, val_mask, test_mask = graph_data.train_mask, graph_data.val_mask, graph_data.test_mask

    X_train, y_train = graph_data.x[train_mask], graph_data.y[train_mask]
    X_val, y_val = graph_data.x[val_mask], graph_data.y[val_mask]
    X_test, y_test = graph_data.x[test_mask], graph_data.y[test_mask]

    gen_mask = torch.tensor([False] * len(train_mask), dtype=torch.bool)
    gen_mask[gen_start:gen_start + generated_size] = True

    with torch.no_grad():
        output = cnn(graph_data)
        _, predicted_labels = torch.max(output, dim=1)
        # for debug purpose
        for data_type, mask_ in [('train', train_mask),
                                 ('val', val_mask),
                                 ('test', test_mask),
                                 ('generated', gen_mask)]:
            true_labels = graph_data.y

            predicted_labels_tmp = predicted_labels[mask_]
            true_labels_tmp = true_labels[mask_]
            y = true_labels_tmp.cpu().numpy()
            y_pred = predicted_labels_tmp.cpu().numpy()

            # Total samples and number of classes
            total_samples = len(y)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y]
            print(f'class_weights: {class_weights}')

            accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
            print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
            conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
            conf_matrix = conf_matrix.astype(int)
            print("Confusion Matrix:\n", conf_matrix)

        # for shared test
        shared_data = local_data['shared_data']
        # all_indices = shared_data['indices']
        # edge_indices = shared_data['edge_indices'].to(DEVICE)  # all edge_indices
        X, Y = shared_data['X'].to(DEVICE), shared_data['y'].to(DEVICE)

        # Shared test data
        shared_test_mask = shared_data['test_mask'].to(DEVICE)
        X_shared_test = X[shared_test_mask].to(DEVICE)
        y_shared_test = Y[shared_test_mask].to(DEVICE)
        # X_test_indices = all_indices[shared_test_mask].to(DEVICE)
        print(f'X_test: {X_shared_test.size()}, {collections.Counter(y_shared_test.tolist())}')
        # edge_indices_test = shared_data['edge_indices_test'].to(DEVICE)

        new_X = torch.cat((X, X_shared_test), dim=0)
        new_y = torch.cat((Y, y_shared_test), dim=0)

        graph_data = Data(x=new_X, y=new_y, edge_index=None, edge_weight=None)
        graph_data.to(DEVICE)

        output = cnn(graph_data)
        _, predicted_labels = torch.max(output, dim=1)

        true_labels = graph_data.y
        mask_ = torch.tensor([False] * len(true_labels), dtype=torch.bool)
        mask_[len(X):] = True
        predicted_labels_tmp = predicted_labels[mask_]
        true_labels_tmp = true_labels[mask_]
        y = true_labels_tmp.cpu().numpy()
        y_pred = predicted_labels_tmp.cpu().numpy()

        # Total samples and number of classes
        total_samples = len(y)
        # Compute class weights
        class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
        sample_weight = [class_weights[y_0.item()] for y_0 in y]
        print(f'class_weights: {class_weights}')

        data_type = 'shared_test'
        accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
        print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
        conf_matrix = conf_matrix.astype(int)
        print("Confusion Matrix:\n", conf_matrix)

        if epoch % 10 == 0:
            evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_shared_test, y_shared_test, verbose=10)


def median(clients_updates, clients_weights, dim=0):
    """
    Compute the weighted median for updates of different shapes using (n,) weights.

    Args:
        updates (list of torch.Tensor): A list of `n` tensors with varying shapes.
        weights (torch.Tensor): A 1D tensor of shape (n,) representing the weights.

    Returns:
        torch.Tensor: The weighted median tensor, matching the shape of the first update.
    """
    n = len(clients_updates)  # Number of updates
    assert clients_weights.shape == (n,), "Weights must be of shape (n,) where n is the number of updates."

    clients_type_pred = np.array(['benign'] * len(clients_updates), dtype='U20')

    # Flatten all updates to 1D and stack along a new dimension (first dimension)
    flattened_updates = [u.flatten() for u in clients_updates]
    stacked_updates = torch.stack(flattened_updates, dim=dim)  # Shape: (n, total_elements)

    # Broadcast weights to match stacked shape
    expanded_weights = clients_weights.view(n, 1).expand_as(stacked_updates)
    # expanded_clients_type_pred = clients_type_pred.view(n, 1).expand_as(stacked_updates)

    # Sort updates and apply sorting indices to weights
    sorted_updates, sorted_indices = torch.sort(stacked_updates, dim=dim)
    sorted_weights = torch.gather(expanded_weights, dim, sorted_indices)
    # sorted_clients_type_pred = torch.gather(expanded_clients_type_pred, dim, sorted_indices)

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=dim)

    # Find index where cumulative weight reaches 50% of total weight
    total_weight = cumulative_weights[-1]  # Total weight for each element
    median_mask = cumulative_weights >= (total_weight / 2)

    # Find the first index that crosses the 50% threshold
    median_index = median_mask.to(dtype=torch.int).argmax(dim=dim)

    # Gather median values from sorted updates
    weighted_median_values = sorted_updates.gather(dim, median_index.unsqueeze(dim)).squeeze(dim)

    # Gather corresponding client type for the weighted median
    # weighted_median_type_indices = sorted_clients_type_pred.gather(dim, median_index.unsqueeze(dim)).squeeze(dim)
    # Reshape back to original shape of the first update

    return weighted_median_values.view(clients_updates[0].shape), None


def mean(clients_updates, clients_weights):
    # weight average
    update = 0.0
    cnt = 0.0
    for j in range(len(clients_updates)):
        update += clients_updates[j] * clients_weights[j]
        cnt += clients_weights[j]
    update = update / cnt
    return update, None


def aggregate_cnns(clients_cnns, clients_info, global_cnn, aggregation_method, histories, epoch):
    print('*aggregate cnn...')

    # Initialize the aggregated state_dict for the global model
    global_state_dict = {key: torch.zeros_like(value).to(DEVICE) for key, value in global_cnn.state_dict().items()}

    # Aggregate parameters for each layer
    for key in global_state_dict:
        # print(f'global_state_dict: {key}')
        # Perform simple averaging of the parameters
        clients_updates = [client_state_dict[key].cpu() for client_state_dict in clients_cnns.values()]
        # each client extra information (such as, number of samples)
        # clients_weights = torch.tensor([1] * len(clients_updates)) # default as 1
        clients_weights = torch.tensor([vs['size'] for vs in clients_info.values()])
        if aggregation_method == 'refined_krum':
            aggregated_update, clients_type_pred = refined_krum(clients_updates, clients_weights, return_average=True)
        elif aggregation_method == 'krum':
            train_info = list(histories['clients'][-1].values())[-1]
            f = train_info['NUM_BYZANTINE_CLIENTS']
            client_type = train_info['client_type']
            aggregated_update, clients_type_pred = krum(clients_updates, clients_info, f, return_average=True)
        elif aggregation_method == 'median':
            aggregated_update, clients_type_pred = median(torch.stack(clients_updates, dim=0), clients_weights, dim=0)
        else:
            aggregated_update, clients_type_pred = mean(clients_updates, clients_weights)
        print(f'aggregation_method: {aggregation_method}, {key}, client_type: {clients_type_pred}')
        global_state_dict[key] = aggregated_update.to(DEVICE)

    # Update the global model with the aggregated parameters
    global_cnn.load_state_dict(global_state_dict)


@timer
def evaluate(local_cnn, local_data, DEVICE, global_cnn, test_type='test', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.

        client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        client_data_ =  (graph_data_, feature_info, client_data_)
    """
    print('---------------------------------------------------------------')
    for model_type, model in [('global', global_cnn), ('local', local_cnn)]:
        # At time t, global model has not been updated yet, however, local_cnn is updated.
        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'***Testing {model_type} model on {test_type}...')
        # cnn = local_cnn(input_dim=64, hidden_dim=32, output_dim=10)
        # cnn.load_state_dict(client_result['client_gm'])
        cnn = model
        cnn = cnn.to(DEVICE)

        # X, Y, train_mask, val_mask, test_mask = train_info['cnn']['data']
        X, Y = local_data['X'], local_data['y']
        train_mask, val_mask, test_mask = local_data['train_mask'], local_data['val_mask'], local_data['test_mask']

        cnn.eval()
        with torch.no_grad():
            output = cnn(X)
            _, predicted_labels = torch.max(output, dim=1)

            # for debug purpose
            for data_type, mask_ in [('train', train_mask),
                                     ('val', val_mask),
                                     ('test', test_mask)]:
                # Calculate accuracy for the labeled data
                # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
                # print(f'labeled_indices {len(labeled_indices)}')
                true_labels = Y

                predicted_labels_tmp = predicted_labels[mask_]
                true_labels_tmp = true_labels[mask_]
                y = true_labels_tmp.cpu().numpy()
                y_pred = predicted_labels_tmp.cpu().numpy()
                print(collections.Counter(y.tolist()))

                # Total samples and number of classes
                total_samples = len(y)
                # Compute class weights
                class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
                # sample_weight = [class_weights[y_0.item()] for y_0 in y]
                sample_weight = [1 for y_0 in y]
                print(f'class_weights: {class_weights}')

                accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)

                train_info[f'{model_type}_{data_type}_accuracy'] = accuracy
                print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
                # if 'all' in test_type:
                #     client_result['labeled_accuracy_all'] = accuracy
                # else:
                #     client_result['labeled_accuracy'] = accuracy
                # print(y, y_pred)
                conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
                conf_matrix = conf_matrix.astype(int)
                train_info[f'{model_type}_{data_type}_cm'] = conf_matrix
                print("Confusion Matrix:\n", conf_matrix)

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


@timer
def evaluate_shared_test(local_cnn, local_data, DEVICE, global_cnn, global_lp,
                         test_type='shared_test_data', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    print('---------------------------------------------------------------')
    shared_data = local_data['shared_data']

    # shared_test
    X_test, y_test = shared_data['X'].to(DEVICE), shared_data['y'].to(DEVICE)
    # X_test_indices = all_indices[shared_test_mask].to(DEVICE)
    print(f'X_test: {X_test.size()}, {collections.Counter(y_test.tolist())}')
    tmp = X_test.cpu().numpy().flatten()
    print(f'X_test: [min: {min(tmp)}, max: {max(tmp)}]')

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
            _, predicted_labels = torch.max(output, dim=1)

            # only on test set
            print('Evaluate on shared test data...')
            # Calculate accuracy for the labeled data
            # NUM_CLASSES = 10
            # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
            # print(f'labeled_indices {len(labeled_indices)}')
            true_labels = y_test

            # predicted_labels = predicted_labels[graph_data.train_mask]
            # true_labels = true_labels[graph_data.train_mask]
            # predicted_labels = predicted_labels[graph_data.test_mask]
            # true_labels = true_labels[graph_data.test_mask]

            y = true_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()

            # Total samples and number of classes
            total_samples = len(y)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y]
            sample_weight = [1 for y_0 in y]
            print(f'class_weights: {class_weights}')

            accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
            train_info[f'{model_type}_shared_accuracy'] = accuracy
            print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
            # if 'all' in test_type:
            #     client_result['labeled_accuracy_all'] = accuracy
            # else:
            #     client_result['labeled_accuracy'] = accuracy

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
            conf_matrix = conf_matrix.astype(int)
            train_info[f'{model_type}_shared_cm'] = conf_matrix
            print("Confusion Matrix:")
            print(conf_matrix)
            # if 'all' in test_type:
            #     client_result['labeled_cm_all'] = conf_matrix
            # else:
            #     client_result['labeled_cm'] = conf_matrix

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


def evaluate_ML(local_cnn, local_data, DEVICE, global_cnn, test_type, client_id, train_info, verbose=10):
    if verbose > 5:
        print('---------------------------------------------------------------')
        print('Evaluate Classical ML on each client...')
    ml_info = {}
    # # Local data without any generated data
    # local_X, local_y = local_data['X'], local_data['y']
    # train_mask, val_mask, test_mask = local_data['train_mask'], local_data['val_mask'], local_data['test_mask']
    # X_train, y_train = local_X[train_mask], local_y[train_mask]
    # X_val, y_val = local_X[val_mask], local_y[val_mask]
    # X_test, y_test = local_X[test_mask], local_y[test_mask]

    # global shared test set
    shared_test_mask = local_data['shared_data']['test_mask']
    X_shared_test, y_shared_test = local_data['shared_data']['X'][shared_test_mask].numpy(), \
        local_data['shared_data']['y'][
            shared_test_mask].numpy()

    # local data with generated data
    graph_data = train_info['cnn']['graph_data']
    X, y = graph_data.x.numpy(), graph_data.y.numpy()
    X_train, y_train = X[graph_data.train_mask], y[graph_data.train_mask]  # local X_train + generated data
    X_val, y_val = X[graph_data.val_mask], y[graph_data.val_mask]  # should be the same as X_val, y_val
    X_test, y_test = X[graph_data.test_mask], y[graph_data.test_mask]  # should be the same as X_test, y_test

    dim = X_train.shape[1]
    num_classes = len(set(y_train))
    if verbose > 5:
        print(f'Number of Features: {dim}, Number of Classes: {num_classes}')
        print(f'\tX_train: {X_train.shape}, y_train: '
              f'{collections.Counter(y_train.tolist())}')
        print(f'\tX_val: {X_val.shape}, y_val: '
              f'{collections.Counter(y_val.tolist())}')
        print(f'\tX_test: {X_test.shape}, y_test: '
              f'{collections.Counter(y_test.tolist())}')

        print(f'\tX_shared_test: {X_shared_test.shape}, y_test: '
              f'{collections.Counter(y_shared_test.tolist())}')

    from sklearn.tree import DecisionTreeClassifier
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    from sklearn.ensemble import GradientBoostingClassifier
    gd = GradientBoostingClassifier(random_state=42)

    from sklearn import svm
    svm = svm.SVC(random_state=42)

    # mlp = MLP(dim, 64, num_classes)
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, 'MLP': mlp}
    clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, }

    # shared_data = client_data['shared_data']
    # test_mask = shared_data['test_mask']
    # X_shared_test = shared_data['X'][test_mask]
    # y_shared_test = shared_data['y'][test_mask]
    for clf_name, clf in clfs.items():
        if verbose > 5:
            print(f"\nTraining {clf_name}")
        # Train the classifier on the training data
        if clf_name == 'MLP':
            clf.fit(X_train, y_train, X_val, y_val)
        else:
            clf.fit(X_train, y_train)
        if verbose > 5:
            print(f"Testing {clf_name}")
        for test_type, X_, y_ in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test),
                                  ('shared_test', X_shared_test, y_shared_test)]:
            if verbose > 5:
                print(f'Testing on {test_type}')
            # Make predictions on the data
            y_pred_ = clf.predict(X_)

            # Total samples and number of classes
            total_samples = len(y_)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_.tolist()).items()}
            # sample_weight = [class_weights[y_0.item()] for y_0 in y_]
            sample_weight = [1 for y_0 in y]
            print(f'class_weights: {class_weights}')

            # Calculate accuracy
            accuracy = accuracy_score(y_, y_pred_, sample_weight=sample_weight)
            if verbose > 5:
                print(f"Accuracy of {clf_name}: {accuracy * 100:.2f}%")
            # Compute confusion matrix
            cm = confusion_matrix(y_, y_pred_, sample_weight=sample_weight)
            cm = cm.astype(int)
            if verbose > 5:
                print(cm)
            ml_info[clf_name] = {test_type: {'accuracy': accuracy, 'cm': cm}}
    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()

    train_info['ml_info'] = ml_info


def evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test,
                 X_shared_test, y_shared_test, verbose=10):
    if verbose > 5:
        print('---------------------------------------------------------------')
        print('Evaluate Classical ML on each client...')
    ml_info = {}

    dim = X_train.shape[1]
    num_classes = len(set(y_train))
    if verbose > 5:
        print(f'Number of Features: {dim}, Number of Classes: {num_classes}')
        print(f'\tX_train: {X_train.shape}, y_train: '
              f'{collections.Counter(y_train.tolist())}')
        print(f'\tX_val: {X_val.shape}, y_val: '
              f'{collections.Counter(y_val.tolist())}')
        print(f'\tX_test: {X_test.shape}, y_test: '
              f'{collections.Counter(y_test.tolist())}')

        print(f'\tX_shared_test: {X_shared_test.shape}, y_shared_test: '
              f'{collections.Counter(y_shared_test.tolist())}')

        print(f'Total (without X_shared_val): X_train + X_val + X_test + X_shared_test = '
              f'{X_train.shape[0] + X_val.shape[0] + X_test.shape[0] + X_shared_test.shape[0]}')

    from sklearn.tree import DecisionTreeClassifier
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    from sklearn.ensemble import GradientBoostingClassifier
    gd = GradientBoostingClassifier(random_state=42)

    from sklearn import svm
    svm = svm.SVC(random_state=42)

    # mlp = MLP(dim, 64, num_classes)
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, 'MLP': mlp}
    clfs = {'Random Forest': rf}

    # shared_data = client_data['shared_data']
    # test_mask = shared_data['test_mask']
    # X_shared_test = shared_data['X'][test_mask]
    # y_shared_test = shared_data['y'][test_mask]
    for clf_name, clf in clfs.items():
        if verbose > 5:
            print(f"\nTraining {clf_name}")
        # Train the classifier on the training data
        if clf_name == 'MLP':
            clf.fit(X_train, y_train, X_val, y_val)
        else:
            clf.fit(X_train, y_train)
        if verbose > 5:
            print(f"Testing {clf_name}")
        ml_info[clf_name] = {}
        for test_type, X_, y_ in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test),
                                  ('shared_test', X_shared_test, y_shared_test)
                                  ]:
            if verbose > 5:
                print(f'Testing on {test_type}')
            # Make predictions on the data
            y_pred_ = clf.predict(X_)

            # Total samples and number of classes
            total_samples = len(y_)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_.tolist()).items()}
            # sample_weight = [class_weights[y_0.item()] for y_0 in y_]
            sample_weight = [1 for y_0 in y_]
            print(f'class_weights: {class_weights}')

            # Calculate accuracy
            accuracy = accuracy_score(y_, y_pred_, sample_weight=sample_weight)
            if verbose > 5:
                print(f"Accuracy of {clf_name}: {accuracy * 100:.2f}%")
            # Compute confusion matrix
            cm = confusion_matrix(y_, y_pred_, sample_weight=sample_weight)
            cm = cm.astype(int)
            if verbose > 5:
                print(cm)
            ml_info[clf_name][test_type] = {'accuracy': accuracy, 'cm': cm}
    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()

    return ml_info


def _gen_models(model, l, size, method='T5'):
    if method == 'T5':

        # pip install sentencepiece
        from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel

        # Initialize the BERT model and tokenizer for embedding extraction
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Initialize the T5 model and tokenizer
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        tokenizer = t5_tokenizer

        # Select a class label and concatenate with random latent vector
        class_label = l
        input_text = f"{class_label}: This is a sample text to generate embedding."

        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Generate text
        outputs = model.generate(**inputs)

        # Decode the output and get embeddings (e.g., use BERT or the model's own hidden states)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Now, use BERT to get the 768-dimensional embedding for the generated text
        encoded_input = bert_tokenizer(generated_text, return_tensors="pt", truncation=True, padding=True,
                                       max_length=512)
        with torch.no_grad():
            outputs = bert_model(**encoded_input)

        # Get the embedding from the [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        # Print the shape of the CLS embedding
        print(f"CLS Embedding Shape: {cls_embedding.shape}")
        embedding = cls_embedding
    elif method == 'gan':
        generator = model
        generator.eval()
        z_dim = generator.latent_dim
        with torch.no_grad():
            z = torch.randn(size, z_dim).to(DEVICE)
            synthetic_data = generator(z)
        embedding = synthetic_data
    elif method == 'cgan':
        generator = model
        generator.eval()
        z_dim = generator.latent_dim
        labels = torch.zeros((size, NUM_CLASSES)).to(DEVICE)
        labels[:, l] = 1
        with torch.no_grad():
            z = torch.randn(size, z_dim).to(DEVICE)
            synthetic_data = generator(z, labels)
            synthetic_data = synthetic_data.squeeze(1)  # Removes the second dimension (size 1)
        embedding = synthetic_data

    else:  # default one is autoencoder
        vae = model
        latent_dim = vae.latent_dim
        # generate latent vector from N(0, 1)
        z = torch.randn(size, latent_dim).to(DEVICE)  # Sample latent vectors
        # ohe_labels = torch.zeros((size, len(LABELS))).to(DEVICE)  # One-hot encoding for class labels
        # ohe_labels[:, l] = 1
        pseudo_logits = vae.decoder(z)  # Reconstruct probabilities from latent space
        embedding = pseudo_logits

    return embedding


def plot_img(generated_imgs, l, show=True, fig_file=f'tmp/generated~.png'):
    if show:
        # real_data = X_train[y_train == l]
        fig, axes = plt.subplots(16, 10, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < len(generated_imgs):
                ax.imshow(generated_imgs[i], cmap='gray')
            # else:
            #     ax.imshow((real_data[i - 100]).astype(int), cmap='gray')
            ax.axis('off')
            # Draw a red horizontal line across the entire figure when i == 100
            if i == 100:
                # Add a red horizontal line spanning the entire width of the plot
                # This gives the y-position of the 100th image in the figure
                line_position = (160 - 100 - 2) / 160
                plt.plot([0, 1], [line_position, line_position], color='red', linewidth=2,
                         transform=fig.transFigure,
                         clip_on=False)
        plt.suptitle(f'Generated class {l}')
        plt.tight_layout()
        dir_path = os.path.dirname(os.path.abspath(fig_file))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(fig_file)
        plt.show()
        plt.close(fig)

def print_histgram(new_probs, value_type='probs'):
    print(f'***Print histgram of {value_type}, min:{min(new_probs)}, max: {max(new_probs)}***')
    # # Convert the probabilities to numpy for histogram calculation
    # new_probs = new_probs.detach().cpu().numpy()
    # Compute histogram
    hist, bin_edges = torch.histogram(torch.tensor(new_probs), bins=5)
    # Print histogram
    for i in range(len(hist)):
        print(f"\tBin {i}: {value_type} Range ({bin_edges[i]}, {bin_edges[i + 1]}), Frequency: {hist[i]}")


def print_histories(histories):
    num_server_epoches = len(histories)
    num_clients = len(histories[0])
    num_classes = len(histories[0][0]["gan"])
    print('num_server_epoches:', num_server_epoches, ' num_clients:', num_clients, ' num_classes:', num_classes)
    # for c in range(num_clients):
    #     print(f"\n\nclient {c}")
    #     for s in range(num_server_epoches):
    #         client = histories[s][c]
    #         local_gan = client['gan']
    #         local_cnn = client['cnn']
    #         print(f'\t*local gan:', local_gan.keys(), f' server_epoch: {s}')
    #         losses_ = [float(f"{v:.2f}") for v in local_gan['losses']]
    #         # print(f'\t\tlocal gan ({len(losses_)}): {losses_[:5]} ... {losses_[-5:]}')
    #         print(f'\tlocal gan ({len(losses_)}): [{", ".join(map(str, losses_[:5]))}, ..., '
    #               f'{", ".join(map(str, losses_[-5:]))}]')
    #         # print('\t*local cnn:', [f"{v:.2f}" for v in local_cnn['losses']])
    #         # labeled_acc = client['labeled_accuracy']
    #         # unlabeled_acc = client['unlabeled_accuracy']
    #         # shared_acc = client['shared_accuracy']
    #         # print(f'\t\tlabeled_acc:{labeled_acc:.2f}, unlabeled_acc:{unlabeled_acc:.2f},
    #         # shared_acc:{shared_acc:.2f}')

    c = num_clients
    for model_type in ['global', 'local']:
        print(f'\n***model_type: {model_type}***')
        ncols = 2
        nrows, r = divmod(c, ncols)
        nrows = nrows if r == 0 else nrows + 1
        fig, axes = plt.subplots(nrows, ncols)
        for c in range(num_clients):
            i, j = divmod(c, ncols)
            print(f"\nclient {c}")
            train_accs = []  # train
            val_accs = []
            unlabeled_accs = []  # test
            shared_accs = []
            for s in range(num_server_epoches):
                client = histories[s][c]
                # local_gan = client['gan']
                # local_cnn = client['cnn']
                # print(f'\t*local gan:', local_gan.keys(), f' server_epoch: {s}')
                # losses_ = [float(f"{v:.2f}") for v in local_gan['losses']]
                # print(f'\t\tlocal gan:', losses_)
                # # print('\t*local cnn:', [f"{v:.2f}" for v in local_cnn['losses']])
                train_acc = client[f'{model_type}_train_accuracy']
                val_acc = client[f'{model_type}_val_accuracy']
                test_acc = client[f'{model_type}_test_accuracy']
                shared_acc = client[f'{model_type}_shared_accuracy']
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
            axes[i, j].set_title(f'Client_{c}')
            axes[i, j].legend(fontsize='small')

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
            axes[i, j].legend(fontsize='small')

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


def normalize(X):
    return (X / 255.0 - 0.5) * 2  # [-1, 1]


def clients_training(epoch, global_cnn):
    clients_cnns = {}
    clients_info = {}  # extra information (e.g., number of samples) of clients that can be used in aggregation
    history = {}

    train_dataset = datasets.MNIST(root="./data", train=True, transform=None, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)
    X_test = test_dataset.data
    y_test = test_dataset.targets
    X_test = normalize(X_test.numpy())
    y_test = y_test.numpy()
    shared_data = {"X": torch.tensor(X_test).float().to(DEVICE), 'y': torch.tensor(y_test).to(DEVICE)}

    X = train_dataset.data  # Tensor of shape (60000, 28, 28)
    y = train_dataset.targets  # Tensor of shape (60000,)
    X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()
    num_samples = len(y)

    random_state = 42
    torch.manual_seed(random_state)
    indices = torch.randperm(num_samples)  # Randomly shuffle
    step = int(num_samples / NUM_BENIGN_CLIENTS)
    # step = 10  # for debugging
    ########################################### Benign Clients #############################################
    for c in range(NUM_BENIGN_CLIENTS):
        client_type = 'benign'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        X_c = X[indices[c * step:(c + 1) * step]]
        y_c = y[indices[c * step:(c + 1) * step]]
        # might be used in server
        train_info = {"client_type": client_type, "gan": {}, "cnn": {}, 'client_id': c, 'server_epoch': epoch}
        # Create indices for train/test split
        num_samples_client = len(y_c)
        indices_sub = np.arange(num_samples_client)
        train_indices, test_indices = train_test_split(indices_sub, test_size=0.2,
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
        clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
        print(f'client_{c} data:', label_cnts)
        print_data(local_data)

        print('Train CNN...')
        # local_cnn = CNN(input_dim=input_dim, hidden_dim=hidden_dim_cnn, output_dim=num_classes)
        local_cnn = CNN(num_classes=NUM_CLASSES)
        train_cnn(local_cnn, global_cnn, local_data, train_info)
        clients_cnns[c] = local_cnn.state_dict()

        print('Evaluate CNNs...')
        evaluate(local_cnn, local_data, DEVICE, global_cnn,
                 test_type='Client data', client_id=c, train_info=train_info)
        evaluate_shared_test(local_cnn, local_data, DEVICE, global_cnn, None,
                             test_type='Shared test data', client_id=c, train_info=train_info)

        history[c] = train_info

    ########################################### Byzantine Clients #############################################
    for c in range(NUM_BENIGN_CLIENTS, NUM_BENIGN_CLIENTS + NUM_BYZANTINE_CLIENTS, 1):
        client_type = 'attacker'
        print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
        X_c = X[indices[(c - NUM_BENIGN_CLIENTS) * step:((c - NUM_BENIGN_CLIENTS) + 1) * step]]
        y_c = y[indices[(c - NUM_BENIGN_CLIENTS) * step:((c - NUM_BENIGN_CLIENTS) + 1) * step]]
        # might be used in server
        train_info = {"client_type": client_type, "gan": {}, "cnn": {}, 'client_id': c, 'server_epoch': epoch}
        train_info['NUM_BYZANTINE_CLIENTS'] = NUM_BYZANTINE_CLIENTS
        local_data = {'client_type': client_type,
                      'X': torch.tensor(X_c).to(DEVICE), 'y': torch.tensor(y_c).to(DEVICE),
                      'train_mask': torch.tensor(train_mask, dtype=torch.bool).to(DEVICE),
                      'val_mask': torch.tensor(val_mask, dtype=torch.bool).to(DEVICE),
                      'test_mask': torch.tensor(test_mask, dtype=torch.bool).to(DEVICE),
                      'shared_data': shared_data}
        label_cnts = collections.Counter(local_data['y'].tolist())
        clients_info[c] = {'label_cnts': label_cnts, 'size': len(local_data['y'])}

        local_cnn = CNN(num_classes=NUM_CLASSES).to(DEVICE)
        # Assign large values to all parameters
        BIG_NUMBER = 1e2  # Example: Set all weights and biases to 1,000,000
        for param in local_cnn.parameters():
            param.data.fill_(BIG_NUMBER)  # Assign big number to each parameter
        clients_cnns[c] = local_cnn.state_dict()
        # train_info['cnn']['data'] = (local_data['X'].float().to(DEVICE), local_data['y'].to(DEVICE),
        #                              local_data["train_mask"], local_data["val_mask"],
        #                              local_data["test_mask"])

        history[c] = train_info

    return clients_cnns, clients_info, history


@timer
def main():
    print(f"\n***************************** Global Models *************************************")
    global_cnn = CNN(num_classes=NUM_CLASSES)
    print(global_cnn)

    histories = {'clients': [], 'server': []}
    for server_epoch in range(SERVER_EPOCHS):
        print(f"\n***************************** {server_epoch}: Client Training ********************************")
        clients_cnns, clients_info, history = clients_training(server_epoch, global_cnn)
        histories['clients'].append(history)

        print(f"\n***************************** {server_epoch}: Sever Aggregation *******************************")
        aggregate_cnns(clients_cnns, clients_info, global_cnn, AGGREGRATION_METHOD, histories, server_epoch)

    prefix = f'-n_{SERVER_EPOCHS}'
    # history_file = f'{IN_DIR}/histories_gan_{prefix}.pth'
    # print(f'saving histories to {history_file}')
    # with open(history_file, 'wb') as f:
    #     pickle.dump(histories, f)
    # torch.save(histories, history_file)

    try:
        print_histories(histories['clients'])
    except Exception as e:
        print('error: ', e)
    print_histories_server(histories['server'])


if __name__ == '__main__':
    IN_DIR = 'fl/mnist'
    LABELS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    # LABELS = {0, 1}
    NUM_CLASSES = len(LABELS)
    GAN_UPDATE_FREQUENCY = 1
    GAN_UPDATE_FLG = True
    print(f'IN_DIR: {IN_DIR}, AGGREGRATION_METHOD: {AGGREGRATION_METHOD}, '
          f'NUM_BENIGN_CLIENTS: {NUM_BENIGN_CLIENTS}, NUM_BYZANTINE_CLIENTS: {NUM_BYZANTINE_CLIENTS}'
          f'num_classes: {NUM_CLASSES}, where classes: {LABELS}')
    main()
