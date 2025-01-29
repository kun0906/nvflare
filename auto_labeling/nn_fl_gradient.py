"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare/auto_labeling
    $PYTHONPATH=. python3 nn_fl_gradient.py


"""

import argparse
import collections
import os

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from auto_labeling.load_data import load_data
from utils import timer

print(os.path.abspath(os.getcwd()))

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set print options for 2 decimal places
torch.set_printoptions(precision=1, sci_mode=False)

DATA = 'cora'
if DATA == 'cora':
    LABELs = {0, 1, 2, 3, 4, 5, 6}
elif DATA == 'mnist':
    LABELs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
else:
    raise ValueError


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedGNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--label_rate', type=float, required=False, default=1e-3,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-l', '--hidden_dimension', type=int, required=False, default=256,
                        help="The hidden dimension of GNN.")
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=5000,
                        help="The number of epochs (integer).")
    parser.add_argument('-p', '--patience', type=float, required=False, default=1.0,
                        help="The patience.")
    # parser.add_argument('-a', '--vae_epochs', type=int, required=False, default=10,
    #                     help="vae epochs.")
    # parser.add_argument('-b', '--vae_beta', type=float, required=False, default=1.0,
    #                     help="vae loss beta.")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
LR = args.label_rate
EPOCHs = args.server_epochs
hidden_dim_nn = args.hidden_dimension
num_layers = int(args.patience)
# For testing, print the parsed parameters
# print(f"label_rate: {label_rate}")
# print(f"server_epochs: {server_epochs}")
print(args)


def L2(x1, x2):
    return np.linalg.norm(x1 - x2, axis=1)



def check_gen_data(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test):
    print('\n\nX_train, y_train as training set')
    ml_info = evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_gen_test, y_gen_test, verbose=10)

    print('\n\nX_gen_test, y_gen_test as training set')
    ml_info2 = evaluate_ML2(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test, verbose=10)

    return ml_info


#
# class NN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, out_dim):
#         super(NN, self).__init__()
#         # Encoder layers
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, out_dim)
#
#     def forward(self, x):
#         """Forward pass."""
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout_prob=0.2):
        super(NN, self).__init__()

        # Encoder layers with Batch Normalization and Dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch Normalization
        # self.dropout1 = nn.Dropout(dropout_prob)  # Dropout layer

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Normalization
        # self.dropout2 = nn.Dropout(dropout_prob)  # Dropout layer

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        # self.bn3 = nn.BatchNorm1d(out_dim)  # Batch Normalization

        # Parametric ReLU activation function (learnable parameter)
        # self.prelu = nn.PReLU()

    def forward(self, x):
        """Forward pass with Batch Normalization and Dropout."""
        x = self.fc1(x)
        # x = self.bn1(x)  # Apply Batch Normalization
        # x = F.relu(x)
        x = F.leaky_relu(x)  # better than relu()
        # x = self.prelu(x)
        # x = self.dropout1(x)  # Apply Dropout

        for _ in range(num_layers):
            x2 = self.fc2(x)
            # x2 = self.bn2(x2)  # Apply Batch Normalization
            x = F.leaky_relu(x2)
            # x = self.prelu(x2)
            # x = self.dropout2(x)  # Apply Dropout

        x = self.fc3(x)
        # x = self.bn3(x)  # Apply Batch Normalization
        x = F.leaky_relu(x)
        # x = self.prelu(x)
        return x


def L2(params1, params2):
    return sum((torch.norm(p1 - p2) ** 2 for p1, p2 in zip(params1, params2))).sqrt()


def train_nn(X_train, y_train, X_val, y_val, X_test, y_test):
    # EPOCHs = 2001
    nns_file = f'nns_{EPOCHs}.pt'
    if os.path.exists(nns_file):
        return torch.load(nns_file)

    input_dim = X_train.shape[1]
    # hidden_dim_nn = input_dim // 2
    num_classes = len(LABELs)

    nns = {l: (NN(input_dim=input_dim, hidden_dim=hidden_dim_nn, out_dim=num_classes),
               {'min_recon': 10000, 'max_recon': 0,
                'mu': np.zeros((1,)), 'std': np.zeros((1,))}) for l in LABELs}

    global_nn = NN(input_dim=input_dim, hidden_dim=hidden_dim_nn, out_dim=num_classes)
    global_nn = global_nn.to(device)
    global_optimizer = optim.Adam(global_nn.parameters(), lr=LR, weight_decay=5e-4)  # L2
    # Define a scheduler
    global_scheduler = StepLR(global_optimizer, step_size=1000, gamma=0.95)

    total_samples = len(y_train)
    class_weights = {c: total_samples / count for c, count in collections.Counter(y_train.tolist()).items()}
    # sample_weight = [class_weights[y_0.item()] for y_0 in y_train]
    history = []
    for server_epoch in range(EPOCHs):
        if server_epoch % 100 == 0:
            print(f'\n***server_epoch: {server_epoch}')

        global_gradients = []
        losses = []
        for l, (local_nn, local_extra) in nns.items():  # each client only has one class data

            # Initialize local_vae with
            local_nn.load_state_dict(global_nn.state_dict())

            local_nn.to(device)

            X, y = X_train, y_train

            label_mask = y == l
            if sum(label_mask) == 0 or l not in LABELs:
                continue

            # print(f'training nn for class {l}...')
            X = X[label_mask]
            y = y[label_mask]

            # random select 100 sample for training
            m = len(X) // 10  # for each client, we only use a subset of data to compute gradient
            if m < 10:
                print(m, len(X))
                m = 10
            indices = torch.randperm(len(X))[:m]  # Randomly shuffle and pick the first 10 indices
            X = X[indices]
            y = y[indices]

            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)

            # Only update available local labels, i.e., not all the local_nns will be updated.
            # local_labels = set(y.tolist())
            # print(f'local labels: {collections.Counter(y.tolist())}, with {len(y)} samples.')

            optimizer = optim.Adam(local_nn.parameters(), lr=0.0001, weight_decay=5e-4)  # L2
            # criterion = CrossEntropyLoss(reduction='sum')
            criterion = CrossEntropyLoss(reduction='sum', weight=torch.tensor([class_weights[l_] for l_ in LABELs])).to(
                device)

            X = X.clone().detach().float().to(device)

            client_epochs = 1
            for j in range(client_epochs):
                # print(f'local X.shape: {X.shape}')
                pred_logits = local_nn(X)
                y = y.clone().detach().to(device)
                # print(pred_logits.device, y.device, flush=True)
                loss = criterion(pred_logits, y)

                optimizer.zero_grad()
                loss.backward()  # compute the gradients

                if client_epochs > 1:
                    optimizer.step()  # update local parameters

            local_gradients = [param.grad.clone() for param in local_nn.parameters()]
            # Send gradients to server for aggregation
            global_gradients.append(local_gradients)
            # losses.append(loss)
            nns[l] = (local_nn, local_extra)

        # Server aggregates gradients
        # each client will send the gradients and the corresponding class to each gradients
        # server aggregate by each class first to filter outlier gradients using Krum.
        aggregated_gradients = []
        for param_idx in range(len(global_gradients[0])):
            param_grads = [client_grads[param_idx] for client_grads in global_gradients]
            # aggregated_gradients.append(torch.mean(torch.stack(param_grads), dim=0))
            aggregated_gradients.append(torch.sum(torch.stack(param_grads), dim=0) / total_samples)

        old_params = [param.clone() for param in global_nn.parameters()]
        flg = True
        if not flg:
            # Update global model using aggregated gradients
            learning_rate = 0.01
            with torch.no_grad():
                for param, grad in zip(global_nn.parameters(), aggregated_gradients):
                    if grad is not None:
                        param -= learning_rate * grad
        else:
            # You will set the gradients of the global model parameters directly to the aggregated gradients
            with torch.no_grad():
                for param, grad in zip(global_nn.parameters(), aggregated_gradients):
                    if grad is not None:
                        param.grad = grad.to(device)  # Assign aggregated gradient to the parameter's grad

            # Now perform a single step of optimization
            global_optimizer.step()  # Update the global model parameters

            global_scheduler.step()

            # Reset gradients for the next iteration
            global_optimizer.zero_grad()

        if server_epoch % 100 == 0:
            print(f'server_epoch: {server_epoch}, grad diff: {L2(global_nn.parameters(), old_params)}')

        info = evaluate({0: (global_nn, None)}, X_train, y_train, X_val, y_val, X_test, y_test, X_test, y_test)
        history.append(info)

    print_histories_server(history)
    return nns


def print_histories_server(histories_server):
    num_server_epoches = len(histories_server)

    for model_type in ['global']:
        print(f'\n***model_type: {model_type}***')
        ncols = 2
        nrows, r = divmod(4, ncols)
        nrows = nrows if r == 0 else nrows + 1
        fig, axes = plt.subplots(nrows, ncols)
        for clf_idx, clf_name in enumerate(['NN']):
            i, j = divmod(clf_idx, ncols)
            print(f"\n {clf_name}")
            train_accs = []  # train
            val_accs = []
            unlabeled_accs = []  # test
            shared_accs = []
            for s in range(num_server_epoches):
                ml_info = histories_server[s]
                #  ml_info[clf_name] = {test_type: {'accuracy': accuracy, 'cm': cm}}
                train_acc = ml_info['train'][0]['accuracy']
                val_acc = ml_info['val'][0]['accuracy']
                test_acc = ml_info['test'][0]['accuracy']
                shared_acc = ml_info['shared_test'][0]['accuracy']

                train_accs.append(train_acc)
                val_accs.append(val_acc)
                unlabeled_accs.append(test_acc)
                shared_accs.append(shared_acc)
                print(f'\t\tEpoch: {s}, train:{train_acc:.2f}, val:{val_acc:.2f}, '
                      f'test:{test_acc:.2f}, '
                      f'shared_test:{shared_acc:.2f}')

            # Training and validation loss on the first subplot
            axes[i, j].plot(range(len(train_accs)), train_accs, label='train', marker='o')
            axes[i, j].plot(range(len(val_accs)), val_accs, label='val', marker='o')
            axes[i, j].plot(range(len(unlabeled_accs)), unlabeled_accs, label='test', marker='+')
            axes[i, j].plot(range(len(shared_accs)), shared_accs, label='shared_test', marker='s')
            axes[i, j].set_xlabel('Server Epochs')
            axes[i, j].set_ylabel('Accuracy')
            axes[i, j].set_title(f'{clf_name}')
            axes[i, j].legend(fontsize='small')

        if model_type == 'global':
            title = f'{model_type}_nn' + '$_{' + f'{num_server_epoches}' + '}$' + f':{LR}'
        else:
            title = f'{model_type}_nn' + '$_{' + f'{num_server_epoches}+1' + '}$' + f':{LR}'
        plt.suptitle(title)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig_file = f'tmp/{num_server_epoches}_accuracy.png'
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300)
        plt.show()


@timer
def evaluate(local_nns, X_train, y_train, X_val, y_val, X_test, y_test, X_shared_test, y_shared_test):
    """
        Evaluate how well each client's model performs on the test set.

        client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        client_data_ =  (graph_data_, feature_info, client_data_)
    """
    alpha = 0
    info = {}
    # global_extra = {l: extra for l, (nn, extra) in global_nn.items()}
    for data_type, X, y in [('train', X_train, y_train),
                            ('val', X_val, y_val),
                            ('test', X_test, y_test),
                            ('shared_test', X_shared_test, y_shared_test)

                            ]:
        # best_preds = np.zeros((len(X), 0))
        info_data_type = {}
        X = torch.tensor(X).to(device)
        for l, (nn, extra) in local_nns.items():
            # print(f'***Testing {model_type} model on {test_type} with nn_{l}...')
            nn = nn.to(device)

            nn.eval()

            with (torch.no_grad()):
                pred_logits = nn(X)
                y_pred = torch.argmax(pred_logits, dim=-1).cpu().numpy()

                y_true = y

                # Total samples and number of classes
                total_samples = len(y_true)
                # Compute class weights
                class_weights = {c: total_samples / count for c, count in collections.Counter(y_true.tolist()).items()}
                sample_weight = [class_weights[y_0.item()] for y_0 in y_true]
                sample_weight = [1] * len(y_true)
                print(f'class_weights: {class_weights}')

                accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)

                # train_info[f'{model_type}_{data_type}_accuracy'] = accuracy
                print(
                    f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y_true.tolist())}")
                # if 'all' in test_type:
                #     client_result['labeled_accuracy_all'] = accuracy
                # else:
                #     client_result['labeled_accuracy'] = accuracy
                # print(y, y_pred)
                conf_matrix = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
                conf_matrix = conf_matrix.astype(int)
                # train_info[f'{model_type}_{data_type}_cm'] = conf_matrix
                print("Confusion Matrix:\n", conf_matrix)

                # auc = roc_auc_score(y_true, y_pred_probs)

                info_data_type[l] = {'accuracy': accuracy, 'conf_matrix': conf_matrix}

        info[data_type] = info_data_type
        # break
        # print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return info


def evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test,
                 X_shared_test=None, y_shared_test=None, verbose=10):
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
    # clfs = {'Random Forest': rf}
    clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm}

    # all_data = client_data['all_data']
    # test_mask = all_data['test_mask']
    # X_shared_test = all_data['X'][test_mask]
    # y_shared_test = all_data['y'][test_mask]
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
            sample_weight = [class_weights[y_0.item()] for y_0 in y_]
            sample_weight = [1] * len(y_)
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


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data=DATA)
    print(len(y_train), collections.Counter(y_train.tolist()))
    print(len(y_val), collections.Counter(y_val.tolist()))
    print(len(y_test), collections.Counter(y_test.tolist()))

    # BEST 75.8% for Cora, 96.9%  for MNIST
    # evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_test, y_test)

    nn = train_nn(X_train, y_train, X_val, y_val, X_test, y_test)

    # evaluate(nn, X_train, y_train, X_val, y_val, X_test, y_test, X_test, y_test)


if __name__ == '__main__':
    main()
