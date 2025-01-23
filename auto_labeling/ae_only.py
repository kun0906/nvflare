import collections

from sklearn.metrics import accuracy_score, confusion_matrix
from torch_geometric.datasets import Planetoid

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

import argparse
import collections
import multiprocessing as mp
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

from attention import aggregate_with_attention
from utils import timer

print(os.path.abspath(os.getcwd()))

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set print options for 2 decimal places
torch.set_printoptions(precision=1, sci_mode=False)

LABELs = {0, 1, 2, 3, 4, 5, 6}


def L2(x1, x2):
    return np.linalg.norm(x1 - x2, axis=1)


@timer
def evaluate(global_vaes, X_train, y_train, X_val, y_val, X_test, y_test, X_shared_test, y_shared_test):
    """
        Evaluate how well each client's model performs on the test set.

        client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        client_data_ =  (graph_data_, feature_info, client_data_)
    """
    alpha = 0

    global_extra = {l: extra for l, (vae, extra) in global_vaes.items()}
    for data_type, X, y in [('train', X_train, y_train),
                            ('val', X_val, y_val),
                            ('test', X_test, y_test),
                            ('shared_test', X_shared_test, y_shared_test)

                            ]:
        best_preds = np.zeros((len(X), 0))
        info = {}
        X = torch.tensor(X).to(device)
        for l, (vae, extra) in global_vaes.items():
            # print(f'***Testing {model_type} model on {test_type} with vae_{l}...')
            vae = vae.to(device)

            vae.eval()

            with (torch.no_grad()):
                recon_X, mu, logvar = vae(X)
                info[l] = (mu, logvar)
                # recon_error = np.linalg.norm(recon_X.cpu().numpy() - X.cpu().numpy(), axis=1)
                recon_error = binary_cross_entropy(recon_X.cpu().numpy(), X.cpu().numpy())
                # recon_error = (recon_error - global_extra[l]['min_recon']) / (global_extra[l]['max_recon']
                #                                                               - global_extra[l]['min_recon'])
                l2 = L2(global_extra[l]['mu'], mu.detach().cpu().numpy()) + \
                     L2(global_extra[l]['std'], logvar.detach().cpu().numpy())
                recon_error = l2 + alpha * recon_error
                best_preds = np.concatenate((best_preds, recon_error.reshape((-1, 1))), axis=1)

        # print(f'\n\n global_extra', global_extra.items())
        predicted_labels = np.argmin(best_preds, axis=1)
        # plot_latent(info, train_info, global_extra, title='local')

        # Calculate accuracy for the labeled data
        # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        # print(f'labeled_indices {len(labeled_indices)}')
        # true_labels = y

        y_true = y
        y_pred = predicted_labels

        # Total samples and number of classes
        total_samples = len(y_true)
        # Compute class weights
        class_weights = {c: total_samples / count for c, count in collections.Counter(y_true.tolist()).items()}
        sample_weight = [class_weights[y_0.item()] for y_0 in y_true]
        print(f'class_weights: {class_weights}')

        accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)

        # train_info[f'{model_type}_{data_type}_accuracy'] = accuracy
        print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y_true.tolist())}")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy
        # print(y, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        conf_matrix = conf_matrix.astype(int)
        # train_info[f'{model_type}_{data_type}_cm'] = conf_matrix
        print("Confusion Matrix:\n", conf_matrix)

        # print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


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
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm}

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


def load_data():
    data_file = 'data.pt'
    if os.path.exists(data_file):
        return torch.load(data_file, weights_only=False)

    # Load the Cora dataset
    dataset = Planetoid(root='./data', name='Cora', split='full')

    # Access the first graph in the dataset
    data = dataset[0]

    # Dataset summary
    print(f"Dataset Summary:\n"
          f"- Number of Nodes: {data.num_nodes}\n"
          f"- Number of Edges: {data.num_edges}\n"
          f"- Node Feature Size: {data.x.shape[1]}\n"
          f"- Number of Classes: {dataset.num_classes}")

    # Extract data into NumPy arrays
    X = data.x.numpy()
    Y = data.y.numpy()
    edge_index = data.edge_index
    # # edge_indices = set([(row[0], row[1]) for row in edge_indices])
    # edge_indices = set(map(tuple, data.edge_index.numpy().T))
    # unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in data.edge_index.numpy().T])
    # print(f'unique edges: {len(unqiue_edges)} =? edge_indices/2: {len(edge_indices) / 2}, '
    #       f'edges: {data.edge_index.shape}')

    X_train = X[data.train_mask]
    y_train = Y[data.train_mask]
    X_val = X[data.val_mask]
    y_val = Y[data.val_mask]
    X_test = X[data.test_mask]
    y_test = Y[data.test_mask]

    torch.save((X, Y, X_val, y_val, X_test, y_test), data_file)

    return X, Y, X_val, y_val, X_test, y_test


def check_gen_data(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test):
    print('\n\nX_train, y_train as training set')
    ml_info = evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_gen_test, y_gen_test, verbose=10)

    print('\n\nX_gen_test, y_gen_test as training set')
    ml_info2 = evaluate_ML2(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test, verbose=10)

    return ml_info


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AE, self).__init__()
        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_fc3 = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim

        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_fc3 = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        """Encode input into latent space parameters (mu, logvar)."""
        h = F.relu(self.encoder_fc1(x))
        # h = F.relu(self.encoder_fc2(h))
        z = F.relu(self.encoder_fc3(h))

        return z

    def decoder(self, z):
        """Decode latent representation into reconstructed input."""
        h = F.relu(self.decoder_fc1(z))
        # h = F.relu(self.decoder_fc2(h))
        x_recon = torch.sigmoid(self.decoder_fc3(h))

        return x_recon

    def forward(self, x):
        """Forward pass through the entire AE."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, z


# MMD Loss
def mmd_loss(real_features, fake_features):
    x_kernel = real_features @ real_features.T
    y_kernel = fake_features @ fake_features.T
    xy_kernel = real_features @ fake_features.T
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


# AE loss function
def vae_loss_function(recon_x, x, mean, log_var, beta=1.):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / (x.shape[0] * x.shape[1]) == reduction='mean'
    # print(np.min(recon_x.detach().numpy()), np.max(recon_x.detach().numpy()), flush=True)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]
    # BCE = F.mse_loss(recon_x, x, reduction='mean')
    # KLD loss for the latent space
    # This assumes a unit Gaussian prior for the latent space
    # (Normal distribution prior, mean 0, std 1)
    # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # KL divergence between q(z|x) and p(z) (standard normal)
    # Normalized by the batch size for stability
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # This is the regularization term that encourages meaningful latent space
    # (also known as the latent space regularization term)
    # When beta > 1, the model is encouraged to use the latent space more effectively
    # It's controlled by the beta value in β-AE
    # Latent loss (KL divergence)
    # You can adjust this term using 'beta' to scale the importance of the latent space regularization
    # The larger the beta, the more emphasis on KL divergence
    # If beta is too large, the model might ignore reconstruction and over-regularize
    # If beta is too small, the model might ignore latent space regularization
    # Hence, a reasonable balance is required.
    KLD = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())) / x.shape[0]
    return (BCE + beta * KLD), {'BCE': BCE.item(), 'KLD': KLD.item()}


def binary_cross_entropy(recon_x, x):
    """
    Compute binary cross-entropy loss.

    Parameters:
        recon_x (ndarray): Predicted probabilities (values between 0 and 1).
        x (ndarray): True binary labels (0 or 1).
        reduction (str): 'sum', 'mean', or 'none' for the type of reduction.

    Returns:
        float or ndarray: The computed binary cross-entropy loss.
    """
    # Ensure recon_x values are in a safe range to avoid log(0)
    epsilon = 1e-7
    recon_x = np.clip(recon_x, epsilon, 1 - epsilon)

    # Binary cross-entropy formula
    bce_loss = - (x * np.log(recon_x) + (1 - x) * np.log(1 - recon_x))

    return np.sum(bce_loss, axis=1)


def train_ae(X_train, y_train, X_val, y_val, X_test, y_test):
    AE_EPOCHs = 10001
    aes_file = f'aes_{AE_EPOCHs}.pt'
    if os.path.exists(aes_file):
        return torch.load(aes_file)

    input_dim = X_train.shape[1]
    hidden_dim_vae = 128
    latent_dim = 5

    vaes = {l: (AE(input_dim=input_dim, hidden_dim=hidden_dim_vae, latent_dim=latent_dim),
                {'min_recon': 10000, 'max_recon': 0,
                 'mu': np.zeros((1, latent_dim)), 'std': np.zeros((1, latent_dim))}) for l in LABELs}
    for l, (local_vae, local_extra) in vaes.items():
        X, y = X_train, y_train

        label_mask = y == l
        if sum(label_mask) == 0:
            continue

        print(f'training ae for class {l}...')
        X = X[label_mask]
        y = y[label_mask]

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.int)

        # Only update available local labels, i.e., not all the local_aes will be updated.
        # local_labels = set(y.tolist())
        print(f'local labels: {collections.Counter(y.tolist())}, with {len(y)} samples.')

        # Initialize the models, optimizers, and loss function
        # z_dim = 100  # Dimension of random noise
        data_dim = X.shape[1]  # Number of features (e.g., Cora node features)
        # lr = 0.0002

        local_vae.to(device)
        optimizer = optim.Adam(local_vae.parameters(), lr=0.0001, weight_decay=5e-5)  # L2
        # Define a scheduler
        scheduler = StepLR(optimizer, step_size=500, gamma=0.8)

        # ohe_labels = torch.zeros((len(y), len(LABELs))).to(device)  # One-hot encoding for class labels
        # for i, l in enumerate(y.tolist()):  # if y is different.
        #     ohe_labels[i, l] = 1
        # new_labeled_X = local_data['X'][predicted_labels == l]
        # X = torch.cat((X, new_labeled_X), dim=0).to(device)

        X = X.clone().detach().float().to(device)
        print(f'local X.shape: {X.shape}')
        losses = []
        # sigma = compute_sigma(X)
        BETA = 0
        print(f'vae_epochs: {AE_EPOCHs}, beta:{BETA}')
        for epoch in range(AE_EPOCHs):
            # Convert X to a tensor and move it to the device
            # X.to(device)
            recon_logits, mu, logvar = local_vae(X)

            # print(np.min(mu.detach().numpy()), np.max(mu.detach().numpy()), [(np.min(vs.detach().numpy()), np.max(vs.detach().numpy())) for vs in local_vae.parameters()], flush=True)

            # recon_logits = torch.clamp(recon_logits, min=1e-7, max=1 - 1e-7)
            loss, info = vae_loss_function(recon_logits, X, mu, logvar, beta=BETA)
            # if loss < 1e-10:
            #     break

            # MMD loss
            # mmd_loss = compute_mmd(X, recon_logits, sigma)
            # alpha = 5.0
            # loss += alpha * mmd_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to stabilize the training
            torch.nn.utils.clip_grad_norm_(local_vae.parameters(), max_norm=0.8)

            optimizer.step()

            # Update learning rate
            scheduler.step()

            if epoch == AE_EPOCHs - 1:
                recon_errors = binary_cross_entropy(recon_logits.detach().cpu().numpy(), X.cpu().numpy())
                local_extra['min_recon'] = min(recon_errors)
                local_extra['max_recon'] = max(recon_errors)
                local_extra['mu'] = np.mean(mu.detach().cpu().numpy(), axis=0)
                local_extra['std'] = np.mean(logvar.detach().cpu().numpy(), axis=0)

            if epoch % 100 == 0:
                print(f'train_vae epoch: {epoch}, local_vae loss: {loss.item():.4f}, {info.items()}, '
                      # f'mmd:{mmd_loss.item()}, sigma:{sigma:.4f} '
                      f'LR: {scheduler.get_last_lr()}')
            losses.append(loss.item())

        # train_info[f'vae_{l}'] = {"losses": losses}

    vaes[l] = (local_vae, local_extra)

    torch.save(vaes, f'aes_{AE_EPOCHs}.pt')

    return vaes


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(len(y_train), collections.Counter(y_train.tolist()))
    print(len(y_val), collections.Counter(y_val.tolist()))
    print(len(y_test), collections.Counter(y_test.tolist()))

    aes = train_ae(X_train, y_train, X_val, y_val, X_test, y_test)

    evaluate(aes, X_train, y_train, X_val, y_val, X_test, y_test, X_test, y_test)


if __name__ == '__main__':
    main()
