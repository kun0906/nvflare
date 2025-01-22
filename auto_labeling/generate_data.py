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
        return torch.load(data_file)

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


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=128, output_dim=-1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh()  # Outputs between -1 and 1
            nn.Sigmoid()  # Outputs between 0 and 1
        )
        self.latent_dim = latent_dim

    def forward(self, z):
        output = self.model(z)
        # binary_output = (output >= 0.5).float()  # Apply thresholding to get binary outputs
        # output = F.gumbel_softmax(output, tau=1, hard=True)  # Differentiable approximation
        return output


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs probability
        )

    def forward(self, x):
        return self.model(x)


# MMD Loss
def mmd_loss(real_features, fake_features):
    x_kernel = real_features @ real_features.T
    y_kernel = fake_features @ fake_features.T
    xy_kernel = real_features @ fake_features.T
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def train_gan(X_train, y_train, X_val, y_val, X_test, y_test):
    epochs = 10001
    gans_file = f'gans_{epochs}.pt'
    if os.path.exists(gans_file):
        return torch.load(gans_file)

    gans = {l: Generator(latent_dim=50, hidden_dim=1280, output_dim=X_train.shape[1]) for l in LABELs}
    for l, local_gan in gans.items():
        X, y = X_train, y_train

        label_mask = y == l
        if sum(label_mask) == 0:
            continue

        print(f'training gan for class {l}...')
        X = X[label_mask]
        y = y[label_mask]

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.int)

        # Only update available local labels, i.e., not all the local_gans will be updated.
        # local_labels = set(y.tolist())
        print(f'local labels: {collections.Counter(y.tolist())}, with {len(y)} samples.')

        # Initialize the models, optimizers, and loss function
        # z_dim = 100  # Dimension of random noise
        data_dim = X.shape[1]  # Number of features (e.g., Cora node features)
        # lr = 0.0002

        # generator = Generator(input_dim=z_dim, output_dim=data_dim).to(device)
        generator = local_gan
        # local_gan.load_state_dict(global_gans[l].state_dict())
        z_dim = generator.latent_dim

        discriminator = Discriminator(input_dim=data_dim, hidden_dim=512).to(device)

        optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, weight_decay=5e-5)  # L2
        scheduler_G = StepLR(optimizer_G, step_size=1000, gamma=0.8)

        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0005, weight_decay=5e-5)  # L2
        scheduler_D = StepLR(optimizer_D, step_size=1000, gamma=0.8)
        # optimizer_G = optim.Adam(generator.parameters(), lr=lr)
        # optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

        adversarial_loss = nn.BCELoss(reduction='mean')  # Binary Cross-Entropy Loss

        # Training loop
        losses = []
        for epoch in range(epochs):
            # ---- Train Discriminator ----
            discriminator.train()
            real_data = X.clone().detach().float().to(device)  # Replace with your local data (class-specific)
            real_data = real_data.to(device)

            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Generate synthetic data
            z = torch.randn(batch_size, z_dim).to(device)
            fake_data = generator(z).detach()  # Freeze Generator when training Discriminator

            # Discriminator Loss
            real_loss = adversarial_loss(discriminator(real_data), real_labels)
            fake_loss = adversarial_loss(discriminator(fake_data), fake_labels)
            d_loss = real_loss + fake_loss

            if epoch % 50 == 0:
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            scheduler_D.step()

            # ---- Train Generator ----
            # we don't need to freeze the discriminator because the optimizer for the discriminator
            # (optimizer_D) is not called.
            # This ensures that no updates are made to the discriminator's parameters,
            # even if gradients are computed during the backward pass.
            generator.train()
            z = torch.randn(batch_size, z_dim).to(device)
            generated_data = generator(z)

            # Generator Loss (Discriminator should classify fake data as real)
            g_loss = adversarial_loss(discriminator(generated_data), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            scheduler_G.step()
            # ---- Logging ----
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} |  "
                      f"LR_D: {scheduler_D.get_last_lr()}, LR_G: {scheduler_G.get_last_lr()}")
            losses.append(g_loss.item())

        gans[l] = generator
    torch.save(gans, f'gans_{epochs}.pt')

    return gans


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(len(y_train), collections.Counter(y_train.tolist()))
    print(len(y_val), collections.Counter(y_val.tolist()))
    print(len(y_test), collections.Counter(y_test.tolist()))

    gans = train_gan(X_train, y_train, X_val, y_val, X_test, y_test)

    sizes = {l: s for l, s in collections.Counter(y_train.tolist()).items()}
    print(sizes)
    generated_data = {}
    for l, gan in gans.items():
        size = sizes[l]

        generator = gan
        generator.eval()
        z_dim = generator.latent_dim
        with torch.no_grad():
            z = torch.randn(size, z_dim).to(device)
            gen_data = generator(z)

        mask = gen_data > 0.5
        gen_data[mask] = 1
        gen_data[~mask] = 0

        generated_data[l] = {'X': gen_data, 'y': [l] * size}

        # test on the generated data
    dim = X_train.shape[1]
    X_gen_test = np.zeros((0, dim))
    y_gen_test = np.zeros((0,), dtype=int)

    for l, vs in generated_data.items():
        X_gen_test = np.concatenate((X_gen_test, vs['X'].cpu()), axis=0)
        y_gen_test = np.concatenate((y_gen_test, vs['y']))

    check_gen_data(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == '__main__':
    main()
