import collections

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim

import argparse


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Demo Script")

    # Add arguments to be parsed
    parser.add_argument('-v', '--distill_weight', type=float, required=False, default=0.9,
                        help="The distillation weight (float).")
    parser.add_argument('-n', '--epochs', type=int, required=False, default=10,
                        help="The number of epochs (integer).")

    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
distill_weight = args.distill_weight
epochs = args.epochs

# For testing, print the parsed parameters
print(f"Distill Weight: {distill_weight}")
print(f"Epochs: {epochs}")


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # # # Create subsets of the first 100 samples
    # train_dataset = Subset(train_dataset, range(1000))
    # test_dataset = Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def load_data_for_clients():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Filter indices for classes 0-4 for Client 1 and 5-9 for Client 2
    client1_train_indices = [i for i, (x, y) in enumerate(train_dataset) if y < 5]
    client2_train_indices = [i for i, (x, y) in enumerate(train_dataset) if y >= 5]

    client1_test_indices = [i for i, (x, y) in enumerate(test_dataset) if y < 5]
    client2_test_indices = [i for i, (x, y) in enumerate(test_dataset) if y >= 5]

    # Create Subsets for both clients
    client1_train_dataset = Subset(train_dataset, client1_train_indices)
    client2_train_dataset = Subset(train_dataset, client2_train_indices)

    client1_test_dataset = Subset(test_dataset, client1_test_indices)
    client2_test_dataset = Subset(test_dataset, client2_test_indices)

    # Create DataLoaders for both clients
    client1_train_loader = DataLoader(client1_train_dataset, batch_size=64, shuffle=True)
    client2_train_loader = DataLoader(client2_train_dataset, batch_size=64, shuffle=True)

    client1_test_loader = DataLoader(client1_test_dataset, batch_size=64, shuffle=False)
    client2_test_loader = DataLoader(client2_test_dataset, batch_size=64, shuffle=False)

    return (client1_train_loader, client1_test_loader), (client2_train_loader, client2_test_loader)


"""
1. Define the Client Model
Each client can have a different model architecture. For simplicity, we'll define two models: ClientModelA and ClientModelB.
"""


# Example models
class ClientModelA(nn.Module):
    def __init__(self):
        super(ClientModelA, self).__init__()
        # Input is 28x28 image with 1 channel (grayscale)
        self.conv1 = nn.Conv2d(1, 32, 3)  # 32 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(32, 64, 3)  # 64 filters, 3x3 kernel
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusting for the output size after conv/pool
        self.fc2 = nn.Linear(128, 32)  # Reduced size for the hidden layer
        self.fc3 = nn.Linear(32, 10)  # Output 10 classes for MNIST digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ClientModelB(nn.Module):
    def __init__(self):
        super(ClientModelB, self).__init__()
        # Input is 28x28 image with 1 channel (grayscale)
        self.conv1 = nn.Conv2d(1, 32, 3)  # 32 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(32, 64, 3)  # 64 filters, 3x3 kernel
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusting for the output size after conv/pool
        self.fc2 = nn.Linear(128, 64)  # Reduced size for the hidden layer
        self.fc3 = nn.Linear(64, 10)  # Output 10 classes for MNIST digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
The server aggregates logits from clients and uses them to update the global model.
the server uses a NN to find the distribution of (all clients) logits 
"""


class VAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, latent_dim=5):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean of latent distribution
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent distribution
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # print(x.device, self.fc1.weight.device, self.fc1.bias.device)
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))  # Sigmoid for normalized output
        return torch.softmax(self.fc4(h3), dim=1)  # softmax for normalized output

    def forward(self, x):
        mu, logvar = self.encode(x)  # Flatten input logits
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    # reconstruction error
    # BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence term
    # We assume standard Gaussian prior
    # The KL term forces the latent distribution to be close to N(0, 1)
    # KL[Q(z|x) || P(z)] = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where sigma^2 = exp(logvar)
    # This is the standard VAE loss function
    # recon_x: the reconstructed logits, x: the true logits
    # mu: mean, logvar: log variance of the latent distribution
    # We assume logvar is the log of variance (log(sigma^2))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Return the total loss
    beta = 0.1
    info = (recon_loss.item(), beta * kl_loss.item())
    return recon_loss + beta * kl_loss, info


def distillation_loss(client_logits, vae_logits, temperature=200.0):
    """

    Args:
        client_logits:
        vae_logits:
        temperature: Higher temperature makes the logits less sharp, promoting more diverse learning.

    Returns:

    """
    client_probs = F.log_softmax(client_logits / temperature, dim=1)
    vae_probs = F.softmax(vae_logits / temperature, dim=1)
    return F.kl_div(client_probs, vae_probs, reduction='batchmean') * (temperature ** 2)


def _gen_pseudo_logits(global_vae, label, count, max_attempts=5):
    """
    Generate pseudo logits by decoding latent vectors, ensuring they match the desired label.

    Args:
        global_vae: VAE model to use for decoding.
        label: The label we are trying to match.
        count: The number of logits required.
        max_attempts: The maximum number of times to try to generate matching logits.

    Returns:
        A list of logits matching the desired label, or empty if unable to generate enough matching logits.
    """
    if count == 0:
        return [], []

    latent_dim = global_vae.latent_dim  # Get the latent dimension from the VAE model
    attempts = 0
    new_logits, new_labels = [], []
    while attempts < max_attempts:
        # Generate latent vectors sampled from N(0, 1)
        z = torch.randn(count * 100, latent_dim).to(device)  # Increase sample size

        # Decode latent vectors into logits (predictions)
        pseudo_logits = global_vae.decode(z)

        # Detach from the computation graph and move to the correct device
        pseudo_logits = pseudo_logits.detach().to(device)

        # Get predicted labels based on the decoded logits
        predicted_labels = pseudo_logits.argmax(dim=1)

        # Find logits where the predicted class matches the desired label
        for logit, label_pred in zip(pseudo_logits, predicted_labels):
            if label_pred.item() == label:
                new_logits.append(logit)
                new_labels.append(label_pred)

            if len(new_labels) >= count:
                return new_logits, new_labels

        # If not enough matching logits, increment the attempt counter and try again
        attempts += 1
        # print(f"Not enough matching logits. Trying again...")

    # If the function reaches here, it means we couldn't generate enough matching logits after max_attempts
    # print(f"Unable to generate enough matching logits for label {label} after {max_attempts} attempts.")
    return new_logits, new_labels  # Return an empty list if not enough matching logits found


def gen_pseudo_logits(global_vae, labels, all_class_labels={0, 1, 2, 3, 4, 5, 6, 7, 8, 9}):
    lb_dict = collections.Counter(labels.tolist())
    max_cnt = max(lb_dict.values())
    new_logits = []
    new_labels = []
    for label in all_class_labels:
        if label in lb_dict:
            if lb_dict[label] == max_cnt:
                continue
            else:
                cnt = max_cnt - lb_dict[label]
                logits, labels = _gen_pseudo_logits(global_vae, label, cnt)
        else:
            logits, labels = _gen_pseudo_logits(global_vae, label, max_cnt)
        new_logits.extend(logits)
        new_labels.extend(labels)
    return torch.vstack(new_logits), torch.stack(new_labels)


def weighted_cross_entropy_loss(logits, targets, all_class_labels={0, 1, 2, 3, 4, 5, 6, 7, 8, 9}):
    """
    Compute weighted cross-entropy loss for a classification task.

    Args:
    - logits (Tensor): The predicted logits (raw scores) from the model (shape: [batch_size, num_classes]).
    - targets (Tensor): The true labels (shape: [batch_size]). Each value should be an integer representing the class index.
    Returns:
    - loss (Tensor): The weighted cross-entropy loss.
    """
    # Get unique values and their counts
    unique_values, counts = torch.unique(targets, return_counts=True)
    class_weights = torch.Tensor([0] * len(all_class_labels))
    # for v,c in zip(unique_values, counts):
    #     class_weights[v] = c/torch.sum(counts)
    class_weights[unique_values] = counts / torch.sum(counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss = criterion(logits, targets)
    return loss


class FL:
    def __init__(self):
        self.results = {}

    def train(self, num_rounds, device):
        """
        The main FL loop coordinates training, aggregation, and distillation across clients.
        """
        # Load data
        train_loader, test_loader = load_data()
        (client1_train_loader, client1_test_loader), (
            client2_train_loader, client2_test_loader) = load_data_for_clients()

        n_classes = 10
        # Initialize client models
        clients = [(ClientModelA(), VAE(input_dim=n_classes, hidden_dim=32, latent_dim=5),
                    client1_train_loader, client1_test_loader),
                   (ClientModelB(), VAE(input_dim=n_classes, hidden_dim=32, latent_dim=5),
                    client2_train_loader, client2_test_loader),
                   ]
        # Instantiate model and optimizer
        global_vae = VAE(input_dim=n_classes, hidden_dim=32, latent_dim=5)
        global_vae = global_vae.to(device)
        losses = []
        for round_num in range(num_rounds):
            print(f"Round {round_num + 1}")

            # Step 1: Train clients
            client_vae_parameters = []
            for client_id, (client_model, client_vae, train_loader_, test_loader_) in enumerate(clients):
                print(f"  Training client {client_id+1}")
                client_info = {'client_id': client_id+1}
                # train vae+personal_model
                client_result = self._train_client(client_model, client_vae, global_vae, train_loader_, device,
                                                   client_info)
                client_vae_parameters.append(client_result['logits'])

                self.evaluate(client_model, train_loader_, device, test_type='train', client_id=client_id)
                self.evaluate(client_model, test_loader_, device, test_type='test', client_id=client_id)
                self.evaluate(client_model, test_loader, device, test_type='test all', client_id=client_id)

            # Step 2: Server aggregates vae parameters
            global_vae = self.aggregate(client_vae_parameters, global_vae)

        print("Federated learning completed.")

    def _train_client(self, model, vae, global_vae, train_loader, device, client_info):
        """
           Each client trains its local model and sends vae parameters to the server.
        """
        labels = []
        [labels.extend(labels_.tolist()) for images_, labels_ in train_loader]
        print(client_info, collections.Counter(labels))

        # CNN
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()  # mean

        # VAE
        # vae.load_state_dict(global_vae.state_dict())  # deep copy of global_vae parameters
        # vae = vae.to(device)
        # vae_optimizer = optim.Adam(vae.parameters(), lr=0.001)
        # # vae_criterion = nn.KLDivLoss(reduction="batchmean")
        global_vae.eval()

        model.train()
        vae.train()
        losses = []
        for epoch in range(epochs):
            epoch_model_loss = 0
            _model_loss, _model_distill_loss = 0, 0
            epoch_vae_loss = 0
            _vae_recon_loss, _vae_kl_loss = 0, 0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # your local personal model
                outputs = model(images)
                model_logits = F.softmax(outputs, dim=1)
                model_loss = criterion(model_logits, labels)  # cross entropy loss

                distillation = True
                if distillation:
                    # Knowledge distillation loss
                    batch_size = images.size(0)
                    latent_dim = vae.latent_dim
                    # generate latent vector from N(0, 1)
                    z = torch.randn(batch_size, latent_dim).to(device)  # Sample latent vectors
                    pseudo_logits = global_vae.decode(z)  # Reconstruct probabilities from latent space
                    pseudo_logits = pseudo_logits.detach().to(device)
                    if i == 0:
                        print(f"{epoch}/{epochs}", collections.Counter(labels.tolist()),
                              collections.Counter(model_logits.argmax(dim=1).tolist()),
                              collections.Counter(pseudo_logits.argmax(dim=1).tolist()))
                    distill_loss = distillation_loss(model_logits, pseudo_logits)
                    # Combine losses
                    # KL divergence: input is log_softmax, and target is softmax
                    # distill_loss = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(outputs, dim=1), pseudo_logits)
                    loss = (1 - distill_weight) * model_loss + distill_weight * distill_loss
                    _model_distill_loss += distill_weight * distill_loss.item()

                    # Diversity regularization (entropy loss)
                    diversity_loss = -torch.mean(
                        torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
                    )
                    loss += 0.1 / diversity_loss
                else:
                    # here, each client know all class information
                    new_logits, new_labels = gen_pseudo_logits(global_vae, labels,
                                                               all_class_labels={0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
                    # print(len(new_labels), collections.Counter(new_labels.tolist()))
                    # model_loss2 = criterion(new_logits, new_labels)
                    model_loss2 = weighted_cross_entropy_loss(new_logits, new_labels,
                                                              all_class_labels={0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
                    loss = model_loss + model_loss2
                    _model_distill_loss += model_loss2.item()

                optimizer.zero_grad()
                loss.backward()

                # # Print gradients for each parameter
                # print("Gradients for model parameters:")
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad}")
                #     else:
                #         print(f"{name}: No gradient (likely frozen or unused)")

                optimizer.step()
                epoch_model_loss += loss.item()
                _model_loss += model_loss.item()

            losses.append((epoch_model_loss, epoch_vae_loss))
            if epoch % 10 == 0:
                print(epoch, ' model:', epoch_model_loss / len(train_loader), _model_loss / len(train_loader),
                      _model_distill_loss / len(train_loader),
                      ' vae:', epoch_vae_loss / len(train_loader), _vae_recon_loss / len(train_loader),
                      _vae_kl_loss / len(train_loader))

        # Generate logits (knowledge) for server aggregation
        logits = []
        model.eval()
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device)
                logits.append(F.softmax(model(images), dim=1).cpu())

        result = {'vae': None, 'logits': torch.cat(logits), 'losses': losses, 'info': client_info}
        return result

    def aggregate(self, client_logits_list, new_vae):
        # Stack logits
        logits = torch.vstack(client_logits_list)  # E.g., 5 samples with 10 classes
        train_loader = DataLoader(logits, batch_size=64, shuffle=True)

        # Initialize a new VAE model
        # new_vae = VAE(input_dim=vae.input_dim, hidden_dim=vae.hidden_dim, latent_dim=vae.latent_dim)
        # Copy parameters from the old VAE model to the new VAE model
        # new_vae.load_state_dict(vae.state_dict())  # Initialize new_vae with the parameters of the old vae

        optimizer = optim.Adam(new_vae.parameters(), lr=0.001)

        new_vae.train()
        # Training loop
        server_epochs = 30
        losses = []
        for epoch in range(server_epochs):
            epoch_loss = 0
            for logits_ in train_loader:
                logits_ = logits_.to(device)
                recon_logits, mu, logvar = new_vae(logits_)
                loss, info = vae_loss_function(recon_logits, logits_, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"\tEpoch {epoch + 1}/{server_epochs}, Loss: {epoch_loss:.4f}, info: {info}")
                # for param in new_vae.parameters():
                #     if param.grad is not None:
                #         print(param.grad.abs().mean())

        # # After training, use VAE to encode the new logits
        # new_vae.eval()
        # with torch.no_grad():
        #     # Encode the new logits
        #     mu_new, logvar_new = new_vae.encode(new_logits)
        #
        #     # Compute the KL divergence between the latent distribution of the new logits and the standard normal
        #     kl_divergence = -0.5 * torch.sum(1 + logvar_new - mu_new.pow(2) - logvar_new.exp())
        #
        #     print(f"KL Divergence for new logits: {kl_divergence.item():.4f}")
        # results = {'model': new_vae, 'losses': losses}
        return new_vae

    def evaluate(self, model, test_loader, device, test_type='test', client_id=0):
        """
            Evaluate how well each client's model performs on the test set.
        """
        model = model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Client {client_id + 1} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    num_rounds = 50

    fl = FL()
    # Training
    fl.train(num_rounds, device)

    # Testing
    # Load data and test the models
    _, test_loader = load_data()
    clients = [ClientModelA(), ClientModelB()]
    for client_id, client_model in enumerate(clients):
        accuracy = fl.evaluate(client_model, test_loader, device, test_type='test all', client_id=client_id)
        # print(f"Client {client_id + 1} Test Accuracy: {accuracy * 100:.2f}%")
