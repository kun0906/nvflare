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
    info = (recon_loss.item(), kl_loss.item())
    # Return the total loss
    beta = 0.3
    return recon_loss + beta * kl_loss, info


def distillation_loss(client_logits, vae_logits, temperature=2.0):
    client_probs = F.log_softmax(client_logits / temperature, dim=1)
    vae_probs = F.softmax(vae_logits / temperature, dim=1)
    return F.kl_div(client_probs, vae_probs, reduction='batchmean') * (temperature ** 2)


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
                print(f"  Training client {client_id}")
                client_info = {'client_id': client_id}
                # train vae+personal_model
                client_result = self._train_client(client_model, client_vae, global_vae, train_loader_, device,
                                                   client_info)
                client_vae_parameters.append(client_result['vae'])

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
        criterion = nn.CrossEntropyLoss()

        # VAE
        vae.load_state_dict(global_vae.state_dict())    # deep copy of global_vae parameters
        vae = vae.to(device)
        vae_optimizer = optim.Adam(vae.parameters(), lr=0.001)
        # vae_criterion = nn.KLDivLoss(reduction="batchmean")

        model.train()
        vae.eval()
        losses = []
        epochs = 11
        distill_weight = 2
        print('Training local model')
        for epoch in range(epochs):
            epoch_model_loss = 0
            epoch_vae_loss = 0
            for j, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # your local personal model
                outputs = model(images)
                model_logits = F.softmax(outputs, dim=1)
                model_loss = criterion(model_logits, labels)

                # Knowledge distillation loss
                if True:
                    batch_size = images.size(0)
                    latent_dim = vae.latent_dim
                    # generate latent vector from N(0, 1)
                    z = torch.randn(batch_size, latent_dim).to(device)  # Sample latent vectors
                    pseudo_logits = global_vae.decode(z)  # Reconstruct probabilities from latent space
                    if client_info['client_id'] == 0:
                        pseudo_logits[:, 5:] = -float('inf')    # only keep 0-4 classes
                    else:
                        # Set the first 5 classes to a large negative number (ignore them)
                        pseudo_logits[:, :5] = -float('inf')    # only keep 5-9 classes
                else:
                    pseudo_logits, mu, logvar = vae(outputs)
                pseudo_logits = pseudo_logits.detach().to(device)
                if epoch %10 == 0 and j == 0:
                    print(epoch, collections.Counter(labels.tolist()),
                          collections.Counter(model_logits.argmax(dim=1).tolist()),
                          collections.Counter(pseudo_logits.argmax(dim=1).tolist()), )
                distill_loss = distillation_loss(model_logits, pseudo_logits)
                # Combine losses
                loss = model_loss + distill_weight * distill_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_model_loss += loss.item()

            losses.append((epoch_model_loss, epoch_vae_loss))
            if epoch % 10 == 0:
                print(epoch, model_loss.item(), distill_weight * distill_loss.item(), loss.item())

        print('Training local VAE')
        model.eval()
        vae.train()
        for epoch in range(epochs):
            epoch_model_loss = 0
            epoch_vae_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # your local personal model
                outputs = model(images)
                model_logits = F.softmax(outputs, dim=1)
                # your local vae model
                # generated logits
                logits = model_logits.detach().to(device)

                reconstructed_logits, mu, logvar = vae(logits)  # reconstructed logits
                vae_loss, vae_info = vae_loss_function(reconstructed_logits, logits, mu, logvar)

                vae_optimizer.zero_grad()
                vae_loss.backward()
                vae_optimizer.step()
                epoch_vae_loss += vae_loss.item()

            losses.append((epoch_model_loss, epoch_vae_loss))
            if epoch % 10 == 0:
                print(epoch, 'vae:', vae_info)

        result = {'vae': vae.state_dict(), 'losses': losses, 'info': client_info}
        return result

    def aggregate(self, client_parameters_list, global_model):
        # Initialize the aggregated state_dict for the global model
        global_state_dict = {key: torch.zeros_like(value) for key, value in global_model.state_dict().items()}

        # Perform simple averaging of the parameters
        for client_state_dict in client_parameters_list:

            # Aggregate parameters for each layer
            for key in global_state_dict:
                global_state_dict[key] += client_state_dict[key]

        # Average the parameters across all clients
        num_clients = len(client_parameters_list)
        for key in global_state_dict:
            global_state_dict[key] /= num_clients

        # Update the global model with the aggregated parameters
        global_model.load_state_dict(global_state_dict)
        return global_model

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
