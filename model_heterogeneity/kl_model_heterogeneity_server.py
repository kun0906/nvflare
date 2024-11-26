import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

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

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # # Create subsets of the first 100 samples
    # train_dataset = Subset(train_dataset, range(1000))
    # test_dataset = Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


"""
Each client trains its local model and sends logits (knowledge) to the server.
"""
def train_client(model, train_loader, epochs, device, client_info):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        losses.append(epoch_loss)

    # Generate logits (knowledge) for server aggregation
    logits = []
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            logits.append(F.softmax(model(images), dim=1).cpu())

    result = {'logits': torch.cat(logits), 'losses': losses, 'info':client_info}
    return result # Return aggregated logits


def evaluate(model, test_loader, device, test_type='test', client_id=1):
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
    print(f"\tClient {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")
    return accuracy


"""
The server aggregates logits from clients and uses them to update the global model.
the server uses a NN to find the distribution of (all clients) logits 
"""

import torch
import torch.nn as nn
import torch.optim as optim


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
    info = (recon_loss, kl_loss)
    # Return the total loss
    beta = 0.3
    return recon_loss+beta*kl_loss, info


def aggregate_logits(client_logits_list, vae):
    # Stack logits
    logits = torch.vstack(client_logits_list)  # E.g., 5 samples with 10 classes
    train_loader = DataLoader(logits, batch_size=64, shuffle=True)

    # Initialize a new VAE model
    new_vae = VAE(input_dim=vae.input_dim, hidden_dim=vae.hidden_dim, latent_dim=vae.latent_dim)
    # Copy parameters from the old VAE model to the new VAE model
    new_vae.load_state_dict(vae.state_dict())  # Initialize new_vae with the parameters of the old vae

    optimizer = optim.Adam(new_vae.parameters(), lr=0.001)

    new_vae.train()
    # Training loop
    epochs = 30
    losses = []
    for epoch in range(epochs):
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
            print(f"\tEpoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, info: {info}")
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
    results = {'model': new_vae.state_dict(), 'losses':losses}
    return results


"""
Clients distill knowledge from the aggregated logits provided by the server.
"""


def distill_client(model, train_loader, vae, device):
    model = model.to(device)
    vae = vae.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.KLDivLoss(reduction="batchmean")

    model.train()
    for images, _ in train_loader:
        images = images.to(device)
        local_logits = F.log_softmax(model(images), dim=1)  # Logits of the client model

        # Assuming you have a trained VAE model
        vae.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Step 1: Encode the input to get the mean (mu) and log variance (logvar)
            mu, logvar = vae.encode(local_logits)  # Input shape: [batch_size, input_dim]
            # Step 2: Reparameterize to get the latent variable z
            z = vae.reparameterize(mu, logvar)
            # Step 3: Decode the latent variable z to generate the output
            global_logits = vae.decode(z)  # vae is used to approximate to the true distribution

        loss = criterion(local_logits, global_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class FL:
    def __init__(self):
        self.results={}

    def train(self, num_clients, num_rounds, device):
        """
        The main FL loop coordinates training, aggregation, and distillation across clients.
        """
        # Load data
        train_loader, test_loader = load_data()

        # Initialize client models
        clients = [ClientModelA(), ClientModelB()]
        client_logits_list = []

        # Instantiate model and optimizer
        n_classes = 10
        vae = VAE(input_dim=n_classes, hidden_dim=32, latent_dim=5)
        vae.to(device)
        losses = []
        for round_num in range(num_rounds):
            print(f"Round {round_num + 1}")

            # Step 1: Train clients
            client_logits_list = []
            for client_id, client_model in enumerate(clients):
                print(f"  Training client {client_id + 1}")
                client_result = train_client(client_model, train_loader, epochs=1, device=device,
                                              client_info=(client_id, client_model))
                evaluate(client_model, train_loader, device, test_type='train', client_id=client_id)
                evaluate(client_model, test_loader, device, test_type='test', client_id=client_id)
                client_logits = client_result['logits']
                client_logits_list.append(client_logits)

            # Step 2: Server aggregates logits
            server_result = aggregate_logits(client_logits_list, vae)
            vae.load_state_dict(server_result['model'])
            # Step 3: Clients distill global knowledge
            for client_id, client_model in enumerate(clients):
                print(f"  Distilling knowledge to client {client_id + 1}")
                distill_client(client_model, train_loader, vae, device=device)

                # Step 4: Validate the model
                self.evaluate(client_model, train_loader, device, test_type='train', client_id=client_id)
                self.evaluate(client_model, test_loader, device, test_type='test', client_id=client_id)
        print("Federated learning completed.")

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
        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    num_clients = 2
    num_rounds = 50

    fl = FL()
    # Training
    fl.train(num_clients, num_rounds, device)

    # Testing
    # Load data and test the models
    _, test_loader = load_data()
    clients = [ClientModelA(), ClientModelB()]
    for client_id, client_model in enumerate(clients):
        accuracy = fl.evaluate(client_model, test_loader, device)
        print(f"Client {client_id + 1} Test Accuracy: {accuracy * 100:.2f}%")
