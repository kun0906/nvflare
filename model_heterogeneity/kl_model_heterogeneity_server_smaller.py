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


class GlobalModel(nn.Module):
    """Much smaller global model that can be shared across clients easily."""

    def __init__(self):
        super(GlobalModel, self).__init__()
        # Input is 28x28 image with 1 channel (grayscale)
        self.conv1 = nn.Conv2d(1, 5, 3)  # 5 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(5, 8, 3)  # 8 filters, 3x3 kernel

        # Calculate the output size after conv2 and pooling
        # Input size: 28x28 -> Conv1 -> 26x26 -> Pool -> 13x13
        # -> Conv2 -> 11x11 -> Pool -> 5x5
        flattened_size = 8 * 5 * 5  # 8 channels, 5x5 feature map

        self.fc1 = nn.Linear(flattened_size, 16)  # Fully connected layer
        self.fc2 = nn.Linear(16, 10)  # Output 10 classes for MNIST digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)  # Output layer
        return x


def distillation_loss(client_logits, vae_logits, temperature=20.0):
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
        clients = [(ClientModelA(), GlobalModel(),
                    client1_train_loader, client1_test_loader),
                   (ClientModelB(), GlobalModel(),
                    client2_train_loader, client2_test_loader),
                   ]
        # Instantiate model and optimizer
        global_model = GlobalModel()
        global_model = global_model.to(device)
        losses = []
        for round_num in range(num_rounds):
            print(f"Round {round_num + 1}")

            # Step 1: Train clients
            client_gm_parameters = []
            for client_id, (client_model, client_gm, train_loader_, test_loader_) in enumerate(clients):
                print(f"  Training client {client_id + 1}")
                client_info = {'client_id': client_id + 1}
                pretrained_teacher = round_num != 0
                if not pretrained_teacher:  # if we don't have pretrained teacher model for each client, we should train it.
                    print(f'pretrain_teacher for client {client_id + 1}')
                    client_model = self._train_teacher(client_model, train_loader_, device, client_info)
                client_result = self._train_client(client_model, client_gm, global_model, train_loader_, device,
                                                   client_info)
                client_gm_parameters.append(client_result['client_gm'])

                self.evaluate(client_gm, train_loader_, device, test_type='train', client_id=client_id)
                self.evaluate(client_gm, test_loader_, device, test_type='test', client_id=client_id)
                self.evaluate(client_gm, test_loader, device, test_type='test all', client_id=client_id)

            # Step 2: Server aggregates vae parameters
            global_model = self.aggregate(client_gm_parameters, global_model)

        print("Federated learning completed.")

    def _train_teacher(self, client_model, train_loader, device, client_info):
        """
           Each client trains its local model and sends vae parameters to the server.
        """
        labels = []
        [labels.extend(labels_.tolist()) for images_, labels_ in train_loader]
        print(client_info, collections.Counter(labels))

        # CNN
        # client_model is the teacher model: we assume we already have it, which is the pretrained model.
        optimizer = optim.Adam(client_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()  # mean

        client_model.train()  # smaller model that can be shared to server
        losses = []
        for epoch in range(epochs):
            epoch_model_loss = 0
            _model_loss, _model_distill_loss = 0, 0
            epoch_vae_loss = 0
            _vae_recon_loss, _vae_kl_loss = 0, 0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # your local personal model
                outputs = client_model(images)
                model_logits = F.softmax(outputs, dim=1)
                loss = criterion(model_logits, labels)  # cross entropy loss

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
                _model_loss += loss.item()

            losses.append((epoch_model_loss, epoch_vae_loss))
            if epoch % 10 == 0:
                print(epoch, ' model:', epoch_model_loss / len(train_loader), _model_loss / len(train_loader),
                      _model_distill_loss / len(train_loader),
                      ' vae:', epoch_vae_loss / len(train_loader), _vae_recon_loss / len(train_loader),
                      _vae_kl_loss / len(train_loader))

        result = {'client_model': client_model, 'logits': None, 'losses': losses, 'info': client_info}
        return client_model

    def _train_client(self, client_model, client_gm, global_model, train_loader, device, client_info):
        """
           Each client trains its local model and sends vae parameters to the server.
        """
        labels = []
        [labels.extend(labels_.tolist()) for images_, labels_ in train_loader]
        print(client_info, collections.Counter(labels))

        # CNN
        # client_model is the teacher model: we assume we already have it, which is the pretrained model.
        client_gm.load_state_dict(global_model.state_dict())  # Initialize client_gm with the parameters of global_model
        optimizer = optim.Adam(client_gm.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()  # mean

        client_model.eval()  # teacher
        client_gm.train()  # smaller model that can be shared to server
        losses = []
        # only train smaller model
        for epoch in range(epochs):
            epoch_model_loss = 0
            _model_loss, _model_distill_loss = 0, 0
            epoch_vae_loss = 0
            _vae_recon_loss, _vae_kl_loss = 0, 0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # your local personal model
                outputs = client_gm(images)
                model_logits = F.softmax(outputs, dim=1)
                model_loss = criterion(model_logits, labels)  # cross entropy loss

                distillation = True
                if distillation:
                    teacher_outputs = client_model(images).detach()
                    pseudo_logits = F.softmax(teacher_outputs, dim=1).to(device)

                    if i == 0:
                        print(f"{epoch}/{epochs}", collections.Counter(labels.tolist()),
                              collections.Counter(model_logits.argmax(dim=1).tolist()),
                              collections.Counter(pseudo_logits.argmax(dim=1).tolist()))
                    distill_loss = distillation_loss(outputs, teacher_outputs)
                    if distill_loss < 0:
                        print(distill_loss, model_logits, pseudo_logits)
                    # Combine losses
                    # KL divergence: input is log_softmax, and target is softmax
                    # distill_loss = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(outputs, dim=1), pseudo_logits)
                    loss = (1 - distill_weight) * model_loss + distill_weight * distill_loss
                    _model_distill_loss += distill_weight * distill_loss.item()

                    # # Diversity regularization (entropy loss)
                    # diversity_loss = -torch.mean(
                    #     torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
                    # )
                    # loss += 0.1 / diversity_loss
                else:
                    _model_loss2 = 0
                    # loss = model_loss + model_loss2
                    # _model_distill_loss += model_loss2.item()

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

        result = {'client_gm': client_gm.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
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
