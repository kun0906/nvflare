"""Centralized CNN evaluated on MNIST
    50 epochs
    Train Accuracy: 99.95%
    Test Accuracy: 99.08%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CentralizedModel(nn.Module):
    def __init__(self):
        super(CentralizedModel, self).__init__()
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


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # # Create subsets of the first 100 samples
    # train_dataset = Subset(train_dataset, range(1000))
    # test_dataset = Subset(test_dataset, range(1000))
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('train_loader batches', len(train_loader.batch_sampler), len(train_loader.batch_sampler) * batch_size)
    print('test_loader batches', len(test_loader.batch_sampler), len(test_loader.batch_sampler) * batch_size)

    return train_loader, test_loader


class CentralizedCase(nn.Module):
    def __init__(self):
        self.results = {}

    def train(self, epochs, device):

        # Load data
        train_loader, test_loader = load_data()

        # Instantiate model and optimizer
        model = CentralizedModel()
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
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}')
        # # Generate logits (knowledge) for server aggregation
        # logits = []
        # model.eval()
        # with torch.no_grad():
        #     for images, _ in train_loader:
        #         images = images.to(device)
        #         logits.append(F.softmax(model(images), dim=1).cpu())

        # Step 4: Validate the model
        self.evaluate(model, train_loader, device, test_type='train')
        self.evaluate(model, test_loader, device, test_type='test')
        self.model = model

    def evaluate(self, model, test_loader, device, test_type='test'):
        """
            Evaluate how well each client's model performs on the test set.
        """
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
        print(f"{test_type} Accuracy: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    epochs = 50

    cc = CentralizedCase()
    # Training
    cc.train(epochs, device)

    # Testing
    _, test_loader = load_data()
    accuracy = cc.evaluate(cc.model, test_loader, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
