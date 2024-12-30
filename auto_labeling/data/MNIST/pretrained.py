import collections

import torch
import torch.nn as nn
# from utils import timer
from torchvision import datasets, models, transforms
import torch.optim as optim


# @timer
def fine_tune_cnn(dataloader, cnn, optimizer, criterion, epochs=5, device='cpu'):
    print(f'fine_tune on dataloader with size: {len(dataloader.sampler)}')
    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if images.shape[0] < 2: continue
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


# @timer
def pretrained_CNN(transform, CustomDataset, device='cpu'):
    # Load a pre-trained CNN (e.g., ResNet18)
    # cnn = models.resnet18(pretrained=True)
    # Using 'weights' parameter instead of 'pretrained'
    cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # or models.ResNet18_Weights.DEFAULT for the most up-to-date weights
    cnn.fc = nn.Linear(cnn.fc.in_features, 16)  # Change the output layer to extract 64-dimensional features
    cnn.to(device)

    # Fine-tune the CNN with the labeled data
    cnn.train()
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    cnn_criterion = nn.CrossEntropyLoss()

    # only use one class data to fune-tune resnet18, cannot get seperated embeddings on 10 classes data.
    # fine_tune_cnn(labeled_loader, cnn, cnn_optimizer, cnn_criterion, epochs=10, device=device)

    # *** use a subset of train set to fine-tune resnet18 ***
    # Must use 10 classes to get seperated embeddings on shared_test_data.
    # transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    # transform = labeled_loader.dataset.transform
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    # Extract data and targets
    X = train_dataset.data  # Tensor of shape (60000, 28, 28)
    y = train_dataset.targets  # Tensor of shape (60000,)
    # Generate local data, and only lable_rate=10% of them has labels
    cnt = X.size(0)
    sampling_size = int(cnt * 0.01)
    labeled_indices = torch.randperm(len(y))[:sampling_size]
    labeled_X = X[labeled_indices]
    labeled_y = y[labeled_indices]
    print(f"For shared test data, fine tune on {cnt} images, y: {collections.Counter(labeled_y.tolist())}")
    labeled_data = CustomDataset(labeled_X, labeled_y, transform=transform)
    labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)

    fine_tune_cnn(labeled_loader, cnn, cnn_optimizer, cnn_criterion, epochs=10, device=device)

    return cnn
