import torch.nn as nn
from utils import timer
from torchvision import datasets, models, transforms
import torch.optim as optim

@timer
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

@timer
def pretrained_CNN(labeled_loader, device='cpu'):
    # Load a pre-trained CNN (e.g., ResNet18)
    # cnn = models.resnet18(pretrained=True)
    # Using 'weights' parameter instead of 'pretrained'
    cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # or models.ResNet18_Weights.DEFAULT for the most up-to-date weights
    cnn.fc = nn.Linear(cnn.fc.in_features, 32)  # Change the output layer to extract 64-dimensional features
    cnn.to(device)

    # Fine-tune the CNN with the labeled data
    cnn.train()
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    cnn_criterion = nn.CrossEntropyLoss()

    fine_tune_cnn(labeled_loader, cnn, cnn_optimizer, cnn_criterion, epochs=10, device=device)

    return cnn
