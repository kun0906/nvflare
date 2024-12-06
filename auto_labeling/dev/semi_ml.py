"""
To implement automated labeling using a Graph Neural Network (GNN) for medical X-ray images,
where you want to propagate labels across images based on their visual similarities,
you can use a framework like PyTorch Geometric (PyG) or DGL (Deep Graph Library).
Below is an example code snippet using PyTorch Geometric to demonstrate the steps mentioned in the previous example.

Example Code for GNN-based Automated Labeling of X-ray Images
Representing the Data as a Graph:
1. Each X-ray image is represented as a node, and edges between nodes represent the similarity between images based on their features.

2. Initial Labels: Some nodes are labeled (e.g., images labeled as "pneumonia", "lung cancer", etc.), while others are unlabeled.

3. Training the GNN: The GNN learns to propagate labels across the graph.

4. Automated Labeling: Once trained, the GNN can label unlabeled images based on their relationships to labeled nodes.


pip install torch torchvision torch-geometric



"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


# Define CNN model to extract features from MNIST images
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Use CNN to extract features
cnn = CNNFeatureExtractor()
cnn.train()  # Make sure CNN is in training mode

# Define a loss function for CNN training
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.001)


def train_cnn(dataset, cnn, criterion, optimizer, epochs=5):
    cnn.train()
    for epoch in range(epochs):
        for images, labels in dataset:
            images = images.unsqueeze(0)  # Add batch dimension
            labels = labels.unsqueeze(0)
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


train_cnn(train_data, cnn, cnn_criterion, cnn_optimizer, epochs=5)


# Extract CNN features from the MNIST dataset
def extract_features(dataset):
    features = []
    for img, _ in dataset:
        img = img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            feature = cnn(img)
        features.append(feature.numpy())
    return np.array(features)


train_features = extract_features(train_data)

# Calculate cosine similarity to build graph edges
similarity_matrix = cosine_similarity(train_features)

# Create graph: Each image is a node, edges based on similarity
threshold = 0.8  # Similarity threshold for creating edges
edges = []
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            edges.append([i, j])

edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Generate node features (features from CNN)
node_features = torch.tensor(train_features, dtype=torch.float)

# Create train mask (10% labeled, 90% unlabeled)
train_mask = torch.zeros(len(train_data), dtype=torch.bool)
train_mask[:int(0.1 * len(train_data))] = 1

# Create labels (10% labeled, others are -1 or placeholder)
labels = torch.full((len(train_data),), -1, dtype=torch.long)  # Initialize labels
labels[:int(0.1 * len(train_data))] = torch.randint(0, 10, (int(0.1 * len(train_data)),))  # Random labels for the 10%

# Prepare data for PyG (PyTorch Geometric)
data = Data(x=node_features, edge_index=edges, y=labels, train_mask=train_mask)


# Define the Graph Neural Network model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x


# Initialize model
model = GNNModel(input_dim=64, hidden_dim=32, output_dim=10)  # Assuming 64 features after CNN

# Loss and optimizer for GNN
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training loop for GNN
def train_gnn(model, data, epochs=10):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Loss calculation: Only for labeled nodes
        loss = criterion(output[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


train_gnn(model, data, epochs=10)

# After training, the model can make predictions for both labeled and unlabeled nodes
model.eval()
with torch.no_grad():
    output = model(data)
    _, predicted_labels = torch.max(output, dim=1)

    # Calculate accuracy for the labeled data
    labeled_indices = data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
    true_labels = data.y[labeled_indices]
    accuracy = accuracy_score(true_labels.numpy(), predicted_labels[labeled_indices].numpy())

    print(f"Accuracy on labeled data: {accuracy * 100:.2f}%")
