"""
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda && conda activate nvflare-3.10 && cd nvflare/auto_labeling

    $PYTHONPATH=.:nsf python3 nsf/ml_models.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

# csv_file = "data/Sentiment140/training.1600000.processed.noemoticon.csv_bert.csv"
# csv_file = "data/SHAKESPEARE/shakespeare_plays.csv_bert.csv"
# csv_file = "data/Mental/Combined Data.csv_bert.csv"
# csv_file = "data/arXiv/arXiv_scientific dataset.csv_bert.csv"
# csv_file = "data/COVID/Corona_NLP_train.csv_bert.csv"
# csv_file = "data/NewsCategory/News_Category_Dataset_v3.json_bert.csv"
csv_file = "data/FakeNews/Fake.csv_bert.csv"
reduce_dimension = False



def main2(csv_file):
    if reduce_dimension:
        df = pd.read_csv(csv_file, dtype=float, header=None)
        X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values.astype(int)
        print(X.shape, y.shape)

        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X)

        # Create a new DataFrame with reduced features + labels
        df_reduced = pd.DataFrame(X_reduced)
        df_reduced[len(df_reduced.columns)] = y  # Append labels as last column

        # Save the reduced dataset
        reduced_csv_file = f"{csv_file}_reduced.csv"
        df_reduced.to_csv(reduced_csv_file, index=False, header=False)

        print(f"Saved reduced dataset to {reduced_csv_file}")
        csv_file = reduced_csv_file

    # Load dataset
    # csv_file = 'data/SHAKESPEARE/shakespeare_plays.csv_bert.csv'  # Use the all dataset
    # # csv_file = 'data/SHAKESPEARE/shakespeare_plays.csv_bert.csv_reduced.csv'  # Use the reduced dataset
    df = pd.read_csv(csv_file, header=None)
    # df = df[df.iloc[:, -1].isin([0, 1, 2])]
    # df = df[df.iloc[:, -1].isin([2, 3, 4])]

    # Split features and labels
    X = df.iloc[:, :-1].values  # Features (all columns except last)
    y = df.iloc[:, -1].values.astype(int)  # Labels (last column)
    print(X.shape, y.shape)
    num_classes = len(np.unique(y))
    print(f'Number of classes: {num_classes}, where {np.unique(y, return_counts=True)}', flush=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Accuracy: {accuracy:.4f}')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=10, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),  # Additional layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def load_and_preprocess_data(csv_file, reduce_dimension=True, n_components=100):
    df = pd.read_csv(csv_file, dtype=float, header=None)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(int)

    if reduce_dimension:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        print(f"PCA applied: Reduced to {n_components} dimensions")

    return X, y


def train_mlp(X_train, y_train, X_test, y_test, input_dim, num_classes, epochs=100, batch_size=64, lr=0.0001):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = MLPClassifier(input_dim=input_dim, output_dim=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            scheduler.step(loss)  # Adjust learning rate based on loss

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f"MLP Test Accuracy: {accuracy:.4f}")


def main(csv_file):

    # Load and preprocess data
    X, y = load_and_preprocess_data(csv_file, reduce_dimension=reduce_dimension, n_components=100)
    num_classes = len(np.unique(y))
    print(f'Number of classes: {num_classes}, where {np.unique(y, return_counts=True)}', flush=True)
    input_dim = X.shape[1]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate MLP
    train_mlp(X_train, y_train, X_test, y_test, input_dim=input_dim, num_classes=num_classes)


if __name__ == '__main__':
    # main(csv_file)    # MLP

    main2(csv_file)     # RF
