""" Centralized Benchmark Models On Semi-Supervised datasets

    1. Semi-Supervised Datasets
        PubMed
            10% labeled data + 90% unlabeled data

    2. Centralized Models
        1) Decision Tree
            During the training phase, decision tree only can use 10% labeled data.

        2) GNN
            During the training phase, GNN can use 10% labeled data + 90% unlabeled data.


"""
import collections
import os

print(os.path.abspath(os.getcwd()))

from torch_geometric.datasets import Planetoid


def gen_data():
    # Load the PubMed dataset
    dataset = Planetoid(root='./data', name='PubMed', split='full')
    # 10%  labeled data + 90% labeled data
    # Access the data (the first graph in the dataset)
    data = dataset[0]

    # Inspect the dataset
    print(f"Number of Nodes: {data.num_nodes}")
    print(f"Number of Edges: {data.num_edges}")
    print(f"Node Feature Size: {data.x.shape[1]}")
    print(f"Number of Classes: {dataset.num_classes}")
    gnn_data = data

    # Get traditional dataset for non-graphical dataset
    X = data.x.numpy()
    Y = data.y.numpy()
    train_mask, val_mask, test_mask = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()
    X_train, y_train = X[train_mask], Y[train_mask]
    X_val, y_val = X[val_mask], Y[val_mask]
    X_test, y_test = X[test_mask], Y[test_mask]
    print('\nX_train shape:', X_train.shape, 'Y_train:', collections.Counter(y_train),
          '\nX_val shape:', X_val.shape, 'Y_val:', collections.Counter(y_train),
          '\nX_test shape:', X_test.shape, 'Y_test:', collections.Counter(y_test))
    trad_data = {"train": (X_train, y_train),
                 "val": (X_val, y_val),
                 "test": (X_test, y_test)}

    return gnn_data, trad_data


import torch.nn as nn


# Define the MLP Model
class MLPBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPBase, self).__init__()
        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size)  # Input to first hidden layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)  # First hidden to second hidden
        self.layer3 = nn.Linear(hidden_size, output_size)  # Second hidden to output
        self.activation = nn.ReLU()  # Non-linear activation function

    def forward(self, x):
        # Forward pass
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)  # Output layer usually has no activation for regression
        return x


def early_stopping(model, X_val, y_val, epoch, pre_val_loss, val_cnt, criterion, patience=10):
    # Validation phase
    model.eval()
    val_loss = 0.0
    stop_training = False
    with torch.no_grad():
        if y_val is not None:  # is not graph data
            outputs_ = model(X_val)
            loss_ = criterion(outputs_, y_val)
        else:
            data = X_val  # here must be graph data
            outputs_ = model(data)
            # Loss calculation: Only for labeled nodes
            loss_ = criterion(outputs_[data.val_mask], data.y[data.val_mask])
        val_loss += loss_

    if epoch == 0:
        pre_val_loss = val_loss
        return val_loss, pre_val_loss, val_cnt, stop_training

    if val_loss <= pre_val_loss:
        pre_val_loss = val_loss
        val_cnt = 0
    else:  # if val_loss > pre_val_loss, it means we should consider early stopping.
        val_cnt += 1
        if val_cnt >= patience:
            # training stops.
            stop_training = True
    return val_loss, pre_val_loss, val_cnt, stop_training


class MLP():

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the model
        self.model = MLPBase(input_size, hidden_size, output_size)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, X, y, X_val=None, y_val=None):
        epochs = 10000
        losses = []
        X = torch.tensor(X, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        X_val = torch.tensor(X_val, device=device)
        y_val = torch.tensor(y_val, dtype=torch.long, device=device)
        pre_val_loss = 0
        val_cnt = 0
        val_losses = []
        for epoch in range(epochs):
            self.model.train()

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(X)

            # Loss calculation: Only for labeled nodes
            loss = self.criterion(output, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            # if epoch % 50 == 0:
            #     print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Train Size: {len(X)}")
            val_loss, pre_val_loss, val_cnt, stop_training = early_stopping(self.model, X_val, y_val, epoch,
                                                                            pre_val_loss,
                                                                            val_cnt, self.criterion, patience=10)
            val_losses.append(val_loss.item())
            if stop_training:
                self.model.stop_training = True
                print(f'Early Stopping. Epoch: {epoch}, Loss: {loss:.4f}')
                break

        show = True
        if show:
            import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 6))
            # plt.plot(range(epochs), losses, label='Training Loss', marker='o')
            plt.plot(range(epoch + 1), losses, label='Training Loss', marker='o')  # when using early_stopping
            plt.plot(range(epoch + 1), val_losses, label='Validating Loss', marker='+')  # when using early_stopping
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            # plt.grid()
            plt.show()

    def predict(self, X):
        print('Testing MLP model...')
        self.model.eval()
        X = torch.tensor(X, device=device)
        with torch.no_grad():
            output = self.model(X)
            _, predicted_labels = torch.max(output, dim=1)

        return predicted_labels.numpy()


def ClassicalML(data):
    # # feature_file = 'feature.pkl'
    # # # semi_ml_pretrain.gen_features(feature_file)
    # # with open(feature_file, 'rb') as f:
    # #     train_info = pickle.load(f)
    # train_features = train_info['train_features']
    # indices = np.array(train_info['indices'])
    # train_labels = train_info['train_labels']
    # # Assuming `train_features` and `train_labels` are your features and labels respectively
    #
    # # # Split the dataset into training and testing sets
    # # X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_features[indices], train_features, train_labels[indices], train_labels
    # # mask = np.array([False] * len(train_features))  # Example boolean mask
    # # mask[indices] = True
    # # X_train, X_test, y_train, y_test = train_features[mask], train_features[~mask], train_labels[mask], train_labels[~mask]

    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    dim = X_train.shape[1]
    num_classes = len(set(y_train))
    print(f'Number of Features: {dim}, Number of Classes: {num_classes}')

    from sklearn.tree import DecisionTreeClassifier
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    from sklearn.ensemble import GradientBoostingClassifier
    gd = GradientBoostingClassifier(random_state=42)

    from sklearn import svm
    svm = svm.SVC(random_state=42)

    mlp = MLP(dim, 64, num_classes)
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, 'MLP': mlp}
    clfs = {'Decision Tree': dt, 'Random Forest': rf, 'MLP': mlp}

    for clf_name, clf in clfs.items():
        print(f"\nTraining {clf_name}")
        # Train the classifier on the training data
        if clf_name == 'MLP':
            clf.fit(X_train, y_train, X_val, y_val)
        else:
            clf.fit(X_train, y_train)

        print(f"\nTesting {clf_name}")
        for X_, y_ in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
            # Make predictions on the data
            y_pred_ = clf.predict(X_)
            # Calculate accuracy
            accuracy = accuracy_score(y_, y_pred_)
            print(f"Accuracy of {clf_name}: {accuracy * 100:.2f}%")
            # Compute confusion matrix
            cm = confusion_matrix(y_, y_pred_)
            print(cm)

    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch_geometric.nn import GCNConv

from utils import timer

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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


class GNNModel4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel4, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Adding a third layer
        self.conv4 = GCNConv(hidden_dim, output_dim)  # Output layer

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        no_edge_attr = True
        if no_edge_attr:
            # no edge_attr is passed to the GCNConv layers
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))  # Additional layer
            x = self.conv4(x, edge_index)  # Final output
        else:
            # Passing edge_attr to the GCNConv layers
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.relu(self.conv3(x, edge_index, edge_attr))  # Additional layer
            x = self.conv4(x, edge_index, edge_attr)  # Final output

        # return F.log_softmax(x, dim=1)
        return x  # for CrossEntropyLoss


from torch_geometric.nn import GraphSAGE


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphSAGEModel, self).__init__()
        # Assuming the GraphSAGE model is defined here with num_layers
        self.conv1 = GraphSAGE(input_dim, hidden_dim, num_layers)  # Pass num_layers to GraphSAGE
        self.conv2 = GraphSAGE(hidden_dim, output_dim, num_layers)  # If you have more layers
        self.fc = torch.nn.Linear(output_dim, output_dim)  # Assuming 10 output classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        # return F.log_softmax(x, dim=1)
        return x  # for CrossEntropyLoss


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4):
        super(GATModel, self).__init__()

        # First GAT layer
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        # Second GAT layer
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.6)

        # Fully connected layer
        self.fc = torch.nn.Linear(output_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GAT layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))

        # Second GAT layer
        x = self.conv2(x, edge_index)

        # Apply log softmax to get class probabilities
        x = self.fc(x)
        # return F.log_softmax(x, dim=1)
        return x  # for CrossEntropyLoss


@timer
# Training loop for GNN
def train_gnn(model, criterion, optimizer, data, epochs=10, show=True):
    model.train()

    losses = []
    pre_val_loss = 0
    val_cnt = 0
    val_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Loss calculation: Only for labeled nodes
        loss = criterion(output[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, train_mask: {sum(data.train_mask)}")

        # X_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
        val_loss, pre_val_loss, val_cnt, stop_training = early_stopping(model, data, None, epoch, pre_val_loss,
                                                                        val_cnt, criterion, patience=10)
        val_losses.append(val_loss.item())
        if stop_training:
            model.stop_training = True
            print(f'Early Stopping. Epoch: {epoch}, Loss: {loss:.4f}')
            break

    if show:
        import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        plt.plot(range(epoch + 1), losses, label='Training Loss', marker='o')
        plt.plot(range(epoch + 1), val_losses, label='Validating Loss', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        # plt.grid()
        plt.show()


@timer
def GraphNN(data):
    data.to(device)

    _, input_dim = data.x.shape
    classes = data.y.numpy().tolist()
    num_classes = len(set(classes))
    print(f'input_dim: {input_dim}, y: {collections.Counter(classes)}')
    # Initialize model and move to GPU
    gnn_base = GNNModel(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
    gnn_base2 = GNNModel4(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
    gsage = GraphSAGEModel(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
    gat = GATModel(input_dim=input_dim, hidden_dim=32, output_dim=num_classes, heads=2)
    gnns = {"GNN": gnn_base, 'GNN2': gnn_base2, 'GraphSAGE': gsage, 'GAT': gat}

    for gnn_name, gnn in gnns.items():
        print(f'\n Training {gnn_name}')
        gnn.to(device)
        # Loss and optimizer for GNN
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(gnn.parameters(), lr=0.001)

        print(f'\nTraining {gnn_name} model...')
        train_gnn(gnn, criterion, optimizer, data, epochs=5000)

        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'\nTesting {gnn_name} model...')
        gnn.eval()
        with torch.no_grad():
            output = gnn(data)
            _, predicted_labels = torch.max(output, dim=1)

            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                print(f'\nlen(mask) {len(mask)}')
                labels_ = data.y[mask]

                y_ = labels_.cpu().numpy()
                y_pred_ = predicted_labels[mask].cpu().numpy()
                accuracy = accuracy_score(y_, y_pred_)
                print(f"Accuracy: {accuracy * 100:.2f}%")

                # Compute the confusion matrix
                conf_matrix = confusion_matrix(y_, y_pred_)
                print("Confusion Matrix:")
                print(conf_matrix)


if __name__ == '__main__':
    graph_data, trad_data = gen_data()

    ClassicalML(trad_data)

    GraphNN(graph_data)
