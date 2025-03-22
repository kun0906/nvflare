import collections

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from auto_labeling.data.MNIST.pretrained import pretrained_CNN

from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Custom Dataset class with transform support
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform  # Add transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.data[idx]
        # target = self.targets[idx]
        #
        # # Apply transform if available
        # if self.transform:
        #     sample = self.transform(sample)
        #
        # return sample, target

        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        from PIL import Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target


# Extract features using the fine-tuned CNN for all the images (labeled + unlabeled)
def extract_features(dataset, pretrained_cnn):
    pretrained_cnn.eval()  # Set the model to evaluation mode
    # pretrained_cnn.eval() ensures that layers like batch normalization and dropout behave appropriately
    # for inference (i.e., no training-specific behavior).
    features = []
    # Create a DataLoader to load data in batches
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    for imgs, _ in dataloader:
        imgs = imgs.to(device)  # Move the batch of images to GPU
        with torch.no_grad():
            feature = pretrained_cnn(imgs)  # Forward pass through the pretrained CNN
        features.append(feature.cpu().numpy())  # Convert feature to numpy

    # Flatten the list of features
    return np.concatenate(features, axis=0)


def plot_data(X, y):
    import numpy as np
    import matplotlib.pyplot as plt
    from umap import UMAP
    from sklearn.preprocessing import LabelEncoder

    # Example data (replace with your X and y)
    np.random.seed(42)
    # X = np.random.rand(100, 16)  # 100 samples, 16 features
    # y = np.random.randint(0, 10, 100)  # 100 labels from 0 to 9

    # Encode labels if not numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(collections.Counter(y.tolist()))
    for n_neighbors in [2, 5, 10, 20, 50]:
        # UMAP projection
        umap = UMAP(n_neighbors, min_dist=0.1, n_components=2, random_state=42)
        X_embedded = umap.fit_transform(X)

        # Visualization
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_encoded, cmap='Spectral', s=50)
        plt.colorbar(scatter, label="Labels")
        plt.title(f"UMAP Visualization: n_neighbors={n_neighbors}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.show()


def decision_tree(client_data):


    # # Split the dataset into training and testing sets

    # X_train, X_test, y_train, y_test = (client_data['features'], client_data['shared_test_data']['features'],
    #                                     client_data['labels'], client_data['shared_test_data']['labels'], )
    X, y = client_data['shared_test_data']['features'], client_data['shared_test_data']['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # mask = np.array([False] * len(train_features))  # Example boolean mask
    # mask[indices] = True
    # X_train, X_test, y_train, y_test = train_features[mask], train_features[~mask], train_labels[mask], train_labels[~mask]

    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the training data
    y_pred = clf.predict(X_train)

    # Calculate accuracy
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Accuracy of the Decision Tree: {accuracy * 100:.2f}%")
    # Compute confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    print(cm)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Decision Tree: {accuracy * 100:.2f}%")
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()

def gen_local_data_mnist(client_data_file, client_id=0, label_rate=0.1):
    """ We assume num_client = num_classes, i.e., each client only has one class data

    Args:
        client_id:
        label_rate=0.1： only 10% local data has labels
    Returns:

    """
    # if os.path.exists(client_data_file):
    #     with open(client_data_file, 'rb') as f:
    #         client_data = pickle.load(f)
    # Change torch.load call to include weights_only=True
    # if os.path.exists(client_data_file):
    #     with open(client_data_file, 'rb') as f:
    #         client_data = torch.load(f, weights_only=False)
    #         decision_tree(client_data)
    #
    #         # plot_data(client_data['features'], client_data['labels'])
    #         # shared_test_data = client_data['shared_test_data']
    #         # plot_data(shared_test_data['features'], shared_test_data['labels'])
    #     return client_data

    # dir_name = os.path.dirname(client_data_file)
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name, exist_ok=True)

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    # Extract data and targets
    data = train_dataset.data  # Tensor of shape (60000, 28, 28)
    targets = train_dataset.targets  # Tensor of shape (60000,)

    # Generate local data, and only lable_rate=10% of them has labels
    X = data[targets == client_id]
    y = targets[targets == client_id]
    labels_mask = torch.tensor([False] * len(y), dtype=torch.bool)
    cnt = X.size(0)
    print(f"Class {client_id}: {cnt} images, y: {collections.Counter(y.tolist())}")
    sampling_size = int(cnt * label_rate)
    labeled_indices = torch.randperm(len(y))[:sampling_size]
    labeled_X = X[labeled_indices]
    labeled_y = y[labeled_indices]
    labels_mask[labeled_indices] = True

    # labeled_data = CustomDataset(labeled_X, labeled_y, transform=transform)
    # labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=64, shuffle=True)
    # Use the local labeled data to fine-tune CNN
    pretrained_cnn = pretrained_CNN(transform, CustomDataset, device=device)

    # Extract features for both labeled and unlabeled data
    data_ = CustomDataset(X, y, transform=transform)
    features = extract_features(data_, pretrained_cnn)
    print(features.shape)

    # plot_data(features, y)

    # *** Each client has its own pretrained_cnn, which leads to different extracted features on the shared test data ***
    # The following test_data (shared by all clients) is used to test each client model's performance, which includes
    # all 10 classes; however, in practice it may not exist.
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    shared_data = test_dataset.data  # Tensor of shape (60000, 28, 28)
    shared_targets = test_dataset.targets  # Tensor of shape (60000,)
    shared_data_ = CustomDataset(shared_data, shared_targets, transform=transform)
    shared_test_features = extract_features(shared_data_, pretrained_cnn)
    print(shared_test_features.shape)
    client_data = {'features': torch.tensor(features, dtype=torch.float), 'labels': y,
                   'labels_mask': labels_mask,  # only 10% data has labels.
                   'shared_test_data': {'features': torch.tensor(shared_test_features, dtype=torch.float),
                                        'labels': shared_targets}
                   }

    # plot_data(shared_test_features, shared_targets)
    # with open(client_data_file, 'wb') as f:
    #     # pickle.dump(client_data, f)
    torch.save(client_data, client_data_file)

    decision_tree(client_data)

    return client_data


client_data_file = 'tmp.pkl'
gen_local_data_mnist(client_data_file, client_id=0, label_rate=0.1)
