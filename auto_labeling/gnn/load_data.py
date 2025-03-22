import os
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid
from torchvision import datasets, transforms

from data.MNIST.pretrained import pretrained_CNN
from ragg.utils import timer

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


@timer
# Extract features using the fine-tuned CNN for all the images (labeled + unlabeled)
def extract_features(dataset, pretrained_cnn):
    pretrained_cnn.eval()  # Set the model to evaluation mode
    # pretrained_cnn.eval() ensures that layers like batch normalization and dropout behave appropriately
    # for inference (i.e., no training-specific behavior).
    features = []
    # Create a DataLoader to load data in batches
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)

    for i, (imgs, _) in enumerate(dataloader):
        print(f'batch {i}')
        imgs = imgs.to(device)  # Move the batch of images to GPU
        with torch.no_grad():
            feature = pretrained_cnn(imgs)  # Forward pass through the pretrained CNN
        features.append(feature.detach().cpu().numpy())  # Convert feature to numpy

    # Flatten the list of features
    return np.concatenate(features, axis=0)


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


def load_data(data='cora'):
    data_file = 'data.pt'
    # if os.path.exists(data_file):
    #     return torch.load(data_file, weights_only=False)

    if data == 'cora':
        # Load the Cora dataset
        dataset = Planetoid(root='./data', name='Cora', split='full')

        # Access the first graph in the dataset
        data = dataset[0]

        # Dataset summary
        print(f"Dataset Summary:\n"
              f"- Number of Nodes: {data.num_nodes}\n"
              f"- Number of Edges: {data.num_edges}\n"
              f"- Node Feature Size: {data.x.shape[1]}\n"
              f"- Number of Classes: {dataset.num_classes}")

        # Extract data into NumPy arrays
        X = data.x.numpy()
        Y = data.y.numpy()
        edge_index = data.edge_index
        # # edge_indices = set([(row[0], row[1]) for row in edge_indices])
        # edge_indices = set(map(tuple, data.edge_index.numpy().T))
        # unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in data.edge_index.numpy().T])
        # print(f'unique edges: {len(unqiue_edges)} =? edge_indices/2: {len(edge_indices) / 2}, '
        #       f'edges: {data.edge_index.shape}')

        # X_train = X[data.train_mask]
        # y_train = Y[data.train_mask]
        # X_val = X[data.val_mask]
        # y_val = Y[data.val_mask]
        # X_test = X[data.test_mask]
        # y_test = Y[data.test_mask]
    elif data == 'mnist':
        data_file2 = 'mnist.data'
        if os.path.exists(data_file2):
            with open(data_file2, 'rb') as f:
                X, Y = torch.load(f)
        else:
            # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
            train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
            in_dir = '.'
            pretrained_cnn_file = f'{in_dir}/pretrained_cnn.pth'
            pretrained_cnn = pretrained_CNN(transform, CustomDataset, device=device)
            # torch.save(pretrained_cnn, pretrained_cnn_file)
            # pretrained_cnn = torch.load(pretrained_cnn_file)

            pretrained_cnn.eval()

            X = train_dataset.data  # Tensor of shape (60000, 28, 28)
            Y = train_dataset.targets  # Tensor of shape (60000,)

            data_ = CustomDataset(X, Y, transform=transform)
            X = extract_features(data_, pretrained_cnn)  # numpy array
            Y = Y.numpy()

            torch.save((X, Y), data_file2)
    else:
        raise NotImplementedError

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    torch.save((X_train, y_train, X_val, y_val, X_test, y_test), data_file)

    return X_train, y_train, X_val, y_val, X_test, y_test
