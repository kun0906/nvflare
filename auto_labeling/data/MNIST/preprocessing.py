""" MNIST

"""

import os
import pickle
import time

import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms

from auto_labeling.data.MNIST.pretrained import pretrained_CNN


# Timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

@timer
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


def gen_shared_data(transform):
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    X = test_dataset.data.numpy()
    Y = test_dataset.targets.tolist()
    shared_test_data = {"X": X, 'y': Y}
    return shared_test_data


@timer
def preprocessing():
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    pretrained_cnn_file = f'{in_dir}/pretrained_cnn.pth'
    pretrained_cnn = pretrained_CNN(transform, CustomDataset, device=device)
    torch.save(pretrained_cnn, pretrained_cnn_file)
    # pretrained_cnn = torch.load(pretrained_cnn_file)

    pretrained_cnn.eval()

    shared_test_data = gen_shared_data(transform)

    # Extract data and targets
    X = train_dataset.data  # Tensor of shape (60000, 28, 28)
    Y = train_dataset.targets  # Tensor of shape (60000,)

    X = X.numpy()
    Y = Y.tolist()

    data = {}
    for i, label in enumerate(Y):
        x = X[i]
        if label not in data:
            data[label] = [x]
        else:
            data[label].append(x)

    os.makedirs(in_dir, exist_ok=True)

    labels = sorted(list(set(Y)))
    for c, l in enumerate(labels):
        vs = np.array(data[l])
        replace = False if len(vs) >= 1000 else True
        print(l, len(vs), replace)
        indices = list(range(len(vs)))
        indices = np.random.choice(indices, size=1000, replace=replace)
        X_ = vs[indices]
        y_ = [l] * len(X_)
        client_data = {'X': X_, 'y': y_, 'shared_data': shared_test_data}
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)

    for c in range(num_clients):
        print(f'\nclient {c}...')
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        client_data_file = f'{in_dir}/{c}.pkl'
        client_images, ys = client_data['X'], client_data['y']
        shared_test_data = client_data['shared_data']
        # X, Y, Y_names,
        data_ = CustomDataset(torch.tensor(client_images), torch.tensor(ys), transform=transform)
        X_ = extract_features(data_, pretrained_cnn)
        Y_ = [c] * len(client_images)
        Y_names_ = ys
        client_data = {'X': X_, 'y': Y_, 'y_names': Y_names_}

        # for shared data: X, Y, Y_names
        client_images, ys = shared_test_data['X'], shared_test_data['y']
        data_ = CustomDataset(torch.tensor(client_images), torch.tensor(ys), transform=transform)
        X_ = extract_features(data_, pretrained_cnn)
        Y_ = ys
        Y_names_ = ys
        client_data['shared_data'] = {'X': X_, 'y': Y_, 'y_names': Y_names_}

        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)


def check_client_data():
    for c in range(num_clients):
        client_data_file = f'{in_dir}/{c}.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        print(c, client_data)


if __name__ == '__main__':
    in_dir = 'data'
    num_clients = 10
    preprocessing()
    check_client_data()
