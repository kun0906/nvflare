import random

import collections

from torch.utils.data import Subset
from torchvision import datasets, transforms


def gen_iid(dataset, class_indices={}, num_clients=10):
    """
       Generate stratified data for federated learning.
       - rate = 0 -> IID (stratified by class but balanced for each client)
       """

    # Initialize client datasets
    client_datasets = {i: [] for i in range(num_clients)}

    # For each class, distribute samples to clients
    for class_id, indices in class_indices.items():
        # Shuffle the indices for the current class
        random.shuffle(indices)

        # Split the data for this class into `num_clients` subsets
        num_samples_per_client = len(indices) // num_clients

        for client_id in range(num_clients):
            start_idx = client_id * num_samples_per_client
            end_idx = start_idx + num_samples_per_client
            client_datasets[client_id].extend(indices[start_idx:end_idx])

    # Create Subsets for each client
    client_subsets = []
    for client_id, indices in client_datasets.items():
        client_subset = Subset(dataset, indices)
        client_subsets.append(client_subset)

    return client_subsets


def generate_non_iid_data(dataset, num_clients=10, rate=0.5):
    """
    Generate non-IID data for federated learning, where rate controls the degree of non-IIDness.
    - rate = 0 -> IID (uniform distribution)
    - rate = 1 -> Extreme Non-IID (each client gets data from only one class)
    - rate in (0, 1) -> Partial Non-IID (clients get a skewed mix of classes)
    """
    # Group data by class labels
    class_indices = collections.defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_indices[label.item()].append(idx)

    # Generate non-IID data for each client based on the rate
    if rate == 0:
        # IID case: Randomly shuffle all indices and divide evenly among clients
        client_datasets = gen_iid(dataset, class_indices, num_clients)
    elif rate == 1:
        # Initialize client datasets
        client_datasets = []
        # Extreme Non-IID case: Each client gets data from one class only
        # In this case, to make things simple, we let num_clients = num_classes = 10
        for client_id in range(num_clients):
            # class_id = client_id % 10  # Rotate through the 10 classes
            label = client_id + 1
            client_indices = class_indices[label]
            client_dataset = Subset(dataset, client_indices)
            client_datasets.append(client_dataset)

    else:
        # Partial Non-IID case: Distribute each class partially across clients
        # each client has the majority class + other even classes.
        # E.g., if rate = 0.3, (we assume we have 10 clients and 10 classes, i.e., num_clients=num_classes)
        # then client 1 has 30% of class 1, and the rest 70% of class 1 is evenly distributed on the rest 9 clients,
        # i.e., 70% classes / 9 for each rest client
        # for client_id in range(num_clients):
        #     print(f"client_id:{client_id}")
        #     client_indices = []
        #     for class_id, indices in class_indices.items():
        #         # Calculate how much data each client gets from each class
        #         num_class_samples = int(len(indices) * rate)
        #         start_idx = (client_id * num_class_samples) % len(indices)  # Wrap around if necessary
        #         end_idx = start_idx + num_class_samples
        #         print(f"\tclass_id:{class_id}, start_idx:{start_idx}, end_idx:{end_idx}, size:{end_idx-start_idx}")
        #         client_indices.extend(indices[start_idx:end_idx])
        #     client_dataset = Subset(dataset, client_indices)
        #     client_datasets.append(client_dataset)

        # Initialize client datasets
        client_datasets = []
        clients_indices = [[] for i in range(num_clients)]  # as the class_id starts from 0
        # we require that num_clients = num_classes
        # For each class, distribute data based on the rate
        for class_id, indices in class_indices.items():
            total_class_samples = len(indices)
            num_class_samples = int(total_class_samples * rate)  # Samples each client gets from this class

            # Add the `rate` portion of the class to the client
            clients_indices[class_id].extend(indices[:num_class_samples])

            # The remaining portion of this class will be distributed to the other clients
            remaining_indices = indices[num_class_samples:]
            remaining_class_samples = len(remaining_indices)

            # Split the remaining samples among other clients
            other_classes = [i for i in class_indices.keys() if i != class_id]
            num_other_samples_per_client = remaining_class_samples // (num_clients - 1)
            # print(class_id, len(class_indices[class_id]), other_classes, num_other_samples_per_client)
            # Distribute the remaining samples to the other clients
            for j, other_class in enumerate(other_classes):
                start_idx = j * num_other_samples_per_client
                end_idx = start_idx + num_other_samples_per_client
                # print(class_id, other_client, len(remaining_indices), start_idx, end_idx)
                clients_indices[other_class].extend(remaining_indices[start_idx:end_idx])

        for client_id in range(num_clients):
            # Create Subset for this client
            client_dataset = Subset(dataset, clients_indices[client_id])
            client_datasets.append(client_dataset)

    return client_datasets


def main():
    # Example usage:
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    rate = 0.3  # Adjust between 0 (IID) and 1 (Extreme Non-IID)
    num_clients = 10  # Total number of clients
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Generate the non-IID data
    client_datasets = generate_non_iid_data(test_dataset, num_clients=num_clients, rate=rate)

    # Verify the data distribution for each client
    for i, client_data in enumerate(client_datasets):
        client_label = client_data.dataset.targets[client_data.indices].tolist()
        print(f"Client {i + 1}: {len(client_data)} samples, {collections.Counter(client_label)}")


if __name__ == '__main__':
    main()