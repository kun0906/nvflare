import torch

def aggregate_logits_weighted(client_logits_list, num_samples_list):
    # Validate input: ensure num_samples_list aligns with client_logits_list
    assert len(client_logits_list) == len(num_samples_list)

    total_samples = sum(num_samples_list)
    # Determine the maximum number of samples (N) across all clients
    max_samples = max(logits.shape[0] for logits in client_logits_list)
    num_classes = client_logits_list[0].shape[1]  # Number of classes (C)

    # Initialize a zero tensor for aggregation
    weighted_logits = torch.zeros(max_samples, num_classes)

    for logits, num_samples in zip(client_logits_list, num_samples_list):
        weight = num_samples / total_samples
        weighted_logits[:logits.shape[0], :] += weight * logits  # Weighted contribution

    return weighted_logits


# Example
client_1_logits = torch.rand(5, 10)  # 5 samples, 10 classes
client_2_logits = torch.rand(7, 10)  # 7 samples, 10 classes
client_3_logits = torch.rand(6, 10)  # 6 samples, 10 classes

client_logits_list = [client_1_logits, client_2_logits, client_3_logits]
num_samples_list = [5, 7, 6]  # Number of samples per client

aggregated_logits = aggregate_logits_weighted(client_logits_list, num_samples_list)
print("Shape of aggregated_logits:", aggregated_logits.shape)


