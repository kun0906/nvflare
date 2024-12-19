"""
    Replace 10 individual VAEs with 1 conditional VAE (CVAE)

    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare
    $PYTHONPATH=. python3 auto_labeling/cnn_fl_cvae_attention.py


"""

import argparse
import collections
import multiprocessing as mp
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchvision import datasets, transforms

from auto_labeling.attention import aggregate_with_attention
from auto_labeling.data.MNIST.pretrained import pretrained_CNN
from utils import timer

print(os.path.abspath(os.getcwd()))

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Demo Script")

    # Add arguments to be parsed
    parser.add_argument('-r', '--label_rate', type=float, required=False, default=0.1,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=100,
                        help="The number of epochs (integer).")

    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
label_rate = args.label_rate
server_epochs = args.server_epochs

# For testing, print the parsed parameters
print(f"label_rate: {label_rate}")
print(f"server_epochs: {server_epochs}")



@timer
def gen_local_data(client_data_file, client_id, label_rate):
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
    #     return client_data

    dir_name = os.path.dirname(client_data_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    print(client_data_file)
    if 'mnist' in client_data_file:
        in_file = f'data/MNIST/data/{client_id}.pkl'
    elif 'shakespeare' in client_data_file:
        in_file = f'data/SHAKESPEARE/data/{client_id}.pkl'
    elif 'reddit' in client_data_file:
        in_file = f'data/REDDIT/data/{client_id}.pkl'
    elif 'sent140' in client_data_file:
        in_file = f'data/Sentiment140/data/{client_id}.pkl'
    elif 'pubmed' in client_data_file:
        in_file = f'data/PubMed/data/{client_id}.pkl'
    else:
        raise NotImplementedError

    with open(in_file, 'rb') as f:
        client_data = pickle.load(f)

    X = torch.tensor(client_data['X'])
    y = torch.tensor(client_data['y'])
    # y_names = torch.tensor(np.array(y_names, dtype=object))
    # Generate local data, and only lable_rate=10% of them has labels
    labels_mask = torch.tensor([False] * len(y), dtype=torch.bool)
    cnt = X.size(0)
    print(f"Class {client_id}: {cnt} images, y: {collections.Counter(y.tolist())}")
    sampling_size = int(cnt * label_rate)
    labeled_indices = torch.randperm(len(y))[:sampling_size]
    # labeled_X = X[labeled_indices]
    # labeled_y = y[labeled_indices]
    labels_mask[labeled_indices] = True

    features = X
    shared_test_features = torch.tensor(client_data['shared_data']['X'])  # Todo: need to use new test set, not X.
    shared_targets = torch.tensor(client_data['shared_data']['y'])
    client_data = {'features': features.clone().detach(), 'labels': y,
                   'labels_mask': labels_mask,  # only 10% data has labels.
                   'shared_test_data': {'features': shared_test_features.clone().detach(),
                                        'labels': shared_targets}
                   }

    # with open(client_data_file, 'wb') as f:
    #     # pickle.dump(client_data, f)
    torch.save(client_data, client_data_file)

    return client_data

# def vae_loss_function(recon_x, x, mu, logvar):
#     # reconstruction error
#     # BCE = nn.BCELoss(reduction='sum')(recon_x, x)
#     recon_loss = F.mse_loss(recon_x, x, reduction='sum')
#     # KL divergence term
#     # We assume standard Gaussian prior
#     # The KL term forces the latent distribution to be close to N(0, 1)
#     # KL[Q(z|x) || P(z)] = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     # where sigma^2 = exp(logvar)
#     # This is the standard VAE loss function
#     # recon_x: the reconstructed logits, x: the true logits
#     # mu: mean, logvar: log variance of the latent distribution
#     # We assume logvar is the log of variance (log(sigma^2))
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     # Return the total loss
#     beta = 0.1
#     info = (recon_loss.item(), beta * kl_loss.item())
#     return recon_loss + beta * kl_loss, info
#

def train_cvae(local_cvae, global_cvae, local_data, train_info={}):
    # Initialize local_cvae with global_cvae
    local_cvae.load_state_dict(global_cvae.state_dict())

    X, y = local_data['features'], local_data['labels']
    mask = local_data['labels_mask']
    # only use labeled data for training cvae
    X = X[mask]
    y = y[mask]
    # Only update available local labels, i.e., not all the local_cvaes will be updated.
    # local_labels = set(y.tolist())
    print(f'local labels: {collections.Counter(y.tolist())}, with {len(y)} samples.')

    local_cvae.to(device)
    optimizer = optim.Adam(local_cvae.parameters(), lr=0.001)

    ohe_labels = torch.zeros((len(y), len(LABELs))).to(device)  # One-hot encoding for class labels
    for i, l in enumerate(y.tolist()):  # if y is different.
        ohe_labels[i, l] = 1

    losses = []
    for epoch in range(300):
        # Convert X to a tensor and move it to the device
        # X.to(device)
        X = X.clone().detach().float().to(device)

        recon_logits, mu, logvar = local_cvae(X, ohe_labels)
        loss, info = vae_loss_function(recon_logits, X, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'train_cvae epoch: {epoch}, local_cvae loss: {loss.item() / X.shape[0]:.4f}')
        losses.append(loss.item())

    train_info['cvae'] = {"losses": losses}


def _gen_models(model, l, size, method='T5'):
    if method == 'T5':

        # pip install sentencepiece
        from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel

        # Initialize the BERT model and tokenizer for embedding extraction
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Initialize the T5 model and tokenizer
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        tokenizer = t5_tokenizer

        # Select a class label and concatenate with random latent vector
        class_label = l
        input_text = f"{class_label}: This is a sample text to generate embedding."

        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Generate text
        outputs = model.generate(**inputs)

        # Decode the output and get embeddings (e.g., use BERT or the model's own hidden states)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Now, use BERT to get the 768-dimensional embedding for the generated text
        encoded_input = bert_tokenizer(generated_text, return_tensors="pt", truncation=True, padding=True,
                                       max_length=512)
        with torch.no_grad():
            outputs = bert_model(**encoded_input)

        # Get the embedding from the [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        # Print the shape of the CLS embedding
        print(f"CLS Embedding Shape: {cls_embedding.shape}")
        embedding = cls_embedding

    else:  # default one is autoencoder
        cvae = model
        latent_dim = cvae.decoder.latent_dim
        # generate latent vector from N(0, 1)
        z = torch.randn(size, latent_dim).to(device)  # Sample latent vectors
        ohe_labels = torch.zeros((size, len(LABELs))).to(device)  # One-hot encoding for class labels
        ohe_labels[:, l] = 1
        pseudo_logits = cvae.decoder(z, ohe_labels)  # Reconstruct probabilities from latent space
        embedding = pseudo_logits

    return embedding


def gen_data(cvae, sizes):
    data = {}
    for l, size in sizes.items():
        cvae.to(device)
        pseudo_logits = _gen_models(cvae, l, size, method='cvae')
        pseudo_logits = pseudo_logits.detach().to(device)

        features = pseudo_logits
        data[l] = {'features': features, 'labels': [l] * size}

    return data


def merge_data(data, local_data):
    new_data = {'features': local_data['features'].to(device),
                'labels': local_data['labels'].to(device),
                'is_generated': torch.tensor(len(local_data['labels']) * [False]).to(
                    device)}  # Start with None for concatenation
    # tmp = {}
    for l, vs in data.items():
        size = len(vs['labels'])
        new_data['features'] = torch.cat((new_data['features'], vs['features']), dim=0)
        new_data['labels'] = torch.cat((new_data['labels'], torch.tensor(vs['labels'], dtype=torch.long).to(device)))
        new_data['is_generated'] = torch.cat((new_data['is_generated'], torch.tensor(size * [True]).to(device)))
    return new_data


def cosine_similarity_torch(features):
    import torch.nn.functional as F
    # Ensure features are normalized (cosine similarity requires normalized vectors)
    features = F.normalize(features, p=2, dim=1)  # L2 normalization along the feature dimension

    # Compute cosine similarity directly on the GPU
    similarity_matrix = torch.mm(features, features.t())  # Matrix multiplication for cosine similarity

    return similarity_matrix


@timer
def gen_edges(train_features, edge_method='cosine', train_info={}):
    if train_features.is_cuda:
        train_features = train_features.cpu().numpy()

    threshold = train_info['threshold']
    if edge_method == 'cosine':
        # Calculate cosine similarity to build graph edges (based on CNN features)
        similarity_matrix = cosine_similarity(train_features)  # [-1, 1]
        # similarity_matrix = cosine_similarity_torch(train_features)
        # Convert NumPy array to PyTorch tensor
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
        # #  # only keep the upper triangle of the matrix and exclude the diagonal entries
        # similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
        print(f'similarity matrix: {similarity_matrix.shape}')
        # Create graph: Each image is a node, edges based on similarity
        # threshold = torch.quantile(similarity_matrix, 0.9)  # input tensor is too large()
        # Convert the tensor to NumPy array
        if threshold is None:
            import scipy.stats as stats
            similarity_matrix_np = similarity_matrix.cpu().numpy()
            # Calculate approximate quantile using scipy
            thresholds = [(v, float(stats.scoreatpercentile(similarity_matrix_np.flatten(), v))) for v in
                          range(0, 100 + 1, 10)]
            print(thresholds)
            per = 99.0
            threshold = stats.scoreatpercentile(similarity_matrix_np.flatten(), per)  # per in [0, 100]
            train_info['threshold'] = threshold
        else:
            per = 99.0
        print('threshold', threshold)
        # Find indices where similarity exceeds the threshold
        edge_indices = (similarity_matrix > threshold).nonzero(as_tuple=False)  # two dimensional data [source, targets]
        print(f"total number of edges: {similarity_matrix.shape}, we only keep {100 - per:.2f}% edges "
              f"with edge_indices.shape: {edge_indices.shape}")
        edge_weight = similarity_matrix[edge_indices[:, 0], edge_indices[:, 1]]  # one dimensional data

    elif edge_method == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(train_features)
        # Convert NumPy array to PyTorch tensor
        distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
        # Convert the tensor to NumPy array
        if threshold is None:
            import scipy.stats as stats
            distance_matrix_np = distance_matrix.cpu().numpy()
            thresholds = [stats.scoreatpercentile(distance_matrix_np.flatten(), v) for v in range(0, 100 + 1, 10)]
            print(thresholds)
            # Calculate approximate quantile using scipy
            threshold = stats.scoreatpercentile(distance_matrix_np.flatten(), 0.009)  # per in [0, 100]
            # threshold = torch.quantile(distance_matrix, 0.5)  # input tensor is too large()
            train_info['threshold'] = threshold
        print('threshold', threshold)
        edge_indices = (distance_matrix < threshold).nonzero(as_tuple=False)
        # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))
        max_dist = max(distance_matrix_np)
        edge_weight = (max_dist - distance_matrix[edge_indices[:, 0], edge_indices[:, 1]]) / max_dist

    elif edge_method == 'knn':
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        # When using metric='cosine' in NearestNeighbors, it internally calculates
        # 1 - cosine similarity
        # so the distances returned are always non-negative [0, 2] (similarity:[-1, 1]).
        # Also, by default, the knn from sklearn includes each node as its own neighbor,
        # usually it will be the value in the results.
        knn.fit(train_features)
        distances, indices = knn.kneighbors(train_features)
        # Flatten source and target indices
        source_nodes = []
        target_nodes = []
        num_nodes = len(train_features)
        dists = []
        for node_i_idx in range(num_nodes):
            for j, node_j_idx in enumerate(indices[node_i_idx]):  # Neighbors of node `i`
                source_nodes.append(node_i_idx)
                target_nodes.append(node_j_idx)
                dists.append(distances[node_i_idx][j])

        # Stack source and target to create edge_index
        edge_indices = torch.tensor([source_nodes, target_nodes], dtype=torch.long).t()

        # one dimensional data
        edge_weight = (2 - torch.tensor(dists, dtype=torch.float32)) / 2  # dists is [0, 2], after this, values is [0,1]
        # edge_weight = torch.sparse_coo_tensor(edge_indices, values, size=(num_nodes, num_nodes)).to_dense()
        # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))

    elif edge_method == 'affinity':
        # from sklearn.feature_selection import mutual_info_classif
        # mi_matrix = mutual_info_classif(train_features, train_labels)
        # # Threshold mutual information to create edges
        # threshold = 0.1
        # edge_indices = (mi_matrix > threshold).nonzero(as_tuple=False)
        # edges = edge_indices.t().contiguous()

        from sklearn.cluster import AffinityPropagation
        affinity_propagation = AffinityPropagation(affinity='euclidean')
        affinity_propagation.fit(train_features)
        exemplars = affinity_propagation.cluster_centers_indices_
    else:
        raise NotImplementedError

    # Convert to edge list format (two rows: [source_nodes, target_nodes])
    edges = edge_indices.t().contiguous()

    return edges, edge_weight


def train_cnn(local_cnn, global_cvae, global_cnn, local_data, train_info={}):
    """
        1. Use vaes to generated data for each class
        2. Use the generated data + local data to train local cnn with initial parameters of global_cnn
        3. Send local cnn'parameters to server.
    Args:
        local_cnn:
        cvae:
        global_cnn:
        local_data:

    Returns:

    """
    local_cnn.to(device)
    local_cnn.load_state_dict(
        global_cnn.state_dict())  # Initialize client_gm with the parameters of global_model
    optimizer = optim.Adam(local_cnn.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()  # mean
    local_cnn.train()  #

    labels_mask = local_data['labels_mask']
    y = local_data['labels'][labels_mask]  # we assume on a tiny labeled data in the local dat
    size = len(y.tolist())
    print(f'local data size: {len(labels_mask)}, y: {size}, label_rate: {size / len(labels_mask)}')
    ct = collections.Counter(y.tolist())
    max_size = max(ct.values())
    max_size = int(max_size * 0.1)  # for each class, only generate 10% percent data to save computational resources.
    if max_size == 0: max_size = 1
    print(f'For each class, we only generate {max_size} samples, '
          f'and use labeled_classes_weights to address class imbalance issue.')
    sizes = {}
    labeled_cnt = {}
    for l in LABELs:
        if l in ct.keys():
            if max_size > ct[l]:
                sizes[l] = max_size - ct[l]
                labeled_cnt[l] = max_size
            else:
                labeled_cnt[l] = ct[l]
        else:
            sizes[l] = max_size
            labeled_cnt[l] = max_size

    # generated new data
    data = gen_data(global_cvae, sizes)
    data = merge_data(data, local_data)
    features = data['features']
    labels = data['labels']

    # Define train, val, and test masks
    generated_data_indices = list(range(len(labels_mask), len(labels)))
    # Get indices of y
    y_indices = labels_mask.nonzero(as_tuple=True)[0].tolist()
    indices = torch.tensor(y_indices + generated_data_indices).to(device)
    new_y = labels[indices]
    print(labeled_cnt.items(), flush=True)
    labeled_classes_weights = {k: sum(labeled_cnt.values()) / v for k, v in labeled_cnt.items()}
    labeled_classes_weights = {k: w / sum(labeled_classes_weights.values()) for k, w in
                               labeled_classes_weights.items()}  # normalize weights
    data['labeled_classes_weights'] = labeled_classes_weights
    print('new_y', collections.Counter(new_y.tolist()), ct.items(),
          '\nlabeled_classes_weights', {k: float(f"{v:.2f}") for k, v in labeled_classes_weights.items()})

    train_info['threshold'] = None
    edges, edge_weight = gen_edges(features, edge_method='knn', train_info=train_info)  # will update threshold
    print(f"edges.shape {edges.shape}, edge_weight min:{edge_weight.min()}, max:{edge_weight.max()}")
    # Create node features (features from CNN)
    # node_features = torch.tensor(features, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.long)
    node_features = features.clone().detach().float()
    labels = labels.clone().detach().long()
    # Prepare Graph data for PyG (PyTorch Geometric)
    print('Form graph data...')
    # # Define train, val, and test masks
    # generated_data_indices = list(range(len(labels_mask), len(labels)))
    # # Get indices of y
    # y_indices = labels_mask.nonzero(as_tuple=True)[0].tolist()
    # indices = torch.tensor(y_indices + generated_data_indices).to(device)
    # Define train_mask and test_mask
    train_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    test_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    train_mask[indices] = True
    test_mask[~train_mask] = True
    # val_mask = torch.tensor([False, False, True, False], dtype=torch.bool)
    # test_mask = torch.tensor([False, False, False, True], dtype=torch.bool)
    graph_data = Data(x=node_features, edge_index=edges, edge_weight=edge_weight,
                      y=labels, train_mask=train_mask, val_mask=None, test_mask=test_mask)
    # only train smaller model
    epochs_client = 5
    losses = []
    # here, you need make sure weight aligned with class order.
    class_weight = torch.tensor(list(data['labeled_classes_weights'].values()), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    x = graph_data.x[graph_data.train_mask].to(device)  # only use train_mask data for training in CNN
    y = graph_data.y[graph_data.train_mask].to(device)
    for epoch in range(epochs_client):
        epoch_model_loss = 0
        # _model_loss, _model_distill_loss = 0, 0
        # epoch_vae_loss = 0
        # _vae_recon_loss, _vae_kl_loss = 0, 0
        # data_size, data_dim = graph_data.x.shape
        # your local personal model
        outputs = local_cnn(x)  # only need x
        # Loss calculation: Only for labeled nodes
        # model_loss = criterion(outputs[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        model_loss = criterion(outputs, y)

        optimizer.zero_grad()
        model_loss.backward()

        # # Print gradients for each parameter
        # print("Gradients for model parameters:")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad}")
        #     else:
        #         print(f"{name}: No gradient (likely frozen or unused)")

        optimizer.step()
        print(f"train_cnn epoch: {epoch}, local_cnn loss: {model_loss.item():.4f}")
        losses.append(model_loss.item())

    train_info['cnn'] = {'graph_data': graph_data, "losses": losses}


def aggregate_cvaes(vaes, locals_info, global_cvae):
    print('*aggregate cvaes...')
    client_parameters_list = [local_cvae.state_dict() for client_i, local_cvae in vaes.items()]
    # aggregate(client_parameters_list, global_cvae)
    aggregate_method = 'global train'
    if aggregate_method == 'parameter':  # aggregate clients' parameters
        aggregate_with_attention(client_parameters_list, global_cvae, device)  # update global_cvae inplace
    else:
        # train a new model on generated data by clients' vaes
        # global_cvae.load_state_dict(global_cvae.state_dict())
        optimizer = optim.Adam(global_cvae.parameters(), lr=0.001)

        losses = []
        for epoch in range(101):
            loss_epoch = 0
            for client_i, local_cvae_state_dict in enumerate(client_parameters_list):
                label_cnts = locals_info[client_i]['label_cnts']
                # Initialize local_cvae with global_cvae
                local_cvae = CVAE(input_dim=input_dim, hidden_dim=32, latent_dim=5, num_classes=len(LABELs))
                local_cvae.load_state_dict(local_cvae_state_dict)
                local_cvae.to(device)

                z = []
                ohe_labels = []
                latent_dim = local_cvae.decoder.latent_dim
                for l, size in label_cnts.items():
                    # generate latent vector from N(0, 1)
                    z_ = torch.randn(size, latent_dim).to(device)  # Sample latent vectors
                    ohe_labels_ = torch.zeros((size, len(LABELs))).to(device)  # One-hot encoding for class labels
                    ohe_labels_[:, l] = 1

                    if len(z) == 0:
                        z = z_.clone().detach()
                        ohe_labels = ohe_labels_.clone().detach()
                    else:
                        z = torch.cat((z, z_), dim=0)
                        ohe_labels = torch.cat((ohe_labels, ohe_labels_), dim=0)

                z = z.to(device)
                ohe_labels = ohe_labels.to(device)
                pseudo_logits = local_cvae.decoder(z, ohe_labels)  # Reconstruct probabilities from latent space
                X_ = pseudo_logits

                # use the generated data X to train global_cvae
                recon_logits, mu, logvar = global_cvae(X_, ohe_labels)
                loss, info = vae_loss_function(recon_logits, X_, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            if epoch % 50 == 0:
                print(f'train global cvae epoch: {epoch}, global  cvae loss: {loss_epoch:.4f}')
            losses.append(loss_epoch)

            # train_info['cvae'] = {"losses": losses}


def aggregate_cnns(cnns, global_cnn):
    print('*aggregate cnn...')
    client_parameters_list = [local_cnn.state_dict() for client_i, local_cnn in cnns.items()]
    aggregate_with_attention(client_parameters_list, global_cnn, device)  # update global_cnn inplace


#
# class VAE(nn.Module):
#     def __init__(self, input_dim=10, hidden_dim=32, latent_dim=5):
#         super(VAE, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean of latent distribution
#         self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent distribution
#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_dim)
#
#     def encoder(self, x):
#         # print(x.device, self.fc1.weight.device, self.fc1.bias.device)
#         h1 = torch.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decoder(self, z):
#         h3 = torch.relu(self.fc3(z))
#         # return torch.sigmoid(self.fc4(h3))  # Sigmoid for normalized output
#         # return torch.softmax(self.fc4(h3), dim=1)  # softmax for normalized output
#         return self.fc4(h3)  # output
#
#     def forward(self, x):
#         mu, logvar = self.encoder(x)  # Flatten input logits
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar

import torch.nn as nn
import torch.nn.functional as F


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)  # Input features + class label
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, class_labels):
        # Concatenate input features with class labels
        x = torch.cat([x, class_labels], dim=-1)  # x should be normalized first?
        x = F.relu(self.fc1(x))
        mean = self.fc2(x)
        log_var = self.fc3(x)
        return mean, log_var


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_classes):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim + num_classes, hidden_dim)  # Latent vector + class label
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, class_labels):
        z = torch.cat([z, class_labels], dim=-1)
        x = F.relu(self.fc1(z))
        # x = torch.sigmoid(self.fc2(x))  # Sigmoid for MNIST data ?
        x = self.fc2(x)
        return x


# VAE
class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_classes)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, class_labels):
        mean, log_var = self.encoder(x, class_labels)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decoder(z, class_labels)
        return recon_x, mean, log_var


# VAE loss function
def vae_loss_function(recon_x, x, mean, log_var):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE = F.mse_loss(recon_x, x, reduction='mean')
    # KLD loss for the latent space
    # This assumes a unit Gaussian prior for the latent space
    # (Normal distribution prior, mean 0, std 1)
    # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD, {'BCE': BCE, 'KLD': KLD}


#
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import torch
#
# # Initialize the model and tokenizer
# model_name = "t5-small"  # You can replace this with "t5-base" or "t5-large"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
#
# # Example class labels for "comedy", "tragedy", "history"
# class_labels = ['comedy', 'tragedy', 'history']
#
# # Random latent vectors (sample 500 random latent points)
# random_latents = torch.randn(500, 768)  # Example random latent vectors
#
# # Generate embeddings
# generated_embeddings = []
#
# for i in range(500):
#     # Select a class label and concatenate with random latent vector
#     class_label = class_labels[i % len(class_labels)]  # Alternate through classes
#     input_text = f"{class_label}: This is a sample text to generate embedding."
#
#     # Tokenize the input
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#
#     # Generate text
#     outputs = model.generate(**inputs)
#
#     # Decode the output and get embeddings (e.g., use BERT or the model's own hidden states)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     # Optionally, extract embeddings from a model like BERT (e.g., using [CLS] token)
#     embedding = model.encoder(inputs['input_ids']).last_hidden_state[:, 0, :].detach().numpy()
#
#     # Store the generated embeddings
#     generated_embeddings.append(embedding)
#
# # Convert list to numpy array for further use
# generated_embeddings = np.array(generated_embeddings)


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN, self).__init__()
        out_channels = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)  # Assuming input size 28x28 (e.g., MNIST)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 10 classes for classification

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))  # X is node features, is (1000, 16)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


@timer
def evaluate(local_cnn, local_data, device, test_type='test', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.

        client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        client_data_ =  (graph_data_, feature_info, client_data_)
    """
    # After training, the model can make predictions for both labeled and unlabeled nodes
    print(f'***Testing cnn model on test_type:{test_type}...')
    # cnn = local_cnn(input_dim=64, hidden_dim=32, output_dim=10)
    # cnn.load_state_dict(client_result['client_gm'])
    cnn = local_cnn
    cnn.to(device)

    graph_data = train_info['cnn']['graph_data'].to(device)  # graph data
    x = graph_data.x.to(device)
    cnn.eval()
    with torch.no_grad():
        output = cnn(x)
        _, predicted_labels = torch.max(output, dim=1)

        # Calculate accuracy for the labeled data
        labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        print(f'labeled_indices {len(labeled_indices)}')
        true_labels = graph_data.y

        predicted_labels_tmp = predicted_labels[labeled_indices]
        true_labels_tmp = true_labels[labeled_indices]
        y = true_labels_tmp.cpu().numpy()
        y_pred = predicted_labels_tmp.cpu().numpy()

        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on labeled data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy

        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # Calculate accuracy for the unlabeled data
        labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        all_indices = torch.arange(graph_data.num_nodes)
        unlabeled_indices = all_indices[~torch.isin(all_indices, labeled_indices)]
        print(f'unlabeled_indices {len(unlabeled_indices)}')
        true_labels = graph_data.y

        predicted_labels_tmp = predicted_labels[unlabeled_indices]
        true_labels_tmp = true_labels[unlabeled_indices]
        y = true_labels_tmp.cpu().numpy()
        y_pred = predicted_labels_tmp.cpu().numpy()

        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on unlabeled data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # # Calculate accuracy for unlabeled data
        # print(f'Evaluate on all data (labeled + unlabeled)')
        # true_labels = client_data_[1]['labels']
        # y = true_labels.cpu().numpy()
        # y_pred = predicted_labels.cpu().numpy()
        # accuracy = accuracy_score(y, y_pred)
        # print(f"Accuracy on all data: {accuracy * 100:.2f}%")
        # if 'all' in test_type:
        #     client_result['accuracy_all'] = accuracy
        # else:
        #     client_result['accuracy'] = accuracy
        #
        # # Compute the confusion matrix
        # conf_matrix = confusion_matrix(y, y_pred)
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # if 'all' in test_type:
        #     client_result['cm_all'] = conf_matrix
        # else:
        #     client_result['cm'] = conf_matrix

    print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


from sklearn.neighbors import NearestNeighbors
import torch


def find_neighbors(new_node_features, existed_node_features, k=5):
    """
    Find the k-nearest neighbors of a new node based on cosine similarity.

    Args:
        new_node_feature (Tensor): Feature vector of the new node.
        node_features (Tensor): Feature matrix of the existing graph (shape: num_nodes x feature_dim).
        k (int): The number of neighbors to connect the new node to.

    Returns:
        torch.Tensor: A tensor containing edge indices (source, target) for the new node's neighbors.
    """
    # Combine the new node feature with the existing node features
    all_node_features = torch.cat([existed_node_features, new_node_features], dim=0)

    # Convert the node features to NumPy arrays (since scikit-learn's NearestNeighbors works with NumPy arrays)
    all_node_features_np = all_node_features.cpu().numpy()

    # Initialize NearestNeighbors with cosine similarity
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(all_node_features_np)  # Fit the k-NN model with all node features

    # Find the k nearest neighbors (including the new node itself, hence k+1)
    distances, indices = knn.kneighbors(new_node_features.cpu().numpy())

    # Prepare edge index pairs (source_node, target_node)
    # The new node is the source node, and the neighbors are the target nodes
    new_edges = []
    new_distances = []
    start_idx = len(existed_node_features)
    for i in range(len(new_node_features)):
        for j, neig_idx in enumerate(indices[i]):
            new_edges.append([start_idx + i, neig_idx])  # New node (source) -> Neighbor (target)
            new_distances.append(distances[i, j])

    # one dimensional data
    edge_weight = (2 - torch.tensor(new_distances,
                                    dtype=torch.float32)) / 2  # dists is [0, 2], after this, values is [0,1]
    # edge_weight = torch.sparse_coo_tensor(edge_indices, values, size=(num_nodes, num_nodes)).to_dense()
    # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))

    return torch.tensor(new_edges, dtype=torch.long).t(), edge_weight


@timer
def evaluate_shared_test(local_cnn, shared_test_data, device, test_type='shared_test_data', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    # After training, the model can make predictions for both labeled and unlabeled nodes
    print(f'***Testing cnn model on test_type:{test_type}...')

    new_features = shared_test_data['features'].to(device)
    new_labels = shared_test_data['labels'].to(device)
    print(f'new_features: {new_features.size()}, {collections.Counter(new_labels.tolist())}')
    if True:  # evaluate on new data
        # Add new_data to the node feature matrix (if you're adding a new node)
        # This could involve concatenating the new feature to the existing node features
        graph_data = train_info['cnn']['graph_data'].to(device)
        node_features = graph_data.x
        y = graph_data.y
        start_idx = len(y)
        # Add edges to the graph (example: add an edge from the new node to its neighbors)
        # Assuming you have a function `find_neighbors` to determine which nodes to connect
        # new_node_feature =  torch.tensor([features], dtype=torch.float32)  # New node's feature vector
        new_edges, new_weight = find_neighbors(new_features, node_features,
                                               k=5)  # (source, target) format for the new edges
        new_edges = new_edges.to(device)
        new_weight = new_weight.to(device)
        # print(f"device: {graph_data.edge_index.device}, {graph_data.edge_weight.device}, {new_edges.device}, {new_weight.device}")
        edges = torch.cat([graph_data.edge_index, new_edges], dim=1)  # Add new edges
        edge_weight = torch.cat([graph_data.edge_weight, new_weight])  # Add new edges
        features = torch.cat((node_features, new_features), dim=0)  # Adding to the existing node features
        labels = torch.cat((y, new_labels)).to(device)
    else:
        # will use the threshold obtained from training set.
        edges, edge_weight = gen_edges(features, edge_method='knn', train_info=train_info)
        start_idx = 0
    print(f"edges.shape {edges.shape}, edge_weight min:{edge_weight.min()}, max:{edge_weight.max()}")
    # Create node features (features from CNN)
    # node_features = torch.tensor(features, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.long)
    node_features = features.clone().detach().float()
    labels = labels.clone().detach().long()
    # Prepare Graph data for PyG (PyTorch Geometric)
    print('Form graph data...')

    graph_data = Data(x=node_features, edge_index=edges, edge_weight=edge_weight,
                      y=labels)
    graph_data.to(device)
    x = graph_data.x.to(device)
    # evaluate the data
    cnn = local_cnn
    cnn.to(device)
    cnn.eval()
    with torch.no_grad():
        output = cnn(x)
        _, predicted_labels = torch.max(output, dim=1)
        predicted_labels = predicted_labels[start_idx:]
        # Calculate accuracy for the labeled data
        # num_classes = 10
        # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        # print(f'labeled_indices {len(labeled_indices)}')
        true_labels = graph_data.y[start_idx:]

        # predicted_labels = predicted_labels[graph_data.train_mask]
        # true_labels = true_labels[graph_data.train_mask]
        # predicted_labels = predicted_labels[graph_data.test_mask]
        # true_labels = true_labels[graph_data.test_mask]

        y = true_labels.cpu().numpy()
        y_pred = predicted_labels.cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        # if 'all' in test_type:
        #     client_result['labeled_cm_all'] = conf_matrix
        # else:
        #     client_result['labeled_cm'] = conf_matrix

    print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


def print_histories(histories):
    num_server_epoches = len(histories)
    num_clients = len(histories[0])
    num_classes = len(histories[0][0]["cvae"])
    print('num_server_epoches:', num_server_epoches, ' num_clients:', num_clients, ' num_classes:', num_classes)
    for c in range(num_clients):
        print(f"\n\nclient {c}")
        for s in range(num_server_epoches):
            client = histories[s][c]
            local_cvae = client['cvae']
            local_cnn = client['cnn']
            print(f'\t*local cvae:', local_cvae.keys(), f' server_epoch: {s}')
            losses_ = [float(f"{v:.2f}") for v in local_cvae['losses']]
            print(f'\t\tlocal cvae:', losses_)
            # print('\t*local cnn:', [f"{v:.2f}" for v in local_cnn['losses']])


def client_process(c, epoch, global_cvae, global_cnn, input_dim, num_classes, label_rate, in_dir, prefix, device):
    """
    Function to be executed in a separate process for each client.
    """
    print(f"\n\n***server_epoch:{epoch}, client_{c} ...")
    l = c  # we should have 'num_clients = num_labels'
    train_info = {"cvae": {}, "cnn": {}}

    # Load local data
    local_data = gen_local_data(client_data_file=f'{in_dir}/c_{c}-{prefix}-data.pth', client_id=c,
                                label_rate=label_rate)
    label_cnts = collections.Counter(local_data['labels'].tolist())
    print(f'client_{c} data:', label_cnts)
    local_info = {'label_cnts': label_cnts}
    # Train CVAE
    local_cvae = CVAE(input_dim=input_dim, hidden_dim=32, latent_dim=5, num_classes=num_classes)
    print('train_cvae...')
    train_cvae(local_cvae, global_cvae, local_data, train_info)

    # Train CNN
    print('train_cnn...')
    local_cnn = CNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
    train_cnn(local_cnn, global_cvae, global_cnn, local_data, train_info)

    # Evaluate CNN
    print('evaluate_cnn...')
    evaluate(local_cnn, None, device, test_type='Testing on client data', client_id=c, train_info=train_info)
    evaluate_shared_test(local_cnn, local_data['shared_test_data'], device, \
                         test_type='Testing on shared test data', client_id=c, train_info=train_info)

    return c, local_cvae, local_info, local_cnn, train_info


@timer
def main(in_dir, input_dim=16):
    num_clients = len(LABELs)
    print(f'input_dim: {input_dim}')

    prefix = f'r_{label_rate}'
    # Generate local data for each client first
    for c in range(num_clients):
        print(f'\nGenerate local data for client_{c}...')
        gen_local_data(client_data_file=f'{in_dir}/c_{c}-{prefix}-data.pth', client_id=c,
                       label_rate=label_rate)

    num_classes = len(LABELs)
    global_cvae = CVAE(input_dim=input_dim, hidden_dim=32, latent_dim=5, num_classes=num_classes)
    global_cnn = CNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)

    debug = True
    if debug:
        histories = []
        for epoch in range(server_epochs):
            # update clients
            cvaes = {}
            locals_info = {}
            cnns = {}
            history = {}
            for c in range(num_clients):
                print(f"\n\n***server_epoch:{epoch}, client_{c} ...")
                # l = c  # we should have 'num_clients = num_labels'
                train_info = {"cvae": {}, "cnn": {}, }
                # local_data = features_info[l]
                local_data = gen_local_data(client_data_file=f'{in_dir}/c_{c}-{prefix}-data.pth', client_id=c,
                                            label_rate=label_rate)
                label_cnts = collections.Counter(local_data['labels'].tolist())
                print(f'client_{c} data:', label_cnts)
                local_cvae = CVAE(input_dim=input_dim, hidden_dim=32, latent_dim=5, num_classes=num_classes)
                print('train_cvae...')
                train_cvae(local_cvae, global_cvae, local_data, train_info)
                cvaes[c] = local_cvae
                locals_info[c] = {'label_cnts': label_cnts}

                print('train_cnn...')
                local_cnn = CNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
                train_cnn(local_cnn, global_cvae, global_cnn, local_data, train_info)  # will generate new data
                cnns[c] = local_cnn

                print('evaluate_cnn...')
                # self.evaluate(client_result, graph_data_, device, test_type='train', client_id=client_id)
                evaluate(local_cnn, None, device,
                         test_type='Testing on client data', client_id=c, train_info=train_info)
                # # Note that client_result will be overridden or appended more.
                evaluate_shared_test(local_cnn, local_data['shared_test_data'], device,
                                     test_type='Testing on shared test data', client_id=c, train_info=train_info)
                history[c] = train_info
            # server aggregation
            aggregate_cvaes(cvaes, locals_info, global_cvae)
            aggregate_cnns(cnns, global_cnn)

            histories.append(history)
    else:
        import multiprocessing
        # Set start method to 'spawn' for CUDA compatibility
        multiprocessing.set_start_method('spawn', force=True)

        histories = []
        for epoch in range(server_epochs):

            cvaes = {}
            locals_info = {}
            cnns = {}
            history = {}
            with mp.Pool(processes=num_clients) as pool:
                # Use apply_async or map to execute client_process concurrently and get results
                results = [pool.apply_async(client_process, args=(
                    c, epoch, global_cvae, global_cnn, input_dim, num_classes, label_rate, in_dir, prefix, device))
                           for c in range(num_clients)]
                # Wait for all results to finish and collect them
                results = [r.get() for r in results]  # return c, local_cvae, local_cnn, train_info
                for r in results:
                    c, cvae, local_info, cnn, train_info = r  # it will run when you call r.get().
                    cvaes[c] = cvae
                    locals_info[c] = local_info
                    cnns[c] = cnn
                    history[c] = train_info

            # Server aggregation
            aggregate_cvaes(cvaes, locals_info, global_cvae)
            aggregate_cnns(cnns, global_cnn)
            # Collect histories
            histories.append(history)

    prefix += f'-n_{server_epochs}'
    history_file = f'{in_dir}/histories_cvae_{prefix}.pth'
    print(f'saving histories to {history_file}')
    # with open(history_file, 'wb') as f:
    #     pickle.dump(histories, f)
    torch.save(histories, history_file)

    # print_histories(histories)


if __name__ == '__main__':
    in_dir = 'fl/mnist'
    input_dim = 16
    LABELs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    #
    # in_dir = 'fl/sent140'
    # input_dim = 768
    # LABELs = {0, 1}

    # in_dir = 'fl/shakespeare'
    # input_dim = 768
    # LABELs = {0, 1, 2}

    # in_dir = 'fl/reddit'
    # input_dim = 768
    # LABELs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    # in_dir = 'fl/pubmed'
    # input_dim = 500
    # LABELs = {0, 1, 2}

    main(in_dir, input_dim)
    # history_file = f'{in_dir}/histories_cvae.pkl'
    # with open(history_file, 'rb') as f:
    #     histories = pickle.load(f)
    # print_histories(histories)
