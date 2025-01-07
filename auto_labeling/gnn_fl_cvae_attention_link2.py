"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare
    $PYTHONPATH=. python3 auto_labeling/gnn_fl_cvae_attention.py


"""

import argparse
import collections
import multiprocessing as mp
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch_geometric.data import Data

from auto_labeling.attention import aggregate_with_attention
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
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=10,
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
# print(f"label_rate: {label_rate}")
print(f"server_epochs: {server_epochs}")


@timer
def gen_local_data(client_data_file, client_id, label_rate=0.1):
    """ We assume num_client = num_classes, i.e., each client only has one class data

    Args:
        client_id:
    Returns:

    """
    # if os.path.exists(client_data_file):
    #     with open(client_data_file, "rb") as f:
    #         client_data = torch.load(f, weights_only=True)
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
    elif 'cora' in client_data_file:
        in_file = f'data/Cora/data/{client_id}.pkl'
    else:
        raise NotImplementedError

    with open(in_file, 'rb') as f:
        client_data = pickle.load(f)

    # Each client has local ((train+val set) 10% labeled and (test set) 90% unlabeled) data + global (test) data
    X = torch.tensor(client_data['X'])  # local data
    y = torch.tensor(client_data['y'])  # local data
    indices = torch.tensor(client_data['indices'])  # local data indices (obtained from original data indices)
    original_indices = torch.tensor(client_data['original_indices'])
    train_mask = torch.tensor(client_data['train_mask'])
    val_mask = torch.tensor(client_data['val_mask'])
    test_mask = torch.tensor(client_data['test_mask'])
    edge_indices = torch.tensor(client_data['edge_indices'])  # local indices (obtained from local data indices)
    unqiue_edges = set([(b, a) if a > b else (a, b) for a, b in edge_indices.t().tolist()])
    print(f'unique edges: {len(unqiue_edges)} =? edges/2: {edge_indices.shape[1] / 2}')

    shared_X = torch.tensor(client_data['all_data']['X'])  # global data x
    shared_y = torch.tensor(client_data['all_data']['y'])  # global data y
    shared_indices = torch.tensor(client_data['all_data']['indices'])  # global data indices
    shared_train_mask = torch.tensor(client_data['all_data']['train_mask'])  # shared_train_mask
    shared_val_mask = torch.tensor(client_data['all_data']['val_mask'])  # shared_val_mask
    shared_test_mask = torch.tensor(client_data['all_data']['test_mask'])  # shared_test_mask
    shared_edge_indices = torch.tensor(client_data['all_data']['edge_indices'])
    client_data = {
        # local client data
        'X': X, 'y': y, 'indices': indices, 'original_indices': original_indices,
        'edge_indices': edge_indices,
        'train_mask': train_mask,  # only 10% data has labels.
        'val_mask': val_mask,  # local data train, val, and test mask
        'test_mask': test_mask,

        # global all data
        'all_data': {'X': shared_X,
                     'y': shared_y, 'indices': shared_indices,
                     'train_mask': shared_train_mask,
                     'val_mask': shared_val_mask,
                     'test_mask': shared_test_mask,
                     'edge_indices': shared_edge_indices,
                     'edge_indices_train': torch.tensor(client_data['all_data']['edge_indices_train']),
                     'edge_indices_val': torch.tensor(client_data['all_data']['edge_indices_val']),
                     'edge_indices_test': torch.tensor(client_data['all_data']['edge_indices_test']),
                     },
    }

    print(f'Client data range: min: {min(X.flatten().tolist())}, '
          f'max:{max(X.flatten().tolist())}')

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

    X, y = local_data['X'], local_data['y']
    mask = local_data['train_mask']
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

        if epoch % 100 == 0:
            print(f'train_cvae epoch: {epoch}, local_cvae loss: {loss.item():.4f}')
        losses.append(loss.item())

    train_info['cvae'] = {"losses": losses}


def binary_loss_function(y_prob, y):
    # Convert y to one-hot encoding
    y_one_hot = torch.zeros_like(y_prob)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # One-hot encode
    BCE = F.binary_cross_entropy(y_prob, y_one_hot, reduction='mean')
    # BCE = F.mse_loss(recon_x, x, reduction='mean')
    # # KLD loss for the latent space
    # # This assumes a unit Gaussian prior for the latent space
    # # (Normal distribution prior, mean 0, std 1)
    # # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # return BCE + KLD, {'BCE': BCE, 'KLD': KLD}
    return BCE


import torch.optim as optim
from torch_geometric.utils import negative_sampling


def train_link_predictor(local_lp, global_lp, local_data, train_info={}):
    local_lp.load_state_dict(global_lp.state_dict())

    optimizer = optim.Adam(local_lp.parameters(), lr=0.001)

    X, y = local_data['X'], local_data['y']  # use all local edges
    # mask = local_data['train_mask']
    # # only use labeled data for training edge
    # X = X[mask]
    # y = y[mask]

    pos_edge_indices = local_data['edge_indices']
    # set_edges = set([(a, b) for a, b in edges.tolist()])
    # set_edges = set(map(tuple, edges.tolist()))

    losses = []
    for epoch in range(500):
        local_lp.train()
        optimizer.zero_grad()

        # Generate negative edges
        neg_edge_indices = negative_sampling(
            edge_index=pos_edge_indices,
            num_nodes=len(X),
            num_neg_samples=pos_edge_indices.size(1)  # excluding the generated data
        )

        # edge_indices = torch.cat((pos_edge_indices, neg_edge_indices), dim=1)

        # Forward pass only use pos_edge_indices
        z = local_lp(X, pos_edge_indices)

        pos_out = local_lp.decode(z, pos_edge_indices)
        neg_out = local_lp.decode(z, neg_edge_indices)
        out = torch.cat((pos_out, neg_out), dim=0)

        y_true = torch.cat([torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])], dim=0)
        loss = F.binary_cross_entropy_with_logits(out.flatten(), y_true)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'train_lp epoch: {epoch}, local_lp loss: {loss.item():.4f}')

            # Evaluate local_lp performance
            local_lp.eval()
            with torch.no_grad():
                # Predictions and ground truth
                y_prob = torch.sigmoid(out).detach().numpy()
                y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
                y_true = y_true.detach().numpy()
                # from sklearn.metrics import roc_auc_score, average_precision_score
                # auc = roc_auc_score(y_true, y_pred)
                # ap = average_precision_score(y_true, y_pred)
                # only on test set
                print('Evaluate on the training data...')
                accuracy = accuracy_score(y_true, y_pred)
                print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
                # if 'all' in test_type:
                #     client_result['labeled_accuracy_all'] = accuracy
                # else:
                #     client_result['labeled_accuracy'] = accuracy

                # # Compute the confusion matrix
                # conf_matrix = confusion_matrix(y_true, y_pred)
                # print("Confusion Matrix:")
                # print(conf_matrix)
                # # if 'all' in test_type:
                # #     client_result['labeled_cm_all'] = conf_matrix
                # # else:
                # #     client_result['labeled_cm'] = conf_matrix

        losses.append(loss.item())

    print_histgram(F.sigmoid(out).detach().numpy())
    train_info['edge'] = {"losses": losses}


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
        data[l] = {'X': features, 'y': [l] * size}
        print(f'Generated data range for class {l}: min: {min(pseudo_logits.flatten().tolist())}, '
              f'max:{max(pseudo_logits.flatten().tolist())}')
    return data


def merge_data(data, local_data):
    new_data = {'X': local_data['X'].to(device),
                'y': local_data['y'].to(device),
                'is_generated': torch.tensor(len(local_data['y']) * [False]).to(
                    device)}  # Start with None for concatenation
    # tmp = {}
    for l, vs in data.items():
        size = len(vs['y'])
        new_data['X'] = torch.cat((new_data['X'], vs['X']), dim=0)
        new_data['y'] = torch.cat((new_data['y'], torch.tensor(vs['y'], dtype=torch.long).to(device)))
        new_data['is_generated'] = torch.cat((new_data['is_generated'], torch.tensor(size * [True]).to(device)))
    return new_data


def cosine_similarity_torch(features):
    import torch.nn.functional as F
    # Ensure features are normalized (cosine similarity requires normalized vectors)
    features = F.normalize(features, p=2, dim=1)  # L2 normalization along the feature dimension

    # Compute cosine similarity directly on the GPU
    similarity_matrix = torch.mm(features, features.t())  # Matrix multiplication for cosine similarity

    return similarity_matrix


def print_histgram(new_probs):
    print(f'***Print histgram of new_probs, min:{min(new_probs)}, max: {max(new_probs)}***')
    # # Convert the probabilities to numpy for histogram calculation
    # new_probs = new_probs.detach().cpu().numpy()
    # Compute histogram
    hist, bin_edges = torch.histogram(torch.tensor(new_probs), bins=5)
    # Print histogram
    for i in range(len(hist)):
        print(f"\tBin {i}: Probability Range ({bin_edges[i]}, {bin_edges[i + 1]}), Frequency: {hist[i]}")


@timer
def gen_edges(train_features, local_size, global_lp, existed_edge_indices=None, edge_method='cosine',
              generated_size=0, local_data=None, train_info={}):
    if train_features.is_cuda:
        train_features = train_features.cpu().numpy()

    # debug = False
    # if debug:
    #     edge_indices = torch.combinations(torch.arange(len(train_features)), r=2).t()
    #     edge_weights = torch.ones((edge_indices.shape[1], ))
    #     return edge_indices, edge_weights
    edge_threshold = 0.8
    train_info['edge_threshold'] = edge_threshold
    print(f'edges between existed nodes ({len(train_features[:local_size, :])}): {existed_edge_indices.shape}')
    existed_weights = [1] * existed_edge_indices.size(1)  # existed_edge_indices.shape is 2xN
    if train_features.shape[0] == local_size:
        print('No generated data.')
        return existed_edge_indices, torch.tensor(existed_weights, dtype=torch.float)

    # If current client has classes (0, 1, 2, 3), then predict edges for new nodes (such as, 4, 5, 6)
    new_nodes = train_features[local_size:, :]
    new_node_pairs = torch.combinations(torch.arange(len(new_nodes)), r=2).t()
    global_lp.eval()  # Set model to evaluation mode
    # If no graph structure is provided, you can assume a fully connected graph for new_nodes.
    # Create a complete edge_index for all nodes in new_nodes:
    z = global_lp.encoder(new_nodes, edge_index=new_node_pairs)  # Embeddings for test nodes (no edges)
    new_probs = F.sigmoid(global_lp.decoder(z, new_node_pairs))
    print_histgram(new_probs.detach().cpu().numpy())
    new_edges = new_node_pairs.t()[new_probs.flatten() > edge_threshold].t()
    # adjust new_edges indices
    new_edges = local_size + new_edges  # src + local_size, dst + local_size
    # new_weights = [1] * len(new_edges) # not correct
    new_weights = [1] * new_edges.shape[1]
    print(f'new edges between new nodes ({len(new_nodes)}): {new_edges.shape}')

    # Predict edges between new and existed nodes
    existed_nodes = train_features[:local_size, :]
    cross_pairs = torch.cartesian_prod(torch.arange(0, local_size),
                                       torch.arange(local_size, local_size + len(new_nodes))).t()
    # # z = global_lp(existed_nodes, existed_new_pairs)
    # features = torch.cat((existed_nodes, new_nodes), dim=0)
    # Assuming `model` is your trained GNN model
    global_lp.eval()  # Set model to evaluation mode
    # Compute embeddings for existing and test nodes
    z_existed = global_lp.encoder(existed_nodes, existed_edge_indices)  # Embeddings for existing nodes
    z_test = global_lp.encoder(new_nodes, edge_index=new_node_pairs)  # Embeddings for test nodes (no edges)
    z = torch.cat([z_existed, z_test], dim=0)
    cross_probs = F.sigmoid(global_lp.decoder(z, cross_pairs))
    print_histgram(cross_probs.detach().cpu().numpy())
    cross_edges = cross_pairs.t()[cross_probs.flatten() > edge_threshold].t()
    # adjust cross_edges indices for new nodes
    # cross_edges[1, :] = local_size + cross_edges[1, :]  # new_edges.shape is 2xN, (src, dst+local_size)
    # cross_edges = torch.zeros((2, 0), dtype=torch.long)
    cross_weights = [1] * cross_edges.shape[1]
    print(f'cross edges between existed nodes ({len(existed_nodes)}) and new nodes ({len(new_nodes)}): '
          f'{cross_edges.shape}')

    # debug = True
    # if debug: we don't need this check here because in this section, we focus on train and generated data.
    #     get_cross_edge_info(cross_edges.t().numpy(), local_size, generated_size)
    #     # get ground truth cross edges between local data and test data from original edges
    #     original_local_indices = local_data['original_indices']
    #     all_data = local_data['all_data']
    #     # shared_test_mask = all_data['test_mask']
    #     # original_test_indices = all_data['indices'][shared_test_mask]
    #     edges_set = set([(i, j) for i, j in all_data['edge_indices'].numpy().T])
    #     cross_edges2 = extract_edges(original_local_indices, edges_set)
    #     check_edges(cross_edges.t().numpy(), cross_edges2)

    # Combine all edges
    edge_indices = torch.cat([existed_edge_indices, new_edges, cross_edges], dim=1)
    edge_weights = existed_weights + new_weights + cross_weights
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    print(f'total edges between all nodes: {edge_indices.shape}')

    return edge_indices, edge_weights

    # # n = len(train_features)
    # # X1 = []
    # # X2 = []
    # # new_edges = []
    # # for i in range(n):
    # #     for j in range(n):
    # #         if i == j: continue
    # #         X1.append(train_features[i])
    # #         X2.append(train_features[j])
    # #         new_edges.append([i, j])
    # # Assume train_features is a 2D NumPy array of shape (n, feature_dim)
    # n = len(train_features)
    #
    # # Create index pairs
    # row_indices, col_indices = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    # mask = row_indices != col_indices  # Exclude self-loops
    #
    # # Get valid index pairs
    # valid_rows = row_indices[mask]
    # valid_cols = col_indices[mask]
    #
    # # Create feature pairs
    # X1 = train_features[valid_rows]
    # X2 = train_features[valid_cols]
    #
    # # Edge indices
    # new_edges = np.stack([valid_rows, valid_cols], axis=1)
    #
    # outputs = global_lp(torch.tensor(np.asarray(X1)), torch.tensor(np.asarray(X2)))
    # y_pred = torch.argmax(outputs, dim=1)
    #
    # tmp = []
    # for i, y_pred_ in enumerate(y_pred.tolist()):
    #     if y_pred_ == 1:
    #         tmp.append(new_edges[i])
    # edge_indices = torch.tensor(np.asarray(tmp)).t()
    # edge_weight = torch.tensor([1] * len(tmp))
    # print(f'predicted edges: {len(edge_indices)}')
    # # threshold = train_info['threshold']
    # # if edge_method == 'cosine':
    # #     # Calculate cosine similarity to build graph edges (based on CNN features)
    # #     similarity_matrix = cosine_similarity(train_features)  # [-1, 1]
    # #     # Set diagonal items to 0
    # #     np.fill_diagonal(similarity_matrix, 0)
    # #     # similarity_matrix = cosine_similarity_torch(train_features)
    # #     # Convert NumPy array to PyTorch tensor
    # #     similarity_matrix = torch.abs(torch.tensor(similarity_matrix, dtype=torch.float32))
    # #     # #  # only keep the upper triangle of the matrix and exclude the diagonal entries
    # #     # similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
    # #     print(f'similarity matrix: {similarity_matrix.shape}')
    # #     # Create graph: Each image is a node, edges based on similarity
    # #     # threshold = torch.quantile(similarity_matrix, 0.9)  # input tensor is too large()
    # #     # Convert the tensor to NumPy array
    # #     if threshold is None:
    # #         import scipy.stats as stats
    # #         similarity_matrix_np = similarity_matrix.cpu().numpy()
    # #         # Calculate approximate quantile using scipy
    # #         thresholds = [(v, float(stats.scoreatpercentile(similarity_matrix_np.flatten(), v))) for v in
    # #                       range(0, 100 + 1, 10)]
    # #         print(thresholds)
    # #         per = 90.0
    # #         threshold = stats.scoreatpercentile(similarity_matrix_np.flatten(), per)  # per in [0, 100]
    # #         train_info['threshold'] = threshold
    # #     else:
    # #         per = 99.0
    # #     print('threshold', threshold)
    # #     # Find indices where similarity exceeds the threshold
    # #     edge_indices = (torch.abs(similarity_matrix) > threshold).nonzero(
    # #         as_tuple=False)  # two dimensional data [source, targets]
    # #     print(f"total number of edges: {similarity_matrix.shape}, we only keep {100 - per:.2f}% edges "
    # #           f"with edge_indices.shape: {edge_indices.shape}")
    # #     edge_weight = similarity_matrix[edge_indices[:, 0], edge_indices[:, 1]]  # one dimensional data
    # #
    # # elif edge_method == 'euclidean':
    # #     # from sklearn.metrics.pairwise import euclidean_distances
    # #     # distance_matrix = euclidean_distances(train_features)
    # #     # # Convert NumPy array to PyTorch tensor
    # #     # distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
    # #     # # Convert the tensor to NumPy array
    # #     # if threshold is None:
    # #     #     import scipy.stats as stats
    # #     #     distance_matrix_np = distance_matrix.cpu().numpy()
    # #     #     thresholds = [stats.scoreatpercentile(distance_matrix_np.flatten(), v) for v in range(0, 100 + 1, 10)]
    # #     #     print(thresholds)
    # #     #     # Calculate approximate quantile using scipy
    # #     #     threshold = stats.scoreatpercentile(distance_matrix_np.flatten(), 0.009)  # per in [0, 100]
    # #     #     # threshold = torch.quantile(distance_matrix, 0.5)  # input tensor is too large()
    # #     #     train_info['threshold'] = threshold
    # #     # print('threshold', threshold)
    # #     # edge_indices = (distance_matrix < threshold).nonzero(as_tuple=False)
    # #     # # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))
    # #     # max_dist = max(distance_matrix_np)
    # #     # edge_weight = (max_dist - distance_matrix[edge_indices[:, 0], edge_indices[:, 1]]) / max_dist
    # #     pass
    # #
    # # elif edge_method == 'knn':
    # #     from sklearn.neighbors import NearestNeighbors
    # #     knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    # #     # When using metric='cosine' in NearestNeighbors, it internally calculates
    # #     # 1 - cosine similarity
    # #     # so the distances returned are always non-negative [0, 2] (similarity:[-1, 1]).
    # #     # Also, by default, the knn from sklearn includes each node as its own neighbor,
    # #     # usually it will be the value in the results.
    # #     knn.fit(train_features)
    # #     distances, indices = knn.kneighbors(train_features)
    # #     # Flatten source and target indices
    # #     source_nodes = []
    # #     target_nodes = []
    # #     num_nodes = len(train_features)
    # #     dists = []
    # #     for node_i_idx in range(num_nodes):
    # #         for j, node_j_idx in enumerate(indices[node_i_idx]):  # Neighbors of node `i`
    # #             if node_i_idx == node_j_idx: continue
    # #             source_nodes.append(node_i_idx)
    # #             target_nodes.append(node_j_idx)
    # #             dists.append(distances[node_i_idx][j])
    # #
    # #     # Stack source and target to create edge_index
    # #     edge_indices = torch.tensor([source_nodes, target_nodes], dtype=torch.long).t()
    # #
    # #     # one dimensional data: large dists, smaller weights
    # #     edge_weight = (2 - torch.tensor(dists, dtype=torch.float32)) / 2  # dists is [0, 2], after this, values is [0,1]
    # #     # edge_weight = torch.sparse_coo_tensor(edge_indices, values, size=(num_nodes, num_nodes)).to_dense()
    # #     # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))
    # #
    # # elif edge_method == 'affinity':
    # #     # # from sklearn.feature_selection import mutual_info_classif
    # #     # # mi_matrix = mutual_info_classif(train_features, train_labels)
    # #     # # # Threshold mutual information to create edges
    # #     # # threshold = 0.1
    # #     # # edge_indices = (mi_matrix > threshold).nonzero(as_tuple=False)
    # #     # # edges = edge_indices.t().contiguous()
    # #     #
    # #     # from sklearn.cluster import AffinityPropagation
    # #     # affinity_propagation = AffinityPropagation(affinity='euclidean')
    # #     # affinity_propagation.fit(train_features)
    # #     # exemplars = affinity_propagation.cluster_centers_indices_
    # #     pass
    # # else:
    # #     raise NotImplementedError
    # #
    # # # Convert to edge list format (two rows: [source_nodes, target_nodes])
    # # edge_indices = edge_indices.t().contiguous()
    #
    # if existed_edge_indices is None:
    #     # edge_indices = edge_indices
    #     pass
    # else:
    #     # merge edge indices
    #     dt = set([(i, j) for i, j in existed_edge_indices.tolist()])
    #     existed_weights = [1] * len(existed_edge_indices)  # is it too large?
    #
    #     if len(edge_weight) == 0:
    #         return torch.tensor(existed_edge_indices, dtype=torch.long).t(), torch.tensor(existed_weights,
    #                                                                                       dtype=torch.float32)
    #
    #     edge_weight = edge_weight.tolist()
    #     print(f'edge_weight: min: {min(edge_weight)}, max: {max(edge_weight)}')
    #     edge_indices = edge_indices.t().tolist()
    #     missed_edge_cnt = 0
    #     existed_cnt = 0
    #     new_weights2 = []
    #     new_edges2 = []
    #     for idx, (i, j) in enumerate(edge_indices):  # check if only new_edge is in edge_indices?
    #         if (i, j) in dt:  # the computed edge already existed in the existed_edge_indices.
    #             existed_cnt += 1
    #             continue
    #         # print(f'old edge ({i}, {j}) is not in new_edge_indices')
    #         missed_edge_cnt += 1
    #         new_edges2.append((i, j))
    #         new_weights2.append(edge_weight[idx])  # is it too large
    #         dt.add((i, j))
    #
    #     # merge edge indices and weights
    #     edge_indices = existed_edge_indices.tolist() + new_edges2
    #     edge_weight = existed_weights + new_weights2
    #     print(f'missed edges cnt {missed_edge_cnt}, existed_cnt: {existed_cnt}, edge_indices {len(edge_indices)}')
    #     edge_indices = torch.tensor(edge_indices, dtype=torch.long).t()
    #     edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    # return edge_indices, edge_weight


def plot_data(X, y, train_mask, gen_size, X_test=None, y_test=None, train_info={}):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    X_ = X[train_mask]
    y_ = y[train_mask]
    X_2d = pca.fit_transform(X_)

    X_train_2d = X_2d[:-gen_size]
    X_gen_2d = X_2d[-gen_size:]
    y_train = y_[:-gen_size]
    y_gen = y_[-gen_size:]

    # Plot training data
    for label in np.unique(y_train):
        plt.scatter(
            X_train_2d[y_train == label, 0],
            X_train_2d[y_train == label, 1],
            label=f'Train Class {label}',
            marker='o',
            alpha=0.7
        )

    # Plot generated data
    for label in np.unique(y_gen):
        plt.scatter(
            X_gen_2d[y_gen == label, 0],
            X_gen_2d[y_gen == label, 1],
            label=f'Generated Class {label}',
            marker='x',
            alpha=0.7
        )

    if X_test is not None:
        # Project test data using the fitted PCA
        X_test_2d = pca.transform(X_test)

        # Plot test data
        for label in np.unique(y_test):
            plt.scatter(
                X_test_2d[y_test == label, 0],
                X_test_2d[y_test == label, 1],
                label=f'Test Class {label}',
                marker='s',
                alpha=0.7
            )

    client_id = train_info['client_id']
    server_epoch = train_info['server_epoch']

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'epoch:{server_epoch}, client_{client_id}')
    plt.legend(fontsize='small')
    # plt.grid(True)
    fig_file = f'{in_dir}/plots/client_{client_id}/epoch_{server_epoch}.png'
    os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=100)
    # plt.show()
    plt.clf()


def early_stopping(model, X_val, y_val, epoch, pre_val_loss, val_cnt, criterion, patience=10, best={}):
    # Validation phase
    model.eval()
    val_loss = 0.0
    stop_training = False
    with torch.no_grad():
        if y_val is not None:  # is not graph data
            outputs_ = model(X_val)
            loss_ = criterion(outputs_, y_val)
            accuracy = accuracy_score(y_val, np.argmax(outputs_, axis=1))
        else:
            data = X_val  # here must be graph data
            # outputs_ = model(data)
            z = model.encoder(data.x, data.edge_index, data.edge_weight)
            outputs_ = model.predictor(z, data.edge_index, data.edge_weight)
            _, predicted_labels = torch.max(outputs_, dim=1)
            # Loss calculation: Only for labeled nodes
            loss_ = criterion(outputs_[data.val_mask], data.y[data.val_mask])

            accuracy = accuracy_score(data.y[data.val_mask].tolist(), predicted_labels[data.val_mask].tolist())
            # print(f"epoch: {epoch} Accuracy on val data: {accuracy * 100:.2f}%")
        val_loss += loss_

    best['accs'].append(accuracy)

    if best['accuracy'] < accuracy:
        best['model'] = model.state_dict()
        best['epoch'] = epoch
        best['val_loss'] = val_loss
        best['accuracy'] = accuracy

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


def train_gnn(local_gnn, global_cvae, global_lp, global_gnn, local_data, train_info={}):
    """
        1. Use vaes to generated data for each class
        2. Use the generated data + local data to train local gnn with initial parameters of global_gnn
        3. Send local gnn'parameters to server.
    Args:
        local_gnn:
        cvae:
        global_gnn:
        local_data:

    Returns:

    """
    local_gnn = local_gnn.to(device)
    local_gnn.load_state_dict(global_gnn.state_dict())  # Initialize client_gm with the parameters of global_model
    optimizer = optim.Adam(local_gnn.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()  # mean
    local_gnn.train()  #

    # local data
    train_mask = local_data['train_mask']
    y = local_data['y'][train_mask]  # we assume on a tiny labeled data in the local dat
    size = len(y.tolist())
    print(f'local data size: {len(train_mask)}, y: {size}, label_rate: {size / len(train_mask)}')
    ct = collections.Counter(y.tolist())
    print(f'y: {ct.items()}')
    print(f"len(ct.keys()):{len(ct.keys())} =? len(LABELs): {len(LABELs)}")
    # for debug
    if len(ct.keys()) < len(LABELs):
        max_size = max(ct.values())
        max_size = int(
            max_size * 0.5)  # for each class, only generate 10% percent data to save computational resources.
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
        print(f'we need to generate class sizes:{sizes}')

        # generated new data
        data = gen_data(global_cvae, sizes)
        train_info['generated_size'] = sum(sizes.values())
        # append the generated data to the end of local X and Y, not the end of train set
        data = merge_data(data, local_data)

        # plot_data(data['X'], data['y'], train_size=None)

        # generate new edges
        features = data['X']
        labels = data['y']
        train_info['generated_size'] = sum(sizes.values())
        generated_size = sum(sizes.values())
    else:
        sizes = {}
        labeled_cnt = ct
        features = local_data['X']
        labels = local_data['y']
        data = {}
        train_info['generated_size'] = sum(sizes.values())
        generated_size = sum(sizes.values())
    train_info['threshold'] = None
    existed_edge_indices = local_data['edge_indices']
    local_size = len(local_data['y'])
    edge_indices, edge_weight = gen_edges(features, local_size, global_gnn, existed_edge_indices,
                                          edge_method=None, generated_size=generated_size, local_data=local_data,
                                          train_info=train_info)  # will update threshold
    if edge_weight.shape[0] > 0:
        print(f"edges.shape {edge_indices.shape}, edge_weight min:{min(edge_weight.tolist())}, "
              f"max:{max(edge_weight.tolist())}")

    # Define train, val, and test masks
    # generated_data_indices = list(range(len(train_mask), len(labels), 1))  # append the generated data
    train_mask = torch.cat([train_mask, torch.tensor([True] * (sum(sizes.values())), dtype=torch.bool)])
    val_mask = torch.cat([local_data['val_mask'], torch.tensor([False] * (sum(sizes.values())), dtype=torch.bool)])
    test_mask = torch.cat([local_data['test_mask'], torch.tensor([False] * (sum(sizes.values())), dtype=torch.bool)])
    # Get indices of y
    y_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    indices = torch.tensor(y_indices).to(device)  # total labeled data
    new_y = labels[indices]
    print(labeled_cnt.items(), flush=True)
    s = sum(labeled_cnt.values())
    labeled_classes_weights = {k: s / v for k, v in labeled_cnt.items()}
    s2 = sum(labeled_classes_weights.values())
    labeled_classes_weights = {k: w / s2 for k, w in labeled_classes_weights.items()}  # normalize weights
    data['labeled_classes_weights'] = labeled_classes_weights
    print('new_y', collections.Counter(new_y.tolist()), ', old_y:', ct.items(),
          '\nlabeled_classes_weights', {k: float(f"{v:.2f}") for k, v in labeled_classes_weights.items()})

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
    # train_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    # test_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    # train_mask[indices] = True
    # test_mask[~train_mask] = True
    # val_mask = torch.tensor([False, False, True, False], dtype=torch.bool)
    # test_mask = torch.tensor([False, False, False, True], dtype=torch.bool)
    graph_data = Data(x=node_features, edge_index=edge_indices, edge_weight=edge_weight,
                      y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    # only train smaller model
    epochs_client = 500
    losses = []
    val_losses = []
    best = {'accuracy': -1.0, 'accs': []}
    val_cnt = 0
    pre_val_loss = 0
    # here, you need make sure weight aligned with class order.
    class_weight = torch.tensor(list(data['labeled_classes_weights'].values()), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs_client):
        # epoch_model_loss = 0
        # _model_loss, _model_distill_loss = 0, 0
        # epoch_vae_loss = 0
        # _vae_recon_loss, _vae_kl_loss = 0, 0
        graph_data.to(device)
        # data_size, data_dim = graph_data.x.shape
        # your local personal model
        z = local_gnn.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
        outputs = local_gnn.predictor(z, graph_data.edge_index, graph_data.edge_weight)
        # Loss calculation: Only for labeled nodes
        model_loss = criterion(outputs[graph_data.train_mask], graph_data.y[graph_data.train_mask])

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
        if epoch % 100 == 0:
            print(f"train_gnn epoch: {epoch}, local_gnn loss: {model_loss.item():.4f}")
        losses.append(model_loss.item())

        # X_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
        val_loss, pre_val_loss, val_cnt, stop_training = early_stopping(local_gnn, graph_data, None, epoch,
                                                                        pre_val_loss, val_cnt, criterion, patience=20,
                                                                        best=best)
        val_losses.append(val_loss.item())
        if stop_training:
            local_gnn.stop_training = True
            print(f'Early Stopping. Epoch: {epoch}, Loss: {model_loss:.4f}')
            break

    train_info['gnn'] = {'graph_data': graph_data, "losses": losses}
    print('best epoch: ', best['epoch'], ' best accuracy: ', best['accuracy'])
    local_gnn.load_state_dict(best['model'])

    show = False
    if show:
        import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, label='Training Loss', marker='o')
        plt.plot(range(len(val_losses)), val_losses, label='Validating Loss', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        best_acc = best['accuracy']
        epoch = best['epoch']
        plt.title(f'Training and Validation Loss Over Epochs. Best_Acc: {best_acc} at Epoch: {epoch}')
        plt.legend()
        # plt.grid()
        plt.show()

        accs_val = best['accs']
        plt.plot(range(len(accs_val)), accs_val, label='', marker='')
        plt.plot(range(len(accs_val)), accs_val, label='Validating Accuracy', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        best_acc = best['accuracy']
        epoch = best['epoch']
        plt.title(f'Training and Validation Acc Over Epochs. Best_Acc: {best_acc} at Epoch: {epoch}')
        plt.legend()
        # plt.grid()
        plt.show()


def aggregate_cvaes(vaes, locals_info, global_cvae):
    print('*aggregate cvaes...')
    client_parameters_list = [local_cvae.state_dict() for client_i, local_cvae in vaes.items()]
    # aggregate(client_parameters_list, global_cvae)
    aggregate_method = 'parameter'
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


def aggregate_lps(lps, global_lp):
    print('*aggregate lp...')
    client_parameters_list = [local_gnn.state_dict() for client_i, local_gnn in lps.items()]
    aggregate_with_attention(client_parameters_list, global_lp, device)  # update global_gnn inplace


def aggregate_gnns(gnns, global_gnn):
    print('*aggregate gnn...')
    client_parameters_list = [local_gnn.state_dict() for client_i, local_gnn in gnns.items()]
    aggregate_with_attention(client_parameters_list, global_gnn, device)  # update global_gnn inplace


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


import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


#
# class GNNLinkPredictor(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(GNNLinkPredictor, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.lin = Linear(2 * hidden_channels, hidden_channels)  # edge/link prediction
#         self.lin2 = Linear(hidden_channels, 1)  # edge/link prediction
#
#     def forward(self, x, edge_index):
#         # GNN layers
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index).relu()
#
#         return x
#
#     def decode(self, z, edge_index):
#         # Compute pairwise embeddings
#         # edge_index.shape = (2, m)
#         src, dst = edge_index[0, :], edge_index[1, :]
#         z_src = z[src]
#         z_dst = z[dst]
#         edge_scores = self.lin(torch.cat([z_src, z_dst], dim=1)).relu()
#         edge_scores = self.lin2(edge_scores)
#         return edge_scores


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


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Adding a third layer

        self.conv41 = GCNConv(hidden_dim, hidden_dim)  #
        self.conv42 = GCNConv(hidden_dim, output_dim)  # Output layer

        self.lin = Linear(2 * hidden_dim, hidden_dim)  # edge/link prediction
        self.lin2 = Linear(hidden_dim, 1)  # edge/link prediction

    # def forward(self, data):
    #     x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
    #
    #     no_edge_weight = False
    #     if no_edge_weight:
    #         # no edge_weight is passed to the GCNConv layers
    #         x = F.relu(self.conv1(x, edge_index))
    #         x = F.relu(self.conv2(x, edge_index))
    #         x = F.relu(self.conv3(x, edge_index))  # Additional layer
    #         x = self.conv4(x, edge_index)  # Final output
    #     else:
    #         # Passing edge_weight to the GCNConv layers
    #         x = F.relu(self.conv1(x, edge_index, edge_weight))
    #         x = F.relu(self.conv2(x, edge_index, edge_weight))
    #         x = F.relu(self.conv3(x, edge_index, edge_weight))  # Additional layer
    #         x = self.conv4(x, edge_index, edge_weight)  # Final output
    #
    #     # return F.log_softmax(x, dim=1)
    #     return x

    def encoder(self, x, edge_index, edge_weight=None):
        # Passing edge_weight to the GCNConv layers
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))  # Additional layer

        return x

    def predictor(self, z, edge_index, edge_weight=None):
        z = self.conv41(z, edge_index, edge_weight)
        x = self.conv42(z, edge_index, edge_weight)  # Final output
        return x

    def decoder(self, z, edge_index, edge_weight=None):
        # Compute pairwise embeddings
        # edge_index.shape = (2, m)
        src, dst = edge_index[0, :], edge_index[1, :]
        z_src = z[src]
        z_dst = z[dst]
        edge_scores = self.lin(torch.cat([z_src, z_dst], dim=1)).relu()
        edge_scores = self.lin2(edge_scores)
        return edge_scores


@timer
def evaluate(local_gnn, local_data, device, test_type='test', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.

        client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        client_data_ =  (graph_data_, feature_info, client_data_)
    """
    # After training, the model can make predictions for both labeled and unlabeled nodes
    print(f'***Testing gnn model on test_type:{test_type}...')
    # gnn = local_gnn(input_dim=64, hidden_dim=32, output_dim=10)
    # gnn.load_state_dict(client_result['client_gm'])
    gnn = local_gnn
    gnn = gnn.to(device)

    graph_data = train_info['gnn']['graph_data'].to(device)  # graph data
    gnn.eval()
    with torch.no_grad():
        # output = gnn(graph_data)
        z = gnn.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
        output = gnn.predictor(z, graph_data.edge_index, graph_data.edge_weight)
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
        train_info['labeled_accuracy'] = accuracy
        print(f"Accuracy on labeled data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy
        # print(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        train_info['labeled_cm'] = conf_matrix
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
        train_info['unlabeled_accuracy'] = accuracy
        print(f"Accuracy on unlabeled data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy
        conf_matrix = confusion_matrix(y, y_pred)
        train_info['unlabeled_cm'] = conf_matrix
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


import torch


def find_neighbors(X_test, X_local, X_local_indices, X_test_indices, edge_indices,
                   global_lp, edge_method='cosine', k=5, train_info={}):
    """
    Find the k-nearest neighbors of a new node based on cosine similarity.

    Args:
        new_node_feature (Tensor): Feature vector of the new node.
        node_features (Tensor): Feature matrix of the existing graph (shape: num_nodes x feature_dim).
        k (int): The number of neighbors to connect the new node to.

    Returns:
        torch.Tensor: A tensor containing edge indices (source, target) for the new node's neighbors.
    """
    local_size = len(X_local)  # append X_test to the end of X_local
    existed_edge_indices = X_local_indices
    existed_weights = [1] * existed_edge_indices.size(1)  # existed_edge_indices.shape is 2xN

    # # If current client has classes (0, 1, 2, 3), then predict edges for new nodes (such as, 4, 5, 6)
    threshold = 0.05
    new_nodes = X_test
    # new_node_pairs = torch.combinations(torch.arange(len(new_nodes)), r=2).t()
    # z = global_lp(new_nodes, new_node_pairs)
    # new_probs = global_lp.decode(z, new_node_pairs)
    # new_edges = new_node_pairs.t()[new_probs.flatten() > threshold].t()
    # # adjust new_edges indices
    # new_edges = local_size + new_edges
    # # new_weights = [1] * len(new_edges) # not correct
    # new_weights = [1] * new_edges.shape[1]
    new_edges = X_test_indices
    new_weights = [1] * new_edges.shape[1]

    # Predict edges between new and existing nodes
    existed_nodes = X_local
    cross_pairs = torch.cartesian_prod(torch.arange(0, local_size), torch.arange(len(new_nodes))).t()
    # z = global_lp(existed_nodes, existed_new_pairs)
    features = torch.cat((existed_nodes, new_nodes), dim=0)
    z = global_lp(features, cross_pairs)  # here, we use all train_features.
    cross_probs = global_lp.decode(z, cross_pairs)
    cross_edges = cross_pairs.t()[cross_probs.flatten() > threshold].t()
    # adjust cross_edges indices for new nodes
    new_edges[1, :] = local_size + new_edges[1, :]  # new_edges.shape is 2xN
    cross_weights = [1] * cross_edges.shape[1]

    # Combine all edges
    edge_indices = torch.cat([existed_edge_indices, new_edges, cross_edges], dim=1)
    edge_weights = existed_weights + new_weights + cross_weights
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    return edge_indices, edge_weights

    # Combine the new node feature with the existing node features
    all_node_features = torch.cat([X_local, X_test], dim=0)  # includes the generated data in X_local
    # Convert the node features to NumPy arrays (since scikit-learn's NearestNeighbors works with NumPy arrays)
    all_node_features_np = all_node_features.cpu().numpy()

    # find existed edges in all edge_indices
    # for pairs of X_local, X_test in all edge_indices. Here we don't consider any generated data
    existed_edge_indices = set()  # store X_local (not include the generated data) and X_test indices
    set_edge_indices = set([(i, j) for i, j in edge_indices.tolist()])
    # here, should be X_local_indices (not including the generated data) as we use e_i, e_j
    for i, e_i in enumerate(X_local_indices.tolist()):
        for j, e_j in enumerate(X_test_indices.tolist()):
            if (e_i, e_j) in set_edge_indices:
                existed_edge_indices.add((i, len(X_local) + j))  # new index, including the generated data
            if (e_j, e_i) in set_edge_indices:
                existed_edge_indices.add((len(X_local) + j, i))
    existed_edge_indices = list(existed_edge_indices)
    existed_weights = [1] * len(existed_edge_indices)

    new_X = []
    for x1 in all_node_features_np:
        for x2 in all_node_features_np:
            new_X.append([x1, x2])
    new_X = torch.tensor(np.asarray(new_X), dtype=torch.float)
    outputs = global_lp.forward(new_X[:, 0, :], new_X[:, 1, :])
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    new_edges = []
    edge_weight = []
    for i, x1 in enumerate(all_node_features_np):
        for j, x2 in enumerate(all_node_features_np):
            t = i * len(all_node_features_np) + j
            if y_pred[t] == 1:
                new_edges.append([i, j])
                edge_weight.append(1)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # threshold = train_info['threshold']
    # if edge_method == 'cosine':
    #     similarity_matrix = cosine_similarity(all_node_features_np)  # [-1, 1]
    #     # Set diagonal items to 0
    #     np.fill_diagonal(similarity_matrix, 0)
    #     similarity_matrix = torch.abs(torch.tensor(similarity_matrix, dtype=torch.float32))
    #     # #  # only keep the upper triangle of the matrix and exclude the diagonal entries
    #     # similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
    #     print(f'similarity matrix: {similarity_matrix.shape}')
    #     # Create graph: Each image is a node, edges based on similarity
    #     # threshold = torch.quantile(similarity_matrix, 0.9)  # input tensor is too large()
    #     # Convert the tensor to NumPy array
    #     if threshold is None:
    #         # import scipy.stats as stats
    #         # similarity_matrix_np = similarity_matrix.cpu().numpy()
    #         # # Calculate approximate quantile using scipy
    #         # thresholds = [(v, float(stats.scoreatpercentile(similarity_matrix_np.flatten(), v))) for v in
    #         #               range(0, 100 + 1, 10)]
    #         # print(thresholds)
    #         # per = 99.0
    #         # threshold = stats.scoreatpercentile(similarity_matrix_np.flatten(), per)  # per in [0, 100]
    #         # # train_info['threshold'] = threshold
    #         raise NotImplementedError
    #     else:
    #         per = 99.0
    #     print('threshold', threshold)
    #     # Find indices where similarity exceeds the threshold
    #     new_edges = (torch.abs(similarity_matrix) > threshold).nonzero(
    #         as_tuple=False)  # two dimensional data [source, targets]
    #     print(f"total number of edges: {similarity_matrix.shape}, we only keep {100 - per:.2f}% edges "
    #           f"with edge_indices.shape: {new_edges.shape}")
    #     edge_weight = similarity_matrix[new_edges[:, 0], new_edges[:, 1]]  # one dimensional data
    #     new_edges = new_edges.tolist()
    # else:
    #     # Initialize NearestNeighbors with cosine similarity
    #     knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    #     knn.fit(all_node_features_np)  # Fit the k-NN model with all node features
    #
    #     # Find the k nearest neighbors (including the new node itself, hence k+1)
    #     distances, indices = knn.kneighbors(X_test.cpu().numpy())
    #
    #     # Prepare edge index pairs (source_node, target_node)
    #     # The new node is the source node, and the neighbors are the target nodes
    #     new_edges = []
    #     new_distances = []
    #     start_idx = len(X_local)  # includes the generated data
    #     for i in range(len(X_test)):
    #         for j, neig_idx in enumerate(indices[i]):
    #             if start_idx + i == neig_idx: continue
    #             new_edges.append([start_idx + i, neig_idx])  # New node (source) -> Neighbor (target)
    #             new_distances.append(distances[i, j])
    #
    #     # one dimensional data
    #     edge_weight = (2 - torch.tensor(new_distances,
    #                                     dtype=torch.float32)) / 2  # dists is [0, 2], after this, values is [0,1]
    # # edge_weight = torch.sparse_coo_tensor(edge_indices, values, size=(num_nodes, num_nodes)).to_dense()
    # # edge_weight = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(0.0))

    if existed_edge_indices is None:
        pass
    else:
        new_edges2 = []
        edge_weight2 = []
        edge_weight = edge_weight.tolist()
        set_existed_edges = set([(i, j) for i, j in existed_edge_indices])
        missed_cnt = 0
        existed_cnt = 0
        for idx, (i, j) in enumerate(new_edges):  # X_local (including generated data) -> X_test
            a = i
            b = j
            if (a, b) in set_existed_edges:
                existed_cnt += 1
                continue
            missed_cnt += 1
            new_edges2.append([a, b])
            edge_weight2.append(edge_weight[idx])
            set_existed_edges.add((a, b))

        if len(new_edges2) > 0:
            print(f'*** in {len(new_edges)} new edges, where missed {len(new_edges2)} edges and existed {existed_cnt} '
                  f'edges in existed_edge_indices: {len(existed_edge_indices)}')
            new_edges = existed_edge_indices + new_edges2
            edge_weight = existed_weights + edge_weight2

    return torch.tensor(new_edges, dtype=torch.long).t(), torch.tensor(edge_weight, dtype=torch.float)


#
# @timer
# def find_edges(X_local_indices, X_test_indices, edge_indices):
#     # for pairs of X_local, X_test in all edge_indices. Here we don't consider any generated data
#     new_edges = set()  # store X_local (not include the generated data) and X_test indices
#     set_edge_indices = set([(i, j) for i, j in edge_indices.tolist()])
#     for i, e_i in enumerate(X_local_indices.tolist()):
#         for j, e_j in enumerate(X_test_indices.tolist()):
#             if (e_i, e_j) in set_edge_indices:
#                 new_edges.add((e_i, e_j))     # original index in all edge_indices
#             if (e_j, e_i) in set_edge_indices:
#                 new_edges.add((e_j, e_i))
#
#     return torch.tensor(list(new_edges))

def get_cross_edge_info(cross_edges, local_size, generated_size):
    print(f'***cross_edges: {cross_edges.shape}', ' local size', local_size, ' generated_size', generated_size)
    num_gen_test = 0  # local_size = train + generated
    num_train_test = 0  # local_size = train + generated
    num_train_gen = 0  # local_size = train + generated
    for i, j in cross_edges:  # i is the index of local data, # j is the index of test data
        if i < local_size - generated_size:  # local train
            num_train_test += 1
        elif local_size - generated_size < i < local_size:
            num_gen_test += 1
        else:
            print(f'unknown: {i}, {j}')

    print(f'number of edges between local train and generated data: {num_train_gen}')
    print(f'number of edges between local train and test data: {num_train_test}')
    print(f'number of edges between local generated and test data: {num_gen_test}')


def extract_edges(orig_indices, edges_set):
    edges = []
    for i, orig_i in enumerate(orig_indices):
        for j, orig_j in enumerate(orig_indices):
            e = (orig_i, orig_j)
            if e in edges_set:
                edges.append([i, j])

    return np.asarray(edges)


def check_edges(edges, edges2):
    print(f'len(gen_cross_edges): {len(edges)}, len(edges_ground_truth): {len(edges2)}')
    edges_set = set([(a, b) for a, b in edges])
    edges2_set = set([(a, b) for a, b in edges2])
    print(f'len(gen_cross_edges_set): {len(edges_set)}, len(edges_ground_truth_set): {len(edges2_set)}')
    diff = edges_set - edges2_set
    tmp = list(diff)[:10] if len(diff) > 10 else diff
    print(f'gen_cross_edges-edges_ground_truth = ({len(diff)}): {tmp}')
    diff2 = edges2_set - edges_set
    tmp2 = list(diff2)[:10] if len(diff2) > 10 else diff2
    print(f'edges_ground_truth-gen_cross_edges = ({len(diff2)}): {tmp2}')
    print(f'gen_cross_edges==edges_ground_truth: {edges_set == edges2_set}')


def gen_test_edges(graph_data, X_test, y_test, test_edges, global_lp, generated_size, local_data, train_info):
    X_local = graph_data.x
    y_local = graph_data.y
    local_size = len(X_local)
    train_mask = graph_data.train_mask

    # plot_data(X_local.numpy(), y_local.numpy(), train_mask.numpy(), generated_size, X_test.numpy(), y_test.numpy(), train_info)

    debug = False
    if debug:
        features = torch.cat((X_local, X_test), dim=0)
        labels = torch.cat((y_local, y_test), dim=0)
        start_idx = len(X_local)
        # features = X_test
        # labels = y_test
        # start_idx = 0
        # Combine all edges
        # edge_indices = test_edges.t()
        edge_indices = torch.combinations(torch.arange(len(features)), r=2).t()
        edge_weights = torch.ones((edge_indices.shape[1],))
        print(f'total edges between all nodes: {edge_indices.shape}')
        return features, labels, edge_indices, edge_weights, start_idx

    existed_edge_indices = graph_data.edge_index
    existed_weights = graph_data.edge_weight.tolist()
    print(f'edges between existed nodes ({len(X_local)}): {existed_edge_indices.shape}')

    # # If current client has classes (0, 1, 2, 3), then predict edges for new nodes (such as, 4, 5, 6)
    edge_threshold = train_info['edge_threshold']
    new_nodes = X_test
    # new_node_pairs = torch.combinations(torch.arange(len(new_nodes)), r=2).t()
    # z = global_lp(new_nodes, new_node_pairs)
    # new_probs = global_lp.decode(z, new_node_pairs)
    # new_edges = new_node_pairs.t()[new_probs.flatten() > threshold].t()
    # # adjust new_edges indices
    # new_edges = local_size + new_edges
    # # new_weights = [1] * len(new_edges) # not correct
    # new_weights = [1] * new_edges.shape[1]
    new_edges = test_edges.t()
    # adjust cross_edges indices for new nodes
    new_edges = local_size + new_edges  # new_edges.shape is 2xN
    new_weights = [1] * new_edges.shape[1]
    print(f'new edges between new nodes ({len(new_nodes)}): {new_edges.shape}')

    # Predict edges between new and existing nodes
    existed_nodes = X_local
    cross_pairs = torch.cartesian_prod(torch.arange(0, local_size),
                                       torch.arange(local_size, local_size + len(new_nodes))).t()
    # z = global_lp(existed_nodes, existed_new_pairs)
    features = torch.cat((existed_nodes, new_nodes), dim=0)
    labels = torch.cat((y_local, y_test), dim=0)
    # z = global_lp(features, cross_pairs)  # here, we use all train_features.
    # Assuming `model` is your trained GNN model
    global_lp.eval()  # Set model to evaluation mode
    # Compute embeddings for existing and test nodes
    z_existed = global_lp.encoder(existed_nodes, existed_edge_indices)  # Embeddings for existing nodes
    z_test = global_lp.encoder(new_nodes, edge_index=new_edges - local_size)  # Embeddings for test nodes (no edges)
    z = torch.cat([z_existed, z_test], dim=0)
    cross_probs = F.sigmoid(global_lp.decoder(z, cross_pairs))
    print_histgram(cross_probs.detach().cpu().numpy())
    cross_edges = cross_pairs.t()[cross_probs.flatten() > edge_threshold].t()
    # adjust cross_edges indices for new nodes
    # cross_edges[1, :] = local_size + cross_edges[1, :]  # new_edges.shape is 2xN
    cross_edges = torch.zeros((2, 0), dtype=torch.long)
    cross_weights = [1] * cross_edges.shape[1]
    print(f'cross edges between existed nodes ({len(existed_nodes)}) and new nodes ({len(new_nodes)}): '
          f'{cross_edges.shape}')

    debug = True
    if debug:
        get_cross_edge_info(cross_edges.t().numpy(), local_size, generated_size)
        # get ground truth cross edges between local data and test data from original edges
        original_local_indices = local_data['original_indices']
        all_data = local_data['all_data']
        shared_test_mask = all_data['test_mask']
        original_test_indices = all_data['indices'][shared_test_mask]
        edges_set = set([(i, j) for i, j in all_data['edge_indices'].numpy().T])
        cross_edges2 = extract_edges(np.concatenate([original_local_indices, original_test_indices], axis=0), edges_set)
        check_edges(cross_edges.t().numpy(), cross_edges2)

    # Combine all edges
    edge_indices = torch.cat([existed_edge_indices, new_edges, cross_edges], dim=1)
    edge_weights = existed_weights + new_weights + cross_weights
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    print(f'total edges between all nodes: {edge_indices.shape}')

    start_idx = len(y_local)
    return features, labels, edge_indices, edge_weights, start_idx


@timer
def evaluate_shared_test(local_gnn, local_data, device, global_lp,
                         test_type='shared_test_data', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    # After training, the model can make predictions for both labeled and unlabeled nodes
    print(f'***Testing gnn model on test_type:{test_type}...')
    all_data = local_data['all_data']
    all_indices = all_data['indices']
    # edge_indices = all_data['edge_indices'].to(device)  # all edge_indices
    X, Y = all_data['X'].to(device), all_data['y'].to(device)

    # Shared test data
    shared_test_mask = all_data['test_mask'].to(device)
    X_test = X[shared_test_mask].to(device)
    y_test = Y[shared_test_mask].to(device)
    # X_test_indices = all_indices[shared_test_mask].to(device)
    print(f'X_test: {X_test.size()}, {collections.Counter(y_test.tolist())}')
    edge_indices_test = all_data['edge_indices_test'].to(device)
    if True:  # evaluate on new data
        # Add new_data to the node feature matrix (if you're adding a new node)
        # This could involve concatenating the new feature to the existing node features
        graph_data = train_info['gnn']['graph_data'].to(device)
        generated_size = train_info['generated_size']
        X_local = graph_data.x  # includes all local train (+ generated data by cvae), val and test
        y = graph_data.y
        features, labels, edges, edge_weight, local_size = gen_test_edges(graph_data, X_test, y_test, edge_indices_test,
                                                                          global_lp, generated_size, local_data,
                                                                          train_info)
        # X_local_indices = local_data['indices'].to(device)  # local indices not include the generated data.
        # # local_data_size_without_generated = len(X_local_indices)
        # # generated_data_size = len(y) - local_data_size_without_generated
        # local_size = len(y)
        # # Add edges to the graph (example: add an edge from the new node to its neighbors)
        # # Assuming you have a function `find_neighbors` to determine which nodes to connect
        # # new_node_feature =  torch.tensor([features], dtype=torch.float32)  # New node's feature vector
        # new_edges, new_weight = find_neighbors(X_test, X_local, X_local_indices, X_test_indices, edge_indices,
        #                                        global_lp, edge_method='cosine', k=5, train_info=train_info)
        # (source, target) format for the new edges
        # new_edges = new_edges.to(device)
        # new_weight = new_weight.to(device)
        # # print(f"device: {graph_data.edge_index.device}, {graph_data.edge_weight.device}, {new_edges.device}, {new_weight.device}")
        # edges = torch.cat([graph_data.edge_index, new_edges], dim=1)  # Add new edges into train set
        # edge_weight = torch.cat([graph_data.edge_weight, new_weight])  # Add new edges into train set
        # features = torch.cat((X_local, X_test), dim=0)  # Adding to the existing node features
        # labels = torch.cat((y, y_test)).to(device)
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

    # evaluate the data
    gnn = local_gnn
    gnn.to(device)
    gnn.eval()
    with torch.no_grad():
        # output = gnn(graph_data)
        z = gnn.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
        output = gnn.predictor(z, graph_data.edge_index, graph_data.edge_weight)
        _, predicted_labels = torch.max(output, dim=1)

        # # here we use new edges (based on local data and test set), so the performance is different from only use
        # # local data
        # # only on local data
        # print('Evaluate on local data...')
        # predicted_labels_ = predicted_labels[:local_size]
        # # Calculate accuracy for the labeled data
        # # num_classes = 10
        # # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        # # print(f'labeled_indices {len(labeled_indices)}')
        # true_labels = graph_data.y[:local_size]
        #
        # # predicted_labels = predicted_labels[graph_data.train_mask]
        # # true_labels = true_labels[graph_data.train_mask]
        # # predicted_labels = predicted_labels[graph_data.test_mask]
        # # true_labels = true_labels[graph_data.test_mask]
        #
        # y = true_labels.cpu().numpy()
        # y_pred = predicted_labels_.cpu().numpy()
        # accuracy = accuracy_score(y, y_pred)
        # print(f"Accuracy on local data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # # if 'all' in test_type:
        # #     client_result['labeled_accuracy_all'] = accuracy
        # # else:
        # #     client_result['labeled_accuracy'] = accuracy
        #
        # # Compute the confusion matrix
        # conf_matrix = confusion_matrix(y, y_pred)
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # # if 'all' in test_type:
        # #     client_result['labeled_cm_all'] = conf_matrix
        # # else:
        # #     client_result['labeled_cm'] = conf_matrix

        # only on test set
        print('Evaluate on shared test data...')
        predicted_labels_ = predicted_labels[local_size:]
        # Calculate accuracy for the labeled data
        # num_classes = 10
        # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
        # print(f'labeled_indices {len(labeled_indices)}')
        true_labels = graph_data.y[local_size:]

        # predicted_labels = predicted_labels[graph_data.train_mask]
        # true_labels = true_labels[graph_data.train_mask]
        # predicted_labels = predicted_labels[graph_data.test_mask]
        # true_labels = true_labels[graph_data.test_mask]

        y = true_labels.cpu().numpy()
        y_pred = predicted_labels_.cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        train_info['shared_accuracy'] = accuracy
        print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        # if 'all' in test_type:
        #     client_result['labeled_accuracy_all'] = accuracy
        # else:
        #     client_result['labeled_accuracy'] = accuracy

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        train_info['shared_cm'] = conf_matrix
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
            local_gnn = client['gnn']
            print(f'\t*local cvae:', local_cvae.keys(), f' server_epoch: {s}')
            losses_ = [float(f"{v:.2f}") for v in local_cvae['losses']]
            print(f'\t\tlocal cvae:', losses_)
            # print('\t*local gnn:', [f"{v:.2f}" for v in local_gnn['losses']])
            # labeled_acc = client['labeled_accuracy']
            # unlabeled_acc = client['unlabeled_accuracy']
            # shared_acc = client['shared_accuracy']
            # print(f'\t\tlabeled_acc:{labeled_acc:.2f}, unlabeled_acc:{unlabeled_acc:.2f}, shared_acc:{shared_acc:.2f}')

    ncols = 2
    nrows, r = divmod(c, ncols)
    nrows = nrows if r == 0 else nrows + 1
    fig, axes = plt.subplots(nrows, ncols)
    for c in range(num_clients):
        i, j = divmod(c, ncols)
        print(f"\nclient {c}")
        labeled_accs = []
        unlabeled_accs = []
        shared_accs = []
        for s in range(num_server_epoches):
            client = histories[s][c]
            # local_cvae = client['cvae']
            # local_gnn = client['gnn']
            # print(f'\t*local cvae:', local_cvae.keys(), f' server_epoch: {s}')
            # losses_ = [float(f"{v:.2f}") for v in local_cvae['losses']]
            # print(f'\t\tlocal cvae:', losses_)
            # # print('\t*local gnn:', [f"{v:.2f}" for v in local_gnn['losses']])
            labeled_acc = client['labeled_accuracy']
            unlabeled_acc = client['unlabeled_accuracy']
            shared_acc = client['shared_accuracy']
            labeled_accs.append(labeled_acc)
            unlabeled_accs.append(unlabeled_acc)
            shared_accs.append(shared_acc)
            print(
                f'\t\tEpoch: {s}, labeled_acc:{labeled_acc:.2f}, unlabeled_acc:{unlabeled_acc:.2f}, shared_acc:{shared_acc:.2f}')

        # Training and validation loss on the first subplot
        axes[i, j].plot(range(len(labeled_accs)), labeled_accs, label='labeled_acc', marker='o')
        axes[i, j].plot(range(len(unlabeled_accs)), unlabeled_accs, label='unlabeled_acc', marker='+')
        axes[i, j].plot(range(len(shared_accs)), shared_accs, label='shared_acc', marker='s')
        axes[i, j].set_xlabel('Server Epochs')
        axes[i, j].set_ylabel('Accuracy')
        axes[i, j].set_title(f'Client_{c}')
        axes[i, j].legend(fontsize='small')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'{in_dir}/accuracy.png', dpi=300)
    plt.show()


def client_process(c, epoch, global_cvae, global_gnn, input_dim, num_classes, label_rate, in_dir, prefix, device):
    """
    Function to be executed in a separate process for each client.
    """
    print(f"\n\n***server_epoch:{epoch}, client_{c} ...")
    l = c  # we should have 'num_clients = num_labels'
    train_info = {"cvae": {}, "gnn": {}}

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

    # Train GNN
    print('train_gnn...')
    local_gnn = GNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
    train_gnn(local_gnn, global_cvae, global_gnn, local_data, train_info)

    # Evaluate GNN
    print('evaluate_gnn...')
    evaluate(local_gnn, None, device, test_type='Testing on client data', client_id=c, train_info=train_info)
    evaluate_shared_test(local_gnn, local_data['shared_test_data'], device, \
                         test_type='Testing on shared test data', client_id=c, train_info=train_info)

    return c, local_cvae, local_info, local_gnn, train_info


@timer
def main(in_dir, input_dim=16):
    num_classes = len(LABELs)

    print(f'in_dir: {in_dir}, '
          f'input_dim: {input_dim}, '
          f'num_clients: {num_clients}, '
          f'num_classes: {num_classes}, where classes: {LABELs}')

    prefix = f'r_{label_rate}'
    # Generate local data for each client first
    for c in range(num_clients):
        print(f'\nGenerate local data for client_{c}...')
        gen_local_data(client_data_file=f'{in_dir}/c_{c}-{prefix}-data.pth', client_id=c,
                       label_rate=label_rate)

    global_cvae = CVAE(input_dim=input_dim, hidden_dim=32, latent_dim=5, num_classes=num_classes)
    global_gnn = GNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)

    debug = True
    if debug:
        histories = []
        for epoch in range(server_epochs):
            # update clients
            cvaes = {}
            gnns = {}
            locals_info = {}  # used in CVAE
            history = {}
            for c in range(num_clients):
                print(f"\n\n***server_epoch:{epoch}, client_{c} ...")
                train_info = {"cvae": {}, "gnn": {}, 'client_id': c, 'server_epoch': epoch}  # might be used in server
                # local_data = features_info[l]
                local_data = gen_local_data(client_data_file=f'{in_dir}/c_{c}-{prefix}-data.pth', client_id=c,
                                            label_rate=label_rate)
                label_cnts = collections.Counter(local_data['y'].tolist())
                locals_info[c] = {'label_cnts': label_cnts}
                print(f'client_{c} data:', label_cnts)

                # Use to generate nodes
                local_cvae = CVAE(input_dim=input_dim, hidden_dim=32, latent_dim=5, num_classes=num_classes)
                print('Train CVAE...')
                train_cvae(local_cvae, global_cvae, local_data, train_info)
                cvaes[c] = local_cvae

                print('Train GNN...')
                local_gnn = GNN(input_dim=input_dim, hidden_dim=32, output_dim=num_classes)
                train_gnn(local_gnn, global_cvae, global_gnn, global_gnn, local_data, train_info)
                gnns[c] = local_gnn

                print('Evaluate GNN...')
                evaluate(local_gnn, None, device,
                         test_type='Testing on client data', client_id=c, train_info=train_info)
                evaluate_shared_test(local_gnn, local_data, device, global_gnn,
                                     test_type='Testing on shared test data', client_id=c, train_info=train_info)
                history[c] = train_info

            print('Server aggregation...')
            aggregate_cvaes(cvaes, locals_info, global_cvae)
            aggregate_gnns(gnns, global_gnn)

            histories.append(history)
    else:
        import multiprocessing
        # Set start method to 'spawn' for CUDA compatibility
        multiprocessing.set_start_method('spawn', force=True)

        histories = []
        for epoch in range(server_epochs):

            cvaes = {}
            locals_info = {}
            gnns = {}
            history = {}
            with mp.Pool(processes=num_clients) as pool:
                # Use apply_async or map to execute client_process concurrently and get results
                results = [pool.apply_async(client_process, args=(
                    c, epoch, global_cvae, global_gnn, input_dim, num_classes, label_rate, in_dir, prefix, device))
                           for c in range(num_clients)]
                # Wait for all results to finish and collect them
                results = [r.get() for r in results]  # return c, local_cvae, local_gnn, train_info
                for r in results:
                    c, cvae, local_info, gnn, train_info = r  # it will run when you call r.get().
                    cvaes[c] = cvae
                    locals_info[c] = local_info
                    gnns[c] = gnn
                    history[c] = train_info

            # Server aggregation
            aggregate_cvaes(cvaes, locals_info, global_cvae)
            aggregate_gnns(gnns, global_gnn)
            # Collect histories
            histories.append(history)

    prefix += f'-n_{server_epochs}'
    history_file = f'{in_dir}/histories_cvae_{prefix}.pth'
    print(f'saving histories to {history_file}')
    # with open(history_file, 'wb') as f:
    #     pickle.dump(histories, f)
    torch.save(histories, history_file)

    print_histories(histories)


if __name__ == '__main__':
    # in_dir = 'fl/mnist'
    # input_dim = 16
    # LABELs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
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

    in_dir = 'fl/cora'
    input_dim = 1433
    LABELs = {0, 1, 2, 3, 4, 5, 6}
    num_clients = 4
    main(in_dir, input_dim)
    # history_file = f'{in_dir}/histories_cvae.pkl'
    # with open(history_file, 'rb') as f:
    #     histories = pickle.load(f)
    # print_histories(histories)
