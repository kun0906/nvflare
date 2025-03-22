import collections
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, Subset

# import exp_weighted_mean
from . import robust_aggregation
from .utils import timer


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # torch.tensor(X, dtype=torch.float32)
        self.y = y  # torch.tensor(y, dtype=torch.long)  # For classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # From 1 channel to 16 channels
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv21 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Adjust the dimensions after the convolution layers
        self.fc2 = nn.Linear(512, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # # self.sigmoid = nn.Sigmoid()
        #
        # self.transform = nn.Sequential(
        #     nn.Linear(28 * 28 + num_classes, 784),
        #     nn.LeakyReLU(0.2),
        # )

    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)  # (N, 1, 28, 28)

        # Ensure input has the correct shape (batch_size, 1, 28, 28)
        x = self.leaky_relu(self.conv1(x))
        # x = self.leaky_relu(self.conv11(x))

        x = self.leaky_relu(self.conv2(x))
        # x = self.leaky_relu(self.conv21(x))

        x = self.leaky_relu(self.conv3(x))
        # x = self.leaky_relu(self.conv31(x))

        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.leaky_relu(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=1)
        x = self.fc2(x)
        return x



class FNN(nn.Module):
    def __init__(self, input_dim=768, num_classes=10):
        super(FNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # From 1 channel to 16 channels
        # self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        #
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        # self.conv21 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        #
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #
        # self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Adjust the dimensions after the convolution layers
        # self.fc2 = nn.Linear(512, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # # self.sigmoid = nn.Sigmoid()
        #
        # self.transform = nn.Sequential(
        #     nn.Linear(28 * 28 + num_classes, 784),
        #     nn.LeakyReLU(0.2),
        # )

        self.fc11 = nn.Linear(input_dim, 32)
        self.fc21 = nn.Linear(32, 16)
        self.fc22 = nn.Linear(16, 8)
        self.fc33 = nn.Linear(8, num_classes)

    def forward(self, x):
        model_type = 'mlp'

        x = self.leaky_relu(self.fc11(x))
        x = self.leaky_relu(self.fc21(x))
        x = self.leaky_relu(self.fc22(x))
        x = self.fc33(x)

        return x


@timer
def train_cnn(local_cnn, global_cnn, local_data, train_info={}):
    """
        1. Use gans to generated data for each class
        2. Use the generated data + local data to train local cnn with initial parameters of global_cnn
        3. Send local cnn'parameters to server.
    Args:
        local_cnn:
        gan:
        global_cnn:
        local_data:

    Returns:

    """
    DEVICE = train_info['DEVICE']
    # X, y, train_mask, val_mask, test_mask = train_info['cnn']['data']
    X, y = local_data['X'], local_data['y']
    train_mask, val_mask, test_mask = local_data['train_mask'], local_data['val_mask'], local_data['test_mask']
    tmp = X.cpu().numpy().flatten()
    print(f'X: min: {min(tmp)}, max: {max(tmp)}')

    X_train, y_train = X[train_mask], y[train_mask]
    print('Compute classes weights...')
    # Get indices of y
    y_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    indices = torch.tensor(y_indices).to(DEVICE)  # total labeled data
    new_y = y[indices]
    labeled_cnt = collections.Counter(new_y.tolist())
    labeled_cnt = dict(sorted(labeled_cnt.items(), key=lambda x: x[0], reverse=False))
    print('labeled_y: ', labeled_cnt.items(), flush=True)
    s = sum(labeled_cnt.values())
    labeled_classes_weights = {k: s / v for k, v in labeled_cnt.items()}
    s2 = sum(labeled_classes_weights.values())
    labeled_classes_weights = {k: w / s2 for k, w in labeled_classes_weights.items()}  # normalize weights
    # data['labeled_classes_weights'] = labeled_classes_weights
    # print('labeled_y (train)', collections.Counter(new_y.tolist()), ', old_y:', ct.items(),
    #       f'\nlabeled_classes_weights ({sum(labeled_classes_weights.values())})',
    #       {k: float(f"{v:.2f}") for k, v in labeled_classes_weights.items()})

    # only train smaller model
    CLIENT_EPOCHS = train_info['CFG'].CLIENT_EPOCHS
    BATCH_SIZE = train_info['CFG'].BATCH_SIZE
    # epochs_client = 20
    losses = []
    val_losses = []
    best = {'epoch': -1, 'val_accuracy': -1.0, 'val_accs': [], 'val_losses': [], 'train_accs': [], 'train_losses': []}
    val_cnt = 0
    pre_val_loss = 0
    # # here, you need make sure weight aligned with class order.
    class_weight = torch.tensor(list(labeled_classes_weights.values()), dtype=torch.float).to(DEVICE)
    print(f'class_weight: {class_weight}, sum(class_weight) = {sum(class_weight)}')

    local_cnn = local_cnn.to(DEVICE)
    local_cnn.load_state_dict(global_cnn.state_dict())  # Initialize client_gm with the parameters of global_model
    # optimizer = optim.Adam(local_cnn.parameters(), lr=0.005)
    optimizer = optim.Adam(local_cnn.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    # optimizer = torch.optim.AdamW(local_cnn.parameters(), lr=0.001, weight_decay=5e-4)
    # criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean').to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    dataset = CustomDataset(X_train, y_train)
    if BATCH_SIZE == -1:
        BATCH_SIZE = len(y_train)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(CLIENT_EPOCHS):
        local_cnn.train()  #
        model_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # your local personal model
            outputs = local_cnn(X_batch)
            # Loss calculation: Only for labeled nodes
            loss_ = criterion(outputs, y_batch)
            loss_.backward()
            # # Print gradients for each parameter
            # print("Gradients for model parameters:")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad}")
            #     else:
            #         print(f"{name}: No gradient (likely frozen or unused)")
            optimizer.step()
            scheduler.step()  # adjust learning rate
            model_loss += loss_.item()
        losses.append(model_loss)

        # val_loss = model_loss
        # if epoch % 10 == 0:
        #     print(f'epoch: {epoch}, model_loss: {model_loss.item()}')
        #     evaluate_train(local_cnn, graph_data, len(local_data['y']), generated_size, epoch, local_data)
        # # X_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
        # val_loss, pre_val_loss, val_cnt, stop_training = early_stopping(local_cnn, graph_data, None, epoch,
        #                                                                 pre_val_loss, val_cnt, criterion,
        #                                                                 patience=epochs_client,
        #                                                                 best=best)
        # val_losses.append(val_loss.item())
        val_loss = model_loss
        if epoch % 100 == 0 or epoch == CLIENT_EPOCHS - 1:
            print(f"train_cnn epoch: {epoch}, local_cnn train loss: {model_loss:.4f}, "
                  f"val_loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]}")

        # if stop_training:
        #     local_cnn.stop_training = True
        #     print(f'Early Stopping. Epoch: {epoch}, Loss: {model_loss:.4f}')
        #     break

    # exit()
    # train_info['cnn'] = {'graph_data': graph_data, "losses": losses}
    # print('***best at epoch: ', best['epoch'], ' best val_accuracy: ', best['val_accuracy'])
    # local_cnn.load_state_dict(best['model'])

    # if train_info['server_epoch'] % 10 == 0:
    #     X_train, y_train = graph_data.x[graph_data.train_mask], graph_data.y[graph_data.train_mask]
    #     X_val, y_val = graph_data.x[graph_data.val_mask], graph_data.y[graph_data.val_mask]
    #     X_test, y_test = graph_data.x[graph_data.test_mask], graph_data.y[graph_data.test_mask]
    #     # test on the generated data
    #     dim = X_train.shape[1]
    #     X_gen_test = np.zeros((0, dim))
    #     y_gen_test = np.zeros((0,), dtype=int)
    #     for l, vs in generated_data.items():
    #         X_gen_test = np.concatenate((X_gen_test, vs['X']), axis=0)
    #         y_gen_test = np.concatenate((y_gen_test, vs['y']))
    #
    #
    #     evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_gen_test, y_gen_test, verbose=10)

    show = False
    if show:
        IN_DIR = '..'
        client_id = train_info['client_id']
        server_epoch = train_info['server_epoch']
        fig_file = f'{IN_DIR}/{client_id}/server_epoch_{server_epoch}.png'
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = axes.reshape((1, 2))
        val_losses = best['val_losses']
        train_losses = best['train_losses']
        axes[0, 0].plot(range(len(losses)), train_losses, label='Training Loss', marker='+')  # in early_stopping
        axes[0, 0].plot(range(len(val_losses)), val_losses, label='Validating Loss', marker='o')  # in early_stopping
        # axes[0, 0].plot(range(len(losses)), losses, label='Training Loss', marker='o')  # in cnn_train
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        best_val_acc = best['val_accuracy']
        train_acc = best['train_accuracy']
        epoch = best['epoch']
        axes[0, 0].set_title(f'Best_Val_Acc: {best_val_acc:.2f}, train: {train_acc:.2f} at Epoch: {epoch}')
        axes[0, 0].legend(fontsize='small')

        train_accs = best['train_accs']
        val_accs = best['val_accs']
        axes[0, 1].plot(range(len(val_accs)), train_accs, label='Training Accuracy', marker='+')  # in early_stopping
        axes[0, 1].plot(range(len(val_accs)), val_accs, label='Validating Accuracy', marker='o')  # in early_stopping
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        best_val_acc = best['val_accuracy']
        epoch = best['epoch']
        axes[0, 1].set_title(f'Best_Val_Acc: {best_val_acc:.2f},  train: {train_acc:.2f}  at Epoch: {epoch}')
        axes[0, 1].legend(fontsize='small')

        # plt.suptitle(title)
        # plt.grid()
        plt.tight_layout()
        plt.savefig(fig_file)
        # plt.show()
        plt.clf()
        plt.close(fig)

    return None


#
# def aggregate_cnns_layers(clients_cnns, clients_info, global_cnn, aggregation_method, histories, epoch):
#     print('*aggregate cnn...')
#
#     # Initialize the aggregated state_dict for the global model
#     global_state_dict = {key: torch.zeros_like(value).to(DEVICE) for key, value in global_cnn.state_dict().items()}
#
#     # Aggregate parameters for each layer
#     for key in global_state_dict:
#         # print(f'global_state_dict: {key}')
#         clients_updates = [client_state_dict[key].cpu() for client_state_dict in clients_cnns.values()]
#         min_value = min([torch.min(v).item() for v in clients_updates[: NUM_HONEST_CLIENTS]])
#         max_value = max([torch.max(v).item() for v in clients_updates[: NUM_HONEST_CLIENTS]])
#         # each client extra information (such as, number of samples)
#         # client_weights will affect median and krum, so be careful to weights
#         # if assign byzantine clients with very large weights (e.g., 1e6),
#         # then median will choose byzantine client's parameters.
#         clients_weights = torch.tensor([1] * len(clients_updates))  # default as 1
#         # clients_weights = torch.tensor([vs['size'] for vs in clients_info.values()])
#         if aggregation_method == 'adaptive_krum':
#             aggregated_update, clients_type_pred = adaptive_krum(clients_updates, clients_weights, trimmed_average=False)
#         elif aggregation_method == 'krum':
#             train_info = list(histories['clients'][-1].values())[-1]
#             f = train_info['NUM_BYZANTINE_CLIENTS']
#             # client_type = train_info['client_type']
#             aggregated_update, clients_type_pred = krum(clients_updates, clients_weights, f, trimmed_average=False)
#         elif aggregation_method == 'median':
#             aggregated_update, clients_type_pred = median(torch.stack(clients_updates, dim=0), clients_weights, dim=0)
#         else:
#             aggregated_update, clients_type_pred = mean(clients_updates, clients_weights)
#         print(f'{aggregation_method}, {key}, clients_type: {clients_type_pred}, '
#               f'client_updates: min: {min_value}, max: {max_value}')  # f'clients_weights: {clients_weights.numpy()},
#         global_state_dict[key] = aggregated_update.to(DEVICE)
#
#     # Update the global model with the aggregated parameters
#     global_cnn.load_state_dict(global_state_dict)
#

@timer
def aggregate_cnns(clients_cnns, clients_info, global_cnn, aggregation_method, histories, epoch):
    print('*aggregate cnn...')
    CFG = histories['CFG']
    # flatten all the parameters into a long vector
    # clients_updates = [client_state_dict.cpu() for client_state_dict in clients_cnns.values()]

    # Concatenate all parameter tensors into one vector.
    # Note: The order here is the iteration order of the OrderedDict, which
    # may not match the order of model.parameters().
    # vector_from_state = torch.cat([param.view(-1) for param in state.values()])
    # flatten_clients_updates = [torch.cat([param.view(-1).cpu() for param in client_state_dict.values()]) for
    #                            client_state_dict in clients_cnns.values()]
    tmp_models = []
    for client_state_dict in clients_cnns.values():
        model = CFG.CNN(num_classes=CFG.NUM_CLASSES)
        model.load_state_dict(client_state_dict)
        tmp_models.append(model)
    flatten_clients_updates = [parameters_to_vector(md.parameters()).detach().cpu() for md in tmp_models]

    # for v in flatten_clients_updates:
    #     # print(v.tolist())
    #     print_histgram(v, bins=10, value_type='update')

    flatten_clients_updates = torch.stack(flatten_clients_updates)
    print(f'each update shape: {flatten_clients_updates[1].shape}')
    # for debugging
    if CFG.VERBOSE >= 30:
        for i, update in enumerate(flatten_clients_updates):
            print(f'client_{i}:', end='  ')
            print_histgram(update, bins=5, value_type='params')

    min_value = min([torch.min(v).item() for v in flatten_clients_updates[: CFG.NUM_HONEST_CLIENTS]])
    max_value = max([torch.max(v).item() for v in flatten_clients_updates[: CFG.NUM_HONEST_CLIENTS]])

    # each client extra information (such as, number of samples)
    # client_weights will affect median and krum, so be careful to weights
    # if assign byzantine clients with very large weights (e.g., 1e6),
    # then median will choose byzantine client's parameters.
    clients_weights = torch.tensor([1] * len(flatten_clients_updates))  # default as 1
    # clients_weights = torch.tensor([vs['size'] for vs in clients_info.values()])
    start = time.time()
    if aggregation_method == 'adaptive_krum':
        aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(flatten_clients_updates,
                                                                                clients_weights,
                                                                                trimmed_average=False,
                                                                                random_projection=False,
                                                                                verbose=CFG.VERBOSE)
    elif aggregation_method == 'adaptive_krum_avg':
        aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(flatten_clients_updates,
                                                                                clients_weights,
                                                                                trimmed_average=True,
                                                                                random_projection=False,
                                                                                verbose=CFG.VERBOSE)

    elif aggregation_method == 'adaptive_krum+rp':  # adaptive_krum + random projection
        aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(
            flatten_clients_updates, clients_weights, trimmed_average=False, random_projection=True,
            random_state=CFG.TRAIN_VAL_SEED, verbose=CFG.VERBOSE)
    elif aggregation_method == 'adaptive_krum+rp_avg':  # adaptive_krum + random projection
        aggregated_update, clients_type_pred = robust_aggregation.adaptive_krum(
            flatten_clients_updates, clients_weights, trimmed_average=True, random_projection=True,
            random_state=CFG.TRAIN_VAL_SEED, verbose=CFG.VERBOSE)
    elif aggregation_method == 'krum':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        f = CFG.NUM_BYZANTINE_CLIENTS
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
                                                                       trimmed_average=False, random_projection=False,
                                                                       verbose=CFG.VERBOSE)
    elif aggregation_method == 'krum_avg':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        f = CFG.NUM_BYZANTINE_CLIENTS
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
                                                                       trimmed_average=True, random_projection=False,
                                                                       verbose=CFG.VERBOSE)
    elif aggregation_method == 'krum+rp':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        f = CFG.NUM_BYZANTINE_CLIENTS
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates,
                                                                       clients_weights, f,
                                                                       trimmed_average=False,
                                                                       random_projection=True,
                                                                       random_state=CFG.TRAIN_VAL_SEED,
                                                                       verbose=CFG.VERBOSE)
    elif aggregation_method == 'krum+rp_avg':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        f = CFG.NUM_BYZANTINE_CLIENTS
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates,
                                                                       clients_weights, f,
                                                                       trimmed_average=True,
                                                                       random_projection=True,
                                                                       random_state=CFG.TRAIN_VAL_SEED,
                                                                       verbose=CFG.VERBOSE)
    elif aggregation_method == 'median':
        p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
        p = p / 2  # top p/2 and bottom p/2 are removed
        aggregated_update, clients_type_pred = robust_aggregation.cw_median(flatten_clients_updates, clients_weights,
                                                                            verbose=CFG.VERBOSE)

    elif aggregation_method == 'medoid':
        p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
        aggregated_update, clients_type_pred = robust_aggregation.medoid(flatten_clients_updates,
                                                                         clients_weights,
                                                                         trimmed_average=False,
                                                                         upper_trimmed_ratio=p,
                                                                         verbose=CFG.VERBOSE)

    elif aggregation_method == 'medoid_avg':
        p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
        aggregated_update, clients_type_pred = robust_aggregation.medoid(flatten_clients_updates,
                                                                         clients_weights,
                                                                         trimmed_average=True,
                                                                         upper_trimmed_ratio=p,
                                                                         verbose=CFG.VERBOSE)

    elif aggregation_method == 'geometric_median':
        aggregated_update, clients_type_pred = robust_aggregation.geometric_median(flatten_clients_updates,
                                                                                   clients_weights,
                                                                                   max_iters=100, tol=1e-6,
                                                                                   verbose=CFG.VERBOSE)

    # elif aggregation_method == 'exp_weighted_mean':
    #     clients_type_pred = None
    #     aggregated_update = exp_weighted_mean.robust_center_exponential_reweighting_tensor(
    #         torch.stack(flatten_clients_updates), x_est=flatten_clients_updates[-1],
    #         r=0.1, max_iters=100, tol=1e-6, verbose=CFG.VERBOSE)
    elif aggregation_method == 'trimmed_mean':
        p = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
        p = p / 2  # top p/2 and bottom p/2 are removed
        aggregated_update, clients_type_pred = robust_aggregation.trimmed_mean(flatten_clients_updates, clients_weights,
                                                                               trim_ratio=p,
                                                                               verbose=CFG.VERBOSE)
    else:
        # empirical mean
        aggregated_update, clients_type_pred = robust_aggregation.cw_mean(flatten_clients_updates, clients_weights,
                                                                          verbose=CFG.VERBOSE)
    end = time.time()
    time_taken = end - start
    n = len(flatten_clients_updates)
    f = CFG.NUM_BYZANTINE_CLIENTS
    # # weight average
    # update = 0.0
    # weight = 0.0
    # for j in range(n-f):  # note here is k+1 because we want to add all values before k+1
    #     update += flatten_clients_updates[j] * clients_weights[j]
    #     weight += clients_weights[j]
    # empirical_mean = update / weight
    empirical_mean = torch.sum(flatten_clients_updates[:n - f] *
                               clients_weights[:n - f, None], dim=0) / torch.sum(clients_weights[:n - f])
    l2_error = torch.norm(empirical_mean - aggregated_update, p=2).item()  # l2 norm
    histories['server'].append({"time_taken": time_taken, 'l2_error': l2_error})
    # f'clients_weights: {clients_weights.numpy()},
    print(f'{aggregation_method}, clients_type: {clients_type_pred}, '
          f'client_updates: min: {min_value:.2f}, max: {max_value:.2f}, '
          f'time taken: {time_taken:.4f}s, l2_error: {l2_error:.2f}')

    # Update the global model with the aggregated parameters
    # w = w0 - (delta_w), where delta_w = \eta*\namba_w
    aggregated_update = parameters_to_vector(global_cnn.parameters()).detach().cpu() - aggregated_update
    aggregated_update = aggregated_update.to(CFG.DEVICE)
    vector_to_parameters(aggregated_update, global_cnn.parameters())  # in_place
    # global_cnn.load_state_dict(aggregated_update)


@timer
def evaluate(local_cnn, local_data, global_cnn, test_type='test', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    print('---------------------------------------------------------------')
    DEVICE = train_info['DEVICE']
    for model_type, model in [('global', global_cnn), ('local', local_cnn)]:
        # At time t, global model has not been updated yet, however, local_cnn is updated.
        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'***Testing {model_type} model on {test_type}...')
        cnn = model
        cnn = cnn.to(DEVICE)

        # X, Y, train_mask, val_mask, test_mask = train_info['cnn']['data']
        X, Y = local_data['X'], local_data['y']
        train_mask, val_mask, test_mask = local_data['train_mask'], local_data['val_mask'], local_data['test_mask']

        cnn.eval()
        with torch.no_grad():
            output = cnn(X)
            _, predicted_labels = torch.max(output, dim=1)

            # for debug purpose
            for data_type, mask_ in [('train', train_mask),
                                     ('val', val_mask),
                                     ('test', test_mask)]:
                # Calculate accuracy for the labeled data
                # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
                # print(f'labeled_indices {len(labeled_indices)}')
                true_labels = Y

                predicted_labels_tmp = predicted_labels[mask_]
                true_labels_tmp = true_labels[mask_]
                y = true_labels_tmp.cpu().numpy()
                y_pred = predicted_labels_tmp.cpu().numpy()
                print(collections.Counter(y.tolist()))

                # Total samples and number of classes
                total_samples = len(y)
                # Compute class weights
                class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
                # sample_weight = [class_weights[y_0.item()] for y_0 in y]
                sample_weight = [1 for y_0 in y]
                # print(f'class_weights: {class_weights}')

                accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)

                train_info[f'{model_type}_{data_type}_accuracy'] = accuracy
                print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
                conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
                conf_matrix = conf_matrix.astype(int)
                train_info[f'{model_type}_{data_type}_cm'] = conf_matrix
                print("Confusion Matrix:\n", conf_matrix)

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


@timer
def evaluate_shared_test(local_cnn, local_data, global_cnn,
                         test_type='shared_test_data', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    print('---------------------------------------------------------------')
    DEVICE = train_info['DEVICE']
    shared_data = local_data['shared_data']

    # shared_test
    X_test, y_test = shared_data['X'].to(DEVICE), shared_data['y'].to(DEVICE)
    print(f'X_test: {X_test.size()}, {collections.Counter(y_test.tolist())}')
    tmp = X_test.cpu().numpy().flatten()
    print(f'X_test: [min: {min(tmp)}, max: {max(tmp)}]')

    for model_type, model in [('global', global_cnn), ('local', local_cnn)]:
        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'***Testing {model_type} model on {test_type}...')
        # evaluate the data
        # cnn = local_cnn
        cnn = model
        cnn.to(DEVICE)
        cnn.eval()
        with torch.no_grad():
            output = cnn(X_test)
            _, predicted_labels = torch.max(output, dim=1)

            # only on test set
            print('Evaluate on shared test data...')
            # Calculate accuracy for the labeled data
            # NUM_CLASSES = 10
            # labeled_indices = graph_data.train_mask.nonzero(as_tuple=True)[0]  # Get indices of labeled nodes
            # print(f'labeled_indices {len(labeled_indices)}')
            true_labels = y_test

            y = true_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()

            # Total samples and number of classes
            total_samples = len(y)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y]
            sample_weight = [1 for y_0 in y]
            # print(f'class_weights: {class_weights}')

            accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
            train_info[f'{model_type}_shared_accuracy'] = accuracy
            print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
            conf_matrix = conf_matrix.astype(int)
            train_info[f'{model_type}_shared_cm'] = conf_matrix
            print("Confusion Matrix:")
            print(conf_matrix)

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


def print_data(local_data):
    print('Local_data: ')
    X, y = local_data['X'], local_data['y']
    vs = collections.Counter(y.tolist())
    vs = dict(sorted(vs.items(), key=lambda x: x[0], reverse=False))
    print(f'X: {X.shape}, y({len(vs)}): {vs}, in which, ')
    train_mask = local_data['train_mask']
    val_mask = local_data['val_mask']
    test_mask = local_data['test_mask']
    X_train, y_train = local_data['X'][train_mask], local_data['y'][train_mask]
    X_val, y_val = local_data['X'][val_mask], local_data['y'][val_mask]
    X_test, y_test = local_data['X'][test_mask], local_data['y'][test_mask]
    print(f'\tX_train: {X_train.shape}, y_train: '
          f'{collections.Counter(y_train.tolist())}')
    print(f'\tX_val: {X_val.shape}, y_val: '
          f'{collections.Counter(y_val.tolist())}')
    print(f'\tX_test: {X_test.shape}, y_test: '
          f'{collections.Counter(y_test.tolist())}')


def normalize(X):
    return (X / 255.0 - 0.5) * 2  # [-1, 1]


def print_histories(histories):
    CFG = histories['CFG']
    clients_histories = histories['clients']
    server_histories = histories['server']
    num_server_epoches = len(clients_histories)
    num_clients = len(clients_histories[0])
    print('num_server_epoches:', num_server_epoches, ' num_clients:', num_clients)

    for model_type in ['global', 'local']:
        print(f'\n***model_type: {model_type}***')
        ncols = 2
        nrows, r = divmod(num_clients, ncols)
        nrows = nrows if r == 0 else nrows + 1
        fig, axes = plt.subplots(nrows, ncols)
        for c in range(num_clients):
            i, j = divmod(c, ncols)
            print(f"\nclient {c}")
            train_accs = []  # train
            val_accs = []
            unlabeled_accs = []  # test
            shared_accs = []
            time_taken_list = []  # server
            l2_error_list = []  # server
            client_type = None
            try:
                for s in range(num_server_epoches):
                    client = clients_histories[s][c]
                    client_type = client["client_type"]
                    train_acc = client[f'{model_type}_train_accuracy']
                    val_acc = client[f'{model_type}_val_accuracy']
                    test_acc = client[f'{model_type}_test_accuracy']
                    shared_acc = client[f'{model_type}_shared_accuracy']
                    train_accs.append(train_acc)
                    val_accs.append(val_acc)
                    unlabeled_accs.append(test_acc)
                    shared_accs.append(shared_acc)

                    time_taken = server_histories[s]['time_taken']
                    time_taken_list.append(time_taken)
                    l2_error = server_histories[s]['l2_error']
                    l2_error_list.append(l2_error)
                    print(f'\t\tEpoch: {s}, labeled_acc:{train_acc:.2f}, val_acc:{val_acc:.2f}, '
                          f'unlabeled_acc:{test_acc:.2f}, '
                          f'shared_acc:{shared_acc:.2f}, '
                          f'time_taken:{time_taken:.2f}, l2_error:{l2_error:.2f}')
            except Exception as e:
                print(f'\t\tException: {e}')
            # Training and validation loss on the first subplot
            axes[i, j].plot(range(len(train_accs)), train_accs, label='labeled_acc', marker='o')
            axes[i, j].plot(range(len(val_accs)), val_accs, label='val_acc', marker='o')
            axes[i, j].plot(range(len(unlabeled_accs)), unlabeled_accs, label='unlabeled_acc', marker='+')
            axes[i, j].plot(range(len(shared_accs)), shared_accs, label='shared_acc', marker='s')
            axes[i, j].set_xlabel('Server Epochs')
            axes[i, j].set_ylabel('Accuracy')
            axes[i, j].set_title(f'Client_{c}: {client_type}')
            axes[i, j].legend(fontsize=6.5)

        attacker_ratio = CFG.NUM_BYZANTINE_CLIENTS / (CFG.NUM_HONEST_CLIENTS + CFG.NUM_BYZANTINE_CLIENTS)
        title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
                 f':{attacker_ratio:.2f}-{CFG.LABELING_RATE:.2f}')
        plt.suptitle(title)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig_file = (f'{CFG.data_out_dir}/{model_type}_{CFG.LABELING_RATE}_{CFG.AGGREGATION_METHOD}_'
                    f'{CFG.SERVER_EPOCHS}_{CFG.NUM_HONEST_CLIENTS}_{CFG.NUM_BYZANTINE_CLIENTS}_accuracy.png')
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300)
        plt.show()
        plt.close(fig)


def print_histgram(new_probs, bins=5, value_type='probs'):
    print(f'***Print histgram of {value_type}, min:{min(new_probs)}, max: {max(new_probs)}***')
    # # Convert the probabilities to numpy for histogram calculation
    # new_probs = new_probs.detach().cpu().numpy()
    # Compute histogram
    hist, bin_edges = torch.histogram(new_probs, bins=bins)
    # Print histogram
    for i in range(len(hist)):
        print(f"\tBin {i}: {value_type} Range ({bin_edges[i]}, {bin_edges[i + 1]}), Frequency: {hist[i]}")


def print_histories_server(histories_server):
    num_server_epoches = len(histories_server)

    for model_type in ['global']:
        print(f'\n***model_type: {model_type}***')
        ncols = 2
        nrows, r = divmod(4, ncols)
        nrows = nrows if r == 0 else nrows + 1
        fig, axes = plt.subplots(nrows, ncols)
        for clf_idx, clf_name in enumerate(['Random Forest']):
            i, j = divmod(clf_idx, ncols)
            print(f"\n {clf_name}")
            train_accs = []  # train
            val_accs = []
            unlabeled_accs = []  # test
            shared_accs = []
            for s in range(num_server_epoches):
                ml_info = histories_server[s][clf_name]
                #  ml_info[clf_name] = {test_type: {'accuracy': accuracy, 'cm': cm}}
                train_acc = ml_info['train']['accuracy']
                val_acc = ml_info['val']['accuracy']
                test_acc = ml_info['test']['accuracy']
                shared_acc = ml_info['shared_test']['accuracy']

                train_accs.append(train_acc)
                val_accs.append(val_acc)
                unlabeled_accs.append(test_acc)
                shared_accs.append(shared_acc)
                print(f'\t\tEpoch: {s}, labeled_acc:{train_acc:.2f}, val_acc:{val_acc:.2f}, '
                      f'unlabeled_acc:{test_acc:.2f}, '
                      f'shared_acc:{shared_acc:.2f}')

            # Training and validation loss on the first subplot
            axes[i, j].plot(range(len(train_accs)), train_accs, label='labeled_acc', marker='o')
            axes[i, j].plot(range(len(val_accs)), val_accs, label='val_acc', marker='o')
            axes[i, j].plot(range(len(unlabeled_accs)), unlabeled_accs, label='unlabeled_acc', marker='+')
            axes[i, j].plot(range(len(shared_accs)), shared_accs, label='shared_acc', marker='s')
            axes[i, j].set_xlabel('Server Epochs')
            axes[i, j].set_ylabel('Accuracy')
            axes[i, j].set_title(f'{clf_name}')
            axes[i, j].legend(fontsize=5)
        #
        # if model_type == 'global':
        #     title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}' + '}$' + f':{LABELING_RATE}'
        # else:
        #     title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' + f':{LABELING_RATE}'
        # plt.suptitle(title)
        #
        # # Adjust layout to prevent overlap
        # plt.tight_layout()
        # fig_file = f'{IN_DIR}/{LABELING_RATE}/{model_type}_accuracy.png'
        # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        # plt.savefig(fig_file, dpi=300)
        plt.show()
        plt.close(fig)


def print_all(all_histories, precision=4):
    # Set print options to avoid scientific notation
    np.set_printoptions(suppress=True)
    print(f"\n***************************** Final Results (mu+/-std) *************************************")
    REPETAS_SEEDS = all_histories.keys()
    for model_type in ['global', 'local']:
        print(f'\n Final***model_type: {model_type}***')
        NUM_CLIENTS = all_histories[0]['CFG'].NUM_CLIENTS
        for idx_client in range(NUM_CLIENTS):
            print(f'*client_{idx_client}:')
            # compute mu +\- std for each client, data_type, metric_type
            SERVER_EPOCHS = all_histories[0]['CFG'].SERVER_EPOCHS
            for idx_server_epoch in range(SERVER_EPOCHS):
                print(f'\tserver_epoch_{idx_server_epoch}, ', end='')
                for data_type in ['train', 'val', 'test', 'shared']:
                    res = [all_histories[repeat_seed]['clients'][idx_server_epoch][idx_client][
                               f'{model_type}_{data_type}_accuracy'] for repeat_seed in REPETAS_SEEDS]  # model type
                    mu, std = np.mean(res), np.std(res)
                    # .f ensures fixed-point notation instead of scientific notation.
                    print(f"{data_type}_acc:{mu:.{precision}f}-{std:.{precision}f}, ", end='')

                for metric_type in ['l2_error', 'time_taken']:
                    res = [all_histories[repeat_seed]['server'][idx_server_epoch][metric_type] for repeat_seed in
                           REPETAS_SEEDS]
                    res = np.array(res)
                    mu, std = np.mean(res), np.std(res)
                    print(f"{metric_type}:{mu:.{precision}f}-{std:.{precision}f}, ", end='')
                print()
