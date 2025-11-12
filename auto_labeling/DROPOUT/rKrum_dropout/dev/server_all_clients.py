import socket
import pickle
import argparse
import time
from dataclasses import dataclass
import os
from model import FNN
from ragg.base import print_all, print_histgram
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
from ragg import robust_aggregation, base


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="IP address of the server")
    parser.add_argument("--workspace_dir", type=str, default=".", help="Workspace directory")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of rounds")
    return parser.parse_args()


args = parse()
print(args)

HOST = args.server_ip
PORT = 60000
NUM_CLIENTS = args.num_clients
INITIAL_VALUE = 42
NUM_EPOCHS = args.num_rounds
NUM_CLASSES = 2
client_sockets = []
VERBOSE = 10


@dataclass
class CONFIG:
    def __init__(self):
        self.SEED = None
        self.TRAIN_VAL_SEED = 42
        self.DEVICE = None
        self.VERBOSE = None
        self.LABELING_RATE = None
        self.BIG_NUMBER = None
        self.SERVER_EPOCHS = None
        self.CLIENT_EPOCHS = 5
        self.BATCH_SIZE = 512
        self.IID_CLASSES_CNT = 5
        self.NUM_CLIENTS = None
        self.NUM_BYZANTINE_CLIENTS = None
        self.NUM_HONEST_CLIENTS = None
        self.AGGREGATION_METHOD = None
        self.LABELS = set()
        self.NUM_CLASSES = None

    def __str__(self):
        return str(self.__dict__)  # Prints attributes as a dictionary

    def __repr__(self):
        return f"CONFIG({self.__dict__})"  # More detailed representation


CFG = CONFIG()
CFG.LABELS = {0, 1}

CFG.NUM_CLIENTS = 2
CFG.BIG_NUMBER = 0.5  # alpha for controlling non-IID level
CFG.NUM_HONEST_CLIENTS = 1
CFG.NUM_BYZANTINE_CLIENTS = 1
CFG.LABELING_RATE = 0.1  # how much data for testing
CFG.VERBOSE = 1

import gen_data

out_dir = f'./data/{CFG.NUM_CLIENTS}_split'
if not os.path.exists(out_dir):
    print('### Start to generate data ###')
    gen_data.gen_client_data(data_dir='data/Sentiment140', out_dir=out_dir, CFG=CFG)
    print('### Finish to generate data ###')
    print()


def recv_from_client(conn):
    # Read first 4 bytes to get the size
    size_data = conn.recv(4)
    payload_size = int.from_bytes(size_data, byteorder='big')

    # Read full payload
    received_bytes = b''
    while len(received_bytes) < payload_size:
        packet = conn.recv(min(4096, payload_size - len(received_bytes)))
        if not packet:
            break
        received_bytes += packet

    value = pickle.loads(received_bytes)

    return value


def broadcast_to_clients(data):
    payload = pickle.dumps(data)
    payload_size = len(payload)
    for conn in client_sockets:
        # conn.sendall(payload)
        conn.sendall(payload_size.to_bytes(4, byteorder='big'))
        conn.sendall(payload)


def aggregate_update(global_model, client_data_list, aggregation_method='rkrum'):
    """ aggregate client data, and update the global model

    :param global_model:
    :param client_data_list:
    :param aggregation_method:
    :return: updated global model parameters
    """
    # @timer
    # def aggregate_cnns(clients_cnns, clients_info, global_cnn, aggregation_method, histories, epoch):
    print('*aggregate cnn...')
    # CFG = histories['CFG']
    # flatten all the parameters into a long vector
    # clients_updates = [client_state_dict.cpu() for client_state_dict in clients_cnns.values()]

    # Concatenate all parameter tensors into one vector.
    # Note: The order here is the iteration order of the OrderedDict, which
    # may not match the order of model.parameters().
    # vector_from_state = torch.cat([param.view(-1) for param in state.values()])
    # flatten_clients_updates = [torch.cat([param.view(-1).cpu() for param in client_state_dict.values()]) for
    #                            client_state_dict in clients_cnns.values()]
    tmp_models = []
    client_sizes = []
    for (client_state_dict, num_training_size) in client_data_list:  # each client returns delta_w, m
        model = FNN(num_classes=NUM_CLASSES)
        model.load_state_dict(client_state_dict)
        tmp_models.append(model)
        client_sizes.append(num_training_size)
    flatten_clients_updates = [parameters_to_vector(md.parameters()).detach().cpu() for md in tmp_models]

    # for v in flatten_clients_updates:
    #     # print(v.tolist())
    #     print_histgram(v, bins=10, value_type='update')

    flatten_clients_updates = torch.stack(flatten_clients_updates)
    print(f'each update shape: {flatten_clients_updates[1].shape}')
    # for debugging
    if VERBOSE >= 30:
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
                                                                       trimmed_average=False,
                                                                       random_projection=False,
                                                                       verbose=CFG.VERBOSE)
    elif aggregation_method == 'krum_avg':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        f = CFG.NUM_BYZANTINE_CLIENTS
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
                                                                       trimmed_average=True,
                                                                       random_projection=False,
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
        aggregated_update, clients_type_pred = robust_aggregation.cw_median(flatten_clients_updates,
                                                                            clients_weights,
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
        aggregated_update, clients_type_pred = robust_aggregation.trimmed_mean(flatten_clients_updates,
                                                                               clients_weights,
                                                                               trim_ratio=p,
                                                                               verbose=CFG.VERBOSE)
    else:
        # empirical mean
        clients_weights = torch.tensor(client_sizes)
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
    # histories['server'].append({"time_taken": time_taken, 'l2_error': l2_error})
    # f'clients_weights: {clients_weights.numpy()},
    print(f'{aggregation_method}, clients_type: {clients_type_pred}, '
          f'client_updates: min: {min_value:.2f}, max: {max_value:.2f}, '
          f'time taken: {time_taken:.4f}s, l2_error: {l2_error:.2f}')

    # Update the global model with the aggregated parameters
    # w = w0 - (delta_w), where delta_w = \eta*\namba_w
    aggregated_update = parameters_to_vector(global_model.parameters()).detach().cpu() - aggregated_update
    # aggregated_update = aggregated_update.to(DEVICE)
    vector_to_parameters(aggregated_update, global_model.parameters())  # in_place

    agg_val = global_model.state_dict()
    return agg_val


def main():
    print("[Server] Starting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(NUM_CLIENTS)
        print(f"[Server] Waiting for {NUM_CLIENTS} clients to connect...")

        # Accept clients: Randomly select clients
        while len(client_sockets) < NUM_CLIENTS:
            conn, addr = s.accept()
            print(f"[Server] Connected to {addr}")
            client_sockets.append(conn)

        print("[Server] All clients connected.")
        global_model = FNN(num_classes=NUM_CLASSES)
        shared_data = {
            'global_model_state_dict': global_model.state_dict(),
            'data_dir': f'data/{NUM_CLIENTS}_split',
            'server_epoch': 0,
            'client_type': 'HONEST_CLIENT',
            'CFG': CFG,
        }

        for epoch in range(NUM_EPOCHS):
            print(f"\n[Server] Epoch {epoch}...")

            # Step 1: Send current model value to all clients
            print(f"[Server] Broadcasting value: {shared_data.keys()}")
            broadcast_to_clients(shared_data)

            # Step 2: Receive updated values from clients
            epoch_data = []
            for conn in client_sockets:
                val = recv_from_client(conn)
                epoch_data.append(val)
            print(f"[Server] Received from clients: {len(epoch_data)}")

            # Step 3: Aggregate updates (average here)
            # current_data = sum(epoch_data) / len(epoch_data)
            update_params = aggregate_update(global_model, epoch_data,
                                             aggregation_method='adaptive_krum')  # update global_model in place.
            shared_data = {
                'global_model_state_dict': update_params,
                'data_dir': f'data/{NUM_CLIENTS}_split',
                'server_epoch': epoch+1,        # here must +1
                'CFG': CFG,
            }
            print(f"[Server] Aggregated value: {shared_data.keys()}")

        end_msg = {"done": True}
        broadcast_to_clients(end_msg)
        time.sleep(0.1)  # give clients time to receive

        # Close connections
        for conn in client_sockets:
            conn.close()
        print("[Server] Training complete. Connections closed.")


if __name__ == "__main__":
    start_time = time.monotonic_ns()
    main()
    end_time = time.monotonic_ns()
    print(f'start time is: {start_time}, end time is: {end_time}')
    print(f'The total time is: {end_time - start_time}ns.')
