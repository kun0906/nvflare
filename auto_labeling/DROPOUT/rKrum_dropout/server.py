import socket
import pickle
import argparse
import time
from dataclasses import dataclass
import os
from model import FNN
from ragg.base import print_all, print_histgram, evaluate_shared_test, evaluate_global_model
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
from ragg import robust_aggregation, base
import random
import numpy as np


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="IP address of the server")
    parser.add_argument("--workspace_dir", type=str, default=".", help="Workspace directory")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of rounds")
    return parser.parse_args()


args = parse()
print(args)


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
        self.BATCH_SIZE = 32
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


HOST = args.server_ip
PORT = 60000
NUM_CLIENTS = args.num_clients
INITIAL_VALUE = 42
NUM_EPOCHS = args.num_rounds
NUM_CLASSES = 2
client_sockets = []
VERBOSE = 5

CFG = CONFIG()
CFG.LABELS = {0, 1}  # 0 is negative and 1 is positive
CFG.NUM_CLIENTS = NUM_CLIENTS
CFG.BIG_NUMBER = 0.5  # alpha for controlling non-IID level
CFG.NUM_BYZANTINE_CLIENTS = (CFG.NUM_CLIENTS - 3) // 2
# CFG.NUM_BYZANTINE_CLIENTS = 1
CFG.NUM_HONEST_CLIENTS = CFG.NUM_CLIENTS - CFG.NUM_BYZANTINE_CLIENTS
CFG.LABELING_RATE = 0.2  # how much data for testing, we set it this the data split
CFG.VERBOSE = VERBOSE
print("Number of clients: ", NUM_CLIENTS)
print("NUM_BYZANTINE_CLIENTS: ", CFG.NUM_BYZANTINE_CLIENTS)
print("NUM_HONEST_CLIENTS: ", CFG.NUM_HONEST_CLIENTS)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# generated data
RANDOM_SEEDS = [v * 100 for v in range(5)]
for random_seed in RANDOM_SEEDS:
    out_dir = f'./data/{CFG.NUM_CLIENTS}_split_seed_{random_seed}'
    if not os.path.exists(out_dir):
        import gen_data

        print('### Start to generate data ###')
        gen_data.gen_client_data(data_dir='data/Sentiment140', out_dir=out_dir, CFG=CFG,
                                 random_state=random_seed)
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


def broadcast_to_clients(data, client_sockets, selected_clients):
    payload = pickle.dumps(data)
    payload_size = len(payload)
    for client_idx in selected_clients:
        conn, addr = client_sockets[client_idx]
        # conn.sendall(payload)
        conn.sendall(payload_size.to_bytes(4, byteorder='big'))
        conn.sendall(payload)


def aggregate_update(global_model, client_data, aggregation_method='rkrum', updated_f=0):
    """ aggregate client data, and update the global model

    :param global_model:
    :param client_data: {client_idx, data}
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
    selected_client_indices = []
    selected_clients_sizes=[]
    for client_idx, (client_state_dict, num_training_size) in client_data.items():  # each client returns delta_w
        model = FNN(num_classes=NUM_CLASSES)
        model.load_state_dict(client_state_dict)
        tmp_models.append(model)
        selected_client_indices.append(client_idx)
        selected_clients_sizes.append(num_training_size)
    flatten_clients_updates = [parameters_to_vector(md.parameters()).detach().cpu() for md in tmp_models]

    # for v in flatten_clients_updates:
    #     # print(v.tolist())
    #     print_histgram(v, bins=10, value_type='update')

    flatten_clients_updates = torch.stack(flatten_clients_updates)
    print(f'each update shape: {flatten_clients_updates[0].shape}')
    # for debugging
    if VERBOSE >= 30:
        for i, update in enumerate(flatten_clients_updates):
            print(f'client_{i}:', end='  ')
            print_histgram(update, bins=5, value_type='params')

    # min_value = min([torch.min(v).item() for v in flatten_clients_updates[: CFG.NUM_HONEST_CLIENTS]])
    # max_value = max([torch.max(v).item() for v in flatten_clients_updates[: CFG.NUM_HONEST_CLIENTS]])
    min_value = -1000
    max_value = -1000
    # each client extra information (such as, number of samples)
    # client_weights will affect median and krum, so be careful to weights
    # if assign byzantine clients with very large weights (e.g., 1e6),
    # then median will choose byzantine client's parameters.
    clients_weights = torch.tensor([1] * len(flatten_clients_updates))  # default as 1
    # clients_weights = torch.tensor([vs['size'] for vs in clients_info.values()])
    start = time.time()
    if aggregation_method == 'adaptive_krum' or aggregation_method == 'rkrum':
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
        # f = CFG.NUM_BYZANTINE_CLIENTS
        f = updated_f  # in each sample clients, how many of them are Byzantine clients
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
                                                                       trimmed_average=False,
                                                                       random_projection=False,
                                                                       verbose=CFG.VERBOSE)
    elif aggregation_method == 'krum_avg':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        # f = CFG.NUM_BYZANTINE_CLIENTS
        f = updated_f  # in each sample clients, how many of them are Byzantine clients
        # client_type = train_info['client_type']
        aggregated_update, clients_type_pred = robust_aggregation.krum(flatten_clients_updates, clients_weights, f,
                                                                       trimmed_average=True,
                                                                       random_projection=False,
                                                                       verbose=CFG.VERBOSE)
    elif aggregation_method == 'krum+rp':
        # train_info = list(histories['clients'][-1].values())[-1]
        # f = train_info['NUM_BYZANTINE_CLIENTS']
        # f = CFG.NUM_BYZANTINE_CLIENTS
        f = updated_f  # in each sample clients, how many of them are Byzantine clients
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
        # f = CFG.NUM_BYZANTINE_CLIENTS
        f = updated_f  # in each sample clients, how many of them are Byzantine clients
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
    elif aggregation_method == 'mean':
        # empirical mean
        clients_weights = torch.tensor(selected_clients_sizes)
        print(f"mean, clients_weights: {clients_weights}")
        aggregated_update, clients_type_pred = robust_aggregation.cw_mean(flatten_clients_updates, clients_weights,
                                                                          verbose=CFG.VERBOSE)
    else:
        raise NotImplementedError(f'Aggregation method not implemented: {aggregation_method}')
    end = time.time()
    time_taken = end - start
    # n = len(flatten_clients_updates)
    # f = CFG.NUM_BYZANTINE_CLIENTS
    # # # weight average
    # # update = 0.0
    # # weight = 0.0
    # # for j in range(n-f):  # note here is k+1 because we want to add all values before k+1
    # #     update += flatten_clients_updates[j] * clients_weights[j]
    # #     weight += clients_weights[j]
    # # empirical_mean = update / weight
    # empirical_mean = torch.sum(flatten_clients_updates[:n - f] *
    #                            clients_weights[:n - f, None], dim=0) / torch.sum(clients_weights[:n - f])
    # l2_error = torch.norm(empirical_mean - aggregated_update, p=2).item()  # l2 norm
    l2_error = -1000
    # histories['server'].append({"time_taken": time_taken, 'l2_error': l2_error})
    # f'clients_weights: {clients_weights.numpy()},
    print(f'{aggregation_method}, clients_type: {clients_type_pred}, '
          f'selected_client_indices: {selected_client_indices}, '
          f'client_updates: min: {min_value:.2f}, max: {max_value:.2f}, '
          f'time taken: {time_taken:.4f}s, l2_error: {l2_error:.2f}')

    # Update the global model with the aggregated parameters
    # w = w0 - (delta_w), where delta_w = \eta*\namba_w
    aggregated_update = parameters_to_vector(global_model.parameters()).detach().cpu() - aggregated_update
    # aggregated_update = aggregated_update.to(DEVICE)
    vector_to_parameters(aggregated_update, global_model.parameters())  # in_place

    agg_val = global_model.state_dict()
    return agg_val


def get_clients(client_indices, dropout_type='random', kwargs=None):
    """ in each epoch, we should get different selected clients.
    
    :param client_indices:
    :param dropout_type:
    :return:
    """
    if dropout_type == 'none':  # use all clients
        selected_clients = client_indices
    elif dropout_type == 'random':
        m = int(0.8 * len(client_indices))  # 80% percent of clients will be chosen. # fixed number.
        # m = random.randint(1, len(client_indices))
        # random.shuffle(client_indices)
        # selected_clients = client_indices[:m]
        selected_clients = random.sample(client_indices, k=m)
    elif dropout_type == 'random_m':
        m = random.randint(1, len(client_indices))
        # random.shuffle(client_indices)
        # selected_clients = client_indices[:m]
        selected_clients = random.sample(client_indices, k=m)
    elif dropout_type == 'bernoulli':
        # bernoulli
        p = 0.8  # each client has a 80% chance of being selected
        selected_clients = [c for c in client_indices if random.random() < p]
    elif dropout_type == 'markov':
        N = len(client_indices)
        more_available_clients = kwargs.get('more_available_clients', None)
        less_available_clients = kwargs.get('less_available_clients', None)
        states = kwargs.get('states', None)

        g = 0.4  # g in (0, 1/2)
        more_available_clients_pi_k_active = 1 / 2 + g
        less_available_clients_pi_k_active = 1 / 2 - g

        #### for each client k in more_available_clients
        # stationary distribution pi_k
        # pi_k = [1 - more_available_clients_pi_k_active, more_available_clients_pi_k_active]
        # #pi_k = [1 - less_available_clients_pi_k_active, less_available_clients_pi_k_active]
        # transition matrix P_k  
        # P_k = [ p_0, 1 - p_0, 
        #         1 - p_1, p_1]
        lambda_2 = 0.5  # p_0 + p_1 - 1
        # Define transition matrices
        def compute_transition_matrix(pi_k, lambda_2):
            p_00 = 1 - (1 - lambda_2) * pi_k
            p_11 = lambda_2 + (1 - lambda_2) * pi_k
            return [[p_00, 1 - p_00],
                    [1 - p_11, p_11]]

        P_k_more = compute_transition_matrix(more_available_clients_pi_k_active, lambda_2)
        print('P_k_more:', P_k_more)
        # check pi_k = [1- pi_k, pi_k] = [(1 - p_11)/(2- p_00 - p_11), (1-p_00)/(2-p_00-p_11)]
        p_11 = P_k_more[1][1]       # active with value 1 
        p_00 = P_k_more[0][0]       # inactive with value 0 
        pi_0 = (1-p_11)/(2-p_00-p_11)
        pi_1 = (1-p_00)/(2-p_00-p_11)
        # assert pi_0 == 1 - more_available_clients_pi_k_active
        # assert pi_1 == more_available_clients_pi_k_active
        print(f'pi_0({pi_0}) == 1 - more_available_clients_pi_k_active({more_available_clients_pi_k_active}): {pi_0 == 1 - more_available_clients_pi_k_active}')
        print(f'pi_1({pi_1}) ==  more_available_clients_pi_k_active({more_available_clients_pi_k_active}): {pi_1 ==  more_available_clients_pi_k_active}')
        
        P_k_less = compute_transition_matrix(less_available_clients_pi_k_active, lambda_2)
        print('P_k_less:', P_k_less)
        p_11 = P_k_less[1][1]
        p_00 = P_k_less[0][0]
        pi_0 = (1 - p_11) / (2 - p_00 - p_11)
        pi_1 = (1 - p_00) / (2 - p_00 - p_11)
        # assert pi_0 == 1 - less_available_clients_pi_k_active
        # assert pi_1 == less_available_clients_pi_k_active
        print(f'pi_0({pi_0})== 1 - less_available_clients_pi_k_active({less_available_clients_pi_k_active}): {pi_0 == 1 - less_available_clients_pi_k_active}')
        print(f'pi_1({pi_1}) ==  less_available_clients_pi_k_active({less_available_clients_pi_k_active}): {pi_1 == less_available_clients_pi_k_active}')
        
        next_states = [0] * N
        for c in more_available_clients:
            next_states[c-1] = np.random.choice([0, 1], p=P_k_more[states[c-1]])

        for c in less_available_clients:
            next_states[c-1] = np.random.choice([0, 1], p=P_k_less[states[c-1]])

        selected_clients = [c for c in client_indices if next_states[c-1] == 1]       # client_indices starts from 1, next_states starts from 0 
    else:
        raise ValueError(f'Dropout type {dropout_type} not supported')

    return selected_clients


def main():
    print("[Server] Starting...", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        NUM_CLIENTS = CFG.NUM_CLIENTS
        server_socket.listen(NUM_CLIENTS)
        print(f"[Server] Waiting for {NUM_CLIENTS} clients to connect...", flush=True)

        client_sockets = {}
        while len(client_sockets) < NUM_CLIENTS:
            conn, addr = server_socket.accept()
            try:
                client_idx = int(conn.recv(1024).decode())
                if client_idx not in client_sockets:
                    client_sockets[client_idx] = (conn, addr)
                    print(f"[Server] Connected to client {client_idx} at {addr}")
                else:
                    print(f"[Server] Duplicate client_idx {client_idx}, ignoring.")
                    conn.close()
            except Exception as e:
                print(f"[Server] Error receiving client ID: {e}")
                conn.close()

        print("[Server] All clients connected.")

        print('-----------------------------------------------------------------------------------')
        results = {}
        aggregate_methods = ['rkrum', 'krum', 'mean']
        for aggregate_method in aggregate_methods:
            results_per_seed = {}
            for random_seed in RANDOM_SEEDS:
                start_time = time.monotonic_ns()
                random.seed(random_seed)  # Fix the seed for reproducibility for each experiment
                print(f'\n ------- aggregate method: {aggregate_method}, random_seed: {random_seed} --------')
                global_model = FNN(num_classes=NUM_CLASSES)
                shared_data = {
                    'global_model_state_dict': global_model.state_dict(),
                    'data_dir': f'data/{NUM_CLIENTS}_split',
                    'server_epoch': 0,
                    'client_type': 'HONEST_CLIENT',
                    'CFG': CFG,
                }
                accs = []
                selected_clients_list=[]
                client_indices = list(client_sockets.keys())
                for epoch in range(NUM_EPOCHS):
                    print(f"\n[Server] Epoch {epoch}...")

                    # select a subset of clients for initialization
                    dropout_type ='random' #'markov'  # 'bernoulli' #'random_m'
                    f = len([v for v in client_indices if v > CFG.NUM_HONEST_CLIENTS])
                    print(f'client_indices ({len(client_indices)}): {client_indices}, f:{f}, {dropout_type}')
                    if dropout_type == 'markov':
                        if epoch == 0:
                            # Randomly select half of clients in the first epoch
                            selected_clients = random.sample(client_indices, len(client_indices) // 2) # client_indices starts from 1
                            # Partition clients
                            num_selected = len(client_indices) // 2
                            more_available_clients = random.sample(client_indices, num_selected)
                            less_available_clients = [v for v in client_indices if v not in set(more_available_clients)]
                            
                        # Initialize states as all unavailable (0)
                        states = [0] * len(client_indices)
                        # Mark selected clients as available (1)
                        for c in selected_clients:
                            states[c-1] = 1
                        kwargs = {'more_available_clients': more_available_clients, 
                                  'less_available_clients': less_available_clients,
                                  'states': states}
                        selected_clients = get_clients(client_indices, dropout_type=dropout_type,
                                                       kwargs=kwargs)  # in each epoch, we select clients using Markov.
                    else:
                        selected_clients = get_clients(client_indices,
                                                       dropout_type=dropout_type)  # in each epoch, we should get different random clients
                    updated_f = len([v for v in selected_clients if
                                     v > CFG.NUM_HONEST_CLIENTS])  # in each sample clients, how many of them are Byzantine clients
                    print(f"select_clients ({len(selected_clients)}): {selected_clients}, updated_f: {updated_f}")
                    selected_clients_list.append(
                        {'selected_clients': selected_clients, "f":f, "updated_f": updated_f}
                    )
                    if len(selected_clients) == 0:
                        continue

                    # Step 1: Send current model value to all clients
                    print(f"[Server] Broadcasting value: {shared_data.keys()}")
                    broadcast_to_clients(shared_data, client_sockets, selected_clients)

                    # Step 2: Receive updated values from clients
                    epoch_data = {}
                    for client_idx in selected_clients:
                        conn, addr = client_sockets[client_idx]
                        val = recv_from_client(conn)
                        epoch_data[client_idx] = val
                    print(f"[Server] Received from clients: {len(epoch_data.keys())}")

                    # Step 3: Aggregate updates (average here)
                    # current_data = sum(epoch_data) / len(epoch_data)
                    update_params = aggregate_update(global_model, epoch_data,
                                                     aggregation_method=aggregate_method,
                                                     updated_f=updated_f)  # update global_model in place.
                    shared_data = {
                        'global_model_state_dict': update_params,
                        'data_dir': f'data/{NUM_CLIENTS}_split',
                        'server_epoch': epoch + 1,  # here must +1
                        'CFG': CFG,
                    }
                    print(f"[Server] Aggregated value: {shared_data.keys()}")

                    global_model = FNN(num_classes=NUM_CLASSES).to(DEVICE)
                    global_model.load_state_dict(update_params)
                    data_file = f'data/{NUM_CLIENTS}_split_seed_{random_seed}/0.pth'  # get shared_test_data
                    with open(data_file, 'rb') as f:
                        local_data = torch.load(f, weights_only=False)
                    res = evaluate_global_model(global_model, local_data['shared_data'], DEVICE,
                                                test_type='Shared test data', train_info={}, verbose=VERBOSE)
                    accs.append(res)
                end_time = time.monotonic_ns()
                print(f'start time is: {start_time}, end time is: {end_time}')
                print(f'The total time is: {end_time - start_time}ns.')
                results_per_seed[random_seed] = {'accs': accs, 'time': end_time - start_time, 'selected_clients_list': selected_clients_list}
            results[aggregate_method] = results_per_seed

        print('-----------------------------------------------------------------------------------')
        end_msg = {"done": True}
        broadcast_to_clients(end_msg, client_sockets, client_indices)
        time.sleep(0.1)  # give clients time to receive

        # Close connections
        for client_idx, (conn, addr) in client_sockets.items():
            conn.close()
        print("[Server] Training complete. Connections closed.")

        print(f'out/{dropout_type}.pkl')
        with open(f'out/{dropout_type}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print('\n')
        
        for random_seed in RANDOM_SEEDS:
            print(f'\n***random_seed:{random_seed}')
            for metric in ['accs', 'time', 'selected_clients_list']:
                for aggregate_method in aggregate_methods:
                    if metric == 'selected_clients_list':
                        print(f'{aggregate_method}_{metric} = ', 
                              [len(v['selected_clients']) for v in results[aggregate_method][random_seed][metric]])
                        print('\tupdated_f:', [v['updated_f'] for v in results[aggregate_method][random_seed][metric]])
                    else:
                        print(f'{aggregate_method}_{metric} = ', results[aggregate_method][random_seed][metric])
        
        print('\n')

if __name__ == "__main__":
    start_time = time.monotonic_ns()
    main()
    end_time = time.monotonic_ns()
    print(f'start time is: {start_time}, end time is: {end_time}')
    print(f'The total time is: {end_time - start_time}ns.')
