import socket
import pickle
import random
import traceback
from dataclasses import dataclass

import torch
import argparse

import collections

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from model import FNN
from ragg.base import print_data, train_cnn, evaluate, evaluate_shared_test

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="IP address of the server")
    parser.add_argument("--workspace_dir", type=str, default=".", help="Workspace directory")
    parser.add_argument("--client_idx", type=int, default=1, help="The index of the client")    # client index starts from 1, not 0
    return parser.parse_args()


args = parse()
print(args)

SERVER_IP = args.server_ip
PORT = 60000
CLIENT_IDX = args.client_idx
# data_dir = ''
NUM_CLASSES = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def local_test():
    pass


def client_train_test(received_value):
    """Simulate local training step"""
    # return received_value + random.uniform(-1, 1)
    CFG = received_value['CFG']
    server_epoch = received_value['server_epoch']
    # client_type = received_value['client_type']    #     client_type = 'Honest'
    client_type = 'Honest' if CLIENT_IDX-1 < CFG.NUM_HONEST_CLIENTS else 'Byzantine'
    global_model_state_dict = received_value['global_model_state_dict']
    data_dir = received_value['data_dir']
    print(f"\n*** server_epoch:{server_epoch}, client_{CLIENT_IDX}, client_type: {client_type}: ... ***")
    # might be used in server
    train_info = {"client_type": client_type, "fnn": {}, 'client_id': CLIENT_IDX, 'server_epoch': server_epoch,
                  'DEVICE': DEVICE, 'CFG': CFG}

    data_file = f'{data_dir}/{CLIENT_IDX-1}.pth'
    with open(data_file, 'rb') as f:
        local_data = torch.load(f, weights_only=False)
    # num_samples_client = len(local_data['y'].tolist())
    # label_cnts = collections.Counter(local_data['y'].tolist())
    # label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
    # # clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
    num_samples_client = len(local_data['y'][local_data['train_mask']].tolist())
    label_cnts = collections.Counter(local_data['y'][local_data['train_mask']].tolist())
    label_cnts = dict(sorted(label_cnts.items(), key=lambda x: x[0], reverse=False))
    # clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples_client}
    print(f'client_{CLIENT_IDX} data ({len(label_cnts)}):', label_cnts)
    # print_data(local_data)

    print('Train FNN...')
    # local_fnn = FNN(input_dim=input_dim, hidden_dim=hidden_dim_fnn, output_dim=num_classes)
    local_fnn = FNN(num_classes=NUM_CLASSES).to(DEVICE)
    global_model = FNN(num_classes=NUM_CLASSES).to(DEVICE)
    global_model.load_state_dict(global_model_state_dict)

    if client_type == 'Honest':
        train_cnn(local_fnn, global_model, local_data, train_info)
        # local_train(local_fnn, global_model_state_dict, local_data, train_info)
    elif client_type == 'Byzantine':
        # only inject noise to partial dimensions of model parameters.
        model = FNN(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(global_model_state_dict)
        ps = parameters_to_vector(model.parameters()).detach().to(DEVICE)
        # # Randomly select the indices
        # cnt = max(1, int(CFG.BIG_NUMBER * len(ps)))
        # print(f'{cnt} parameters ({CFG.BIG_NUMBER*100}%) are changed.')

        # selected_indices = random.sample(range(len(ps)), cnt)
        # noise = torch.normal(0, 10, size=(cnt, )).to(DEVICE)
        # ps[selected_indices] = ps[selected_indices] + noise

        noise = torch.normal(0, 10, size=ps.shape).to(DEVICE)
        ps = ps + noise

        vector_to_parameters(ps, model.parameters())  # in_place
        new_state_dict = model.state_dict()

        local_fnn.load_state_dict(new_state_dict)
    else:
        raise ValueError(f'client_type {client_type} is not supported')
    # w = w0 - \eta * \namba_w, so delta_w = w0 - w
    delta_w = {key: global_model_state_dict[key] - local_fnn.state_dict()[key].cpu() for key in global_model_state_dict}
    # clients_fnns[c] = delta_w
    delta_dist = sum([torch.norm(local_fnn.state_dict()[key].cpu() - global_model_state_dict[key].cpu()) for key
                      in global_model_state_dict])
    print(f'dist(local, global): {delta_dist}')

    print('Evaluate FNNs...')
    evaluate(local_fnn, local_data, global_model,
             test_type='Client data', client_id=CLIENT_IDX, train_info=train_info)
    evaluate_shared_test(local_fnn, local_data, global_model,
                         test_type='Shared test data', client_id=CLIENT_IDX, train_info=train_info)
    # local_test()
    # history[c] = train_info

    return delta_w, num_samples_client


def main():
    print(f"[Client {CLIENT_IDX}] Connecting to server at {SERVER_IP}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((SERVER_IP, PORT))
        s.sendall(str(CLIENT_IDX).encode())  # Send client index to server

        while True:
            try:
                # Read first 4 bytes to get the size
                size_data = s.recv(4)
                payload_size = int.from_bytes(size_data, byteorder='big')
                # Read full payload
                received_bytes = b''
                while len(received_bytes) < payload_size:
                    packet = s.recv(min(4096, payload_size - len(received_bytes)))
                    if not packet:
                        break
                    received_bytes += packet

                server_data = pickle.loads(received_bytes)
                if server_data.get("done"):
                    print(f"[Client {CLIENT_IDX}] Server signals training complete. Exiting.")
                    break

                if not received_bytes:
                    print(f"[Client {CLIENT_IDX}] Server closed connection. Exiting.")
                    break

                print(f"[Client {CLIENT_IDX}] Received: {server_data.keys()}")

                # Simulate local training and send update
                updated_local_data = client_train_test(server_data)
                print(f"[Client {CLIENT_IDX}] Sending updated value")

                # s.sendall(pickle.dumps(updated_local_data))
                payload = pickle.dumps(updated_local_data)
                payload_size = len(payload)
                s.sendall(payload_size.to_bytes(4, byteorder='big'))
                s.sendall(payload)

            except Exception as e:
                traceback.print_exc()
                print(f"[Client {CLIENT_IDX}] Error: {e}")
                break

    print(f"[Client {CLIENT_IDX}] Done.")


if __name__ == "__main__":
    main()
