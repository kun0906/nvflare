"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare/auto_labeling
    $PYTHONPATH=. python3 cnn_fl_gan_attention_link_jaccard.py


"""

import argparse
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torchvision import datasets

from attention import aggregate_with_krum
from ragg.utils import timer

print(os.path.abspath(os.getcwd()))

# Check if GPU is available and use it
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Set print options for 2 decimal places
torch.set_printoptions(precision=1, sci_mode=False)


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedCNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--labeling_rate', type=float, required=False, default=0.1,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-l', '--hidden_dimension', type=int, required=False, default=32,
                        help="The hidden dimension of CNN.")
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=30,
                        help="The number of epochs (integer).")
    parser.add_argument('-p', '--patience', type=int, required=False, default=10,
                        help="The patience.")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
LABELING_RATE = args.labeling_rate
SERVER_EPOCHS = args.server_epochs
# hidden_dim_cnn = args.hidden_dimension
# patience = args.patience
# For testing, print the parsed parameters
# print(f"labeling_rate: {LABELING_RATE}")
# print(f"server_epochs: {SERVER_EPOCHS}")
print(args)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        # Fully connected layer to project z into a 128 * 7 * 7 tensor
        self.fc1 = nn.Linear(latent_dim + num_classes, 128 * 7 * 7)

        # Transposed convolutional layers to upsample to 28x28
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 14x14
        self.conv11 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)  #

        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample to 28x28
        self.conv21 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)  #

        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)  # Keep 28x28
        self.conv31 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1)  # Keep 28x28

        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)  # Keep 28x28
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.latent_dim = latent_dim

    def forward(self, z, c, num_layers=1):
        # x: (N, 100), c: (N, 10)
        # x, c = x.view(x.size(0), -1), c.float()  # may not need
        v = torch.cat((z, c), 1)  # v: (N, 110)
        x = self.fc1(v)
        x = x.view(-1, 128, 7, 7)  # Reshape into feature map (batch_size, channels, height, width)

        x = self.leaky_relu(self.conv1(x))  # Upsample to 14x14
        for _ in range(num_layers):
            x = self.leaky_relu(self.conv11(x))  # Upsample to 14x14

        x = self.leaky_relu(self.conv2(x))  # Upsample to 28x28
        for _ in range(num_layers):
            x = self.leaky_relu(self.conv21(x))  # Upsample to 28x28

        x = self.leaky_relu(self.conv3(x))  # Keep 28x28
        for _ in range(num_layers):
            x = self.leaky_relu(self.conv31(x))  # Upsample to 28x28

        x = self.tanh(self.conv4(x))  # Output is in the range [-1, 1]

        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # From 1 channel to 16 channels
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv21 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Adjust the dimensions after the convolution layers
        self.fc2 = nn.Linear(512, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        self.transform = nn.Sequential(
            nn.Linear(28 * 28 + num_classes, 784),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.float()  # may not need
        v = torch.cat((x, c), 1)  # v: (N, 794)
        y_ = self.transform(v)  # (N, 784)
        x = y_.view(y_.shape[0], 1, 28, 28)  # (N, 1, 28, 28)

        # Ensure input has the correct shape (batch_size, 1, 28, 28)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv11(x))

        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv21(x))

        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv31(x))

        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


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


def train_gan(local_gan, global_cgan, local_data, train_info={}):
    print(f'train lcoal gan on client ...')
    # Initialize local_gan with global_gan
    local_gan.load_state_dict(global_cgan.state_dict())

    X, y = local_data['X'], local_data['y']
    mask = local_data['train_mask']
    # only use labeled data for training gan
    X = X[mask]
    y = y[mask]

    onehot_labels = torch.zeros((len(y), len(LABELS))).to(DEVICE)
    for i in LABELS:
        mask = y.numpy() == i
        onehot_labels[mask, i] = 1

    # Only update available local labels, i.e., not all the local_gans will be updated.
    # local_labels = set(y.tolist())
    print(f'local labels: {collections.Counter(y.tolist())}, with {len(y)} samples.')

    generator = local_gan
    generator = generator.to(DEVICE)
    # local_gan.load_state_dict(global_gans[l].state_dict())
    z_dim = generator.latent_dim

    discriminator = Discriminator(num_classes=NUM_CLASSES).to(DEVICE)

    LR = 1e-3
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    scheduler_G = StepLR(optimizer_G, step_size=1000, gamma=0.8)

    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    scheduler_D = StepLR(optimizer_D, step_size=1000, gamma=0.8)
    # optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    # optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    adversarial_loss = nn.BCELoss(reduction='mean')  # Binary Cross-Entropy Loss

    # Training loop
    losses = []
    STEP = 2
    ALPHA = 0.1
    EPOCHs = 5000
    m = 100  # subset
    for epoch in range(EPOCHs):
        # ---- Train Discriminator ----
        discriminator.train()

        indices = torch.randperm(len(X))[:m]  # Randomly shuffle and pick the first 10 indices
        real_data = X[indices].float().to(DEVICE)
        labels = onehot_labels[indices]

        # real_data = X.clone().detach().float().to(DEVICE) / 255  # Replace with your local data (class-specific)
        real_data = real_data.to(DEVICE)
        real_data = real_data.view(-1, 1, 28, 28)  # Ensure shape is (batch_size, 1, 28, 28)

        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

        # Generate synthetic data
        z = torch.randn(batch_size, z_dim).to(DEVICE)
        fake_data = generator(z, labels).detach()  # Freeze Generator when training Discriminator
        # print(fake_data.shape,flush=True)
        # print(epoch, collections.Counter(labels.cpu().numpy().argmax(axis=1).tolist()))

        # Discriminator Loss
        real_loss = adversarial_loss(discriminator(real_data, labels), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_data, labels), fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        scheduler_D.step()

        STEP = 5

        generator.train()
        z = torch.randn(batch_size, z_dim).to(DEVICE)
        generated_data = generator(z, labels)

        # Generator Loss (Discriminator should classify fake data as real)
        g_loss = adversarial_loss(discriminator(generated_data, labels), real_labels)
        mse_loss = torch.nn.functional.mse_loss(generated_data, real_data)
        g_loss += ALPHA * mse_loss

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # print(epoch, d_loss, '...')
        while g_loss > d_loss and STEP > 0:
            # print(epoch, g_loss, d_loss.item(), STEP)
            # ---- Train Generator ----
            # we don't need to freeze the discriminator because the optimizer for the discriminator
            # (optimizer_D) is not called.
            # This ensures that no updates are made to the discriminator's parameters,
            # even if gradients are computed during the backward pass.
            generator.train()
            z = torch.randn(batch_size, z_dim).to(DEVICE)
            generated_data = generator(z, labels)

            # Generator Loss (Discriminator should classify fake data as real)
            g_loss = adversarial_loss(discriminator(generated_data, labels), real_labels)
            mse_loss = torch.nn.functional.mse_loss(generated_data, real_data)
            g_loss += ALPHA * mse_loss

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            STEP -= 1

        scheduler_G.step()

        # ---- Logging ----
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} |  "
                  f"LR_D: {scheduler_D.get_last_lr()}, LR_G: {scheduler_G.get_last_lr()}, "
                  f"mse: {mse_loss.item():.4f}, STEP:{STEP}")
        losses.append(g_loss.item())

        # Display some generated images after each epoch
        if epoch % 100 == 0:

            with torch.no_grad():
                z = torch.randn(100, 100).to(DEVICE)
                c = torch.zeros((100, NUM_CLASSES)).to(DEVICE)
                s = c.shape[0] // NUM_CLASSES
                for i in range(NUM_CLASSES):
                    c[i * s:(i + 1) * s, i] = 1

                generated_imgs = generator(z, c)

                # # Binarization: Threshold the generated image at 0.5 (assuming Tanh output range is [-1, 1])
                # generated_images = (generated_imgs + 1) / 2  # Scale from [-1, 1] to [0, 1]
                # binarized_images = (generated_images > 0.5).float()

                generated_imgs = (generated_imgs + 1) * 127.5  # Convert [-1, 1] back to [0, 255]
                generated_imgs = generated_imgs.clamp(0, 255)  # Ensure values are within [0, 255]
                generated_imgs = generated_imgs.cpu().numpy().astype(int)

                fig, axes = plt.subplots(16, 10, figsize=(8, 8))
                for i, ax in enumerate(axes.flatten()):
                    if i < 100:
                        ax.imshow(generated_imgs[i, 0], cmap='gray')
                    else:
                        ax.imshow(((real_data[i - 100, 0].cpu().numpy() + 1) * 127.5).astype(int), cmap='gray')

                    ax.axis('off')

                    # Draw a red horizontal line across the entire figure when i == 100
                    if i == 100:
                        # Add a red horizontal line spanning the entire width of the plot
                        # This gives the y-position of the 100th image in the figure
                        line_position = (160 - 100 - 2) / 160
                        plt.plot([0, 1], [line_position, line_position], color='red', linewidth=2,
                                 transform=fig.transFigure,
                                 clip_on=False)

                plt.suptitle(f'epoch {epoch}')
                plt.tight_layout()
                fig_file = f'tmp/generated_{epoch}.png'
                dir_path = os.path.dirname(os.path.abspath(fig_file))
                os.makedirs(dir_path, exist_ok=True)
                plt.savefig(fig_file)
                plt.show()

    train_info[f'cgan'] = {"losses": losses}

    local_gan.load_state_dict(generator.state_dict())  # update local_gan again


def gen_local_data(global_cgan, local_data, train_info):
    print_data(local_data)

    # local data
    train_mask = local_data['train_mask']
    local_size = len(local_data['y'])
    y = local_data['y'][train_mask]  # we assume on a tiny labeled data in the local dat
    size = len(y.tolist())
    print(f'client data size: {len(train_mask)}, labeled y: {size}, labeling_rate: {size / local_size:.2f}')
    ct = collections.Counter(y.tolist())
    print(f'labeled y: {ct.items()}')
    print(f"len(ct.keys()):{len(ct.keys())} =? len(LABELS): {len(LABELS)}")

    debug = False
    if not debug and len(ct.keys()) < len(LABELS):
        max_size = max(ct.values())
        # for each class, only generate 10% percent data to save computational resources.
        # if max_size > 100:
        max_size = int(max_size * 0.1)

        if max_size == 0: max_size = 1
        print(f'For each class, we only generate {max_size} samples, '
              f'and use labeled_classes_weights to address class imbalance issue.')
        sizes = {}
        labeled_cnt = {}
        for l in LABELS:
            if l in ct.keys():
                if max_size > ct[l]:
                    sizes[l] = max_size - ct[l]
                    labeled_cnt[l] = max_size
                else:
                    labeled_cnt[l] = ct[l]
            else:
                sizes[l] = max_size
                labeled_cnt[l] = max_size
        print(f'we need to generate samples ({sum(sizes.values())}),where each class sizes:{sizes}')

        # generated new data
        generated_data = gen_data(global_cgan, sizes, similarity_method=None,
                                  local_data=local_data)
        # train_info['generated_size'] = sum(sizes.values())

        # if train_info['server_epoch'] % 10 == 0:
        #     print('check_gen_data...')
        #     check_gen_data(generated_data, local_data)
        # return global_lp
        # append the generated data to the end of local X and Y, not the end of train set
        print('Merge local data and generated data...')
        data = merge_data(generated_data, local_data)

        # generate new edges
        features = data['X']
        labels = data['y']
        train_info['generated_size'] = sum(sizes.values())
        generated_size = sum(sizes.values())

        # debug = False
        # if debug:  # plot the generated data
        #     train_mask = torch.cat(
        #         [local_data['train_mask'], torch.tensor([True] * (sum(sizes.values())), dtype=torch.bool)])
        #     plot_data(data['X'], data['y'], train_mask, generated_size, train_info, local_data, global_vaes)
        #     return global_lp
    else:
        sizes = {}
        labeled_cnt = ct
        features = local_data['X']
        labels = local_data['y']
        data = {}
        train_info['generated_size'] = sum(sizes.values())
        generated_size = sum(sizes.values())
    # train_info['threshold'] = None
    # existed_edge_indices = local_data['edge_indices']
    # local_size = len(local_data['y'])

    print('Update train, val, and test masks based on merged data...')
    # generated_data_indices = list(range(len(train_mask), len(labels), 1))  # append the generated data
    gen_indices = np.arange(train_info['generated_size'])
    gen_train_indices, gen_val_indices = train_test_split(gen_indices, test_size=0.1, shuffle=True,
                                                          random_state=42)
    gen_train_mask = torch.tensor([False] * len(gen_indices), dtype=torch.bool)
    gen_train_mask[gen_train_indices] = True
    train_mask = torch.cat([train_mask, gen_train_mask])

    gen_val_mask = torch.tensor([False] * len(gen_indices), dtype=torch.bool)
    gen_val_mask[gen_val_indices] = True
    val_mask = torch.cat([local_data['val_mask'], gen_val_mask])

    test_mask = torch.cat([local_data['test_mask'], torch.tensor([False] * (sum(sizes.values())), dtype=torch.bool)])

    # Create node features (features from CNN)
    # node_features = torch.tensor(features, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.long)
    node_features = features.clone().detach().float()
    labels = labels.clone().detach().long()
    # Prepare Graph data for PyG (PyTorch Geometric)
    # print('Form graph data...')
    # # Define train, val, and test masks
    # generated_data_indices = list(range(len(labels_mask), len(labels)))
    # # Get indices of y
    # y_indices = labels_mask.nonzero(as_tuple=True)[0].tolist()
    # indices = torch.tensor(y_indices + generated_data_indices).to(DEVICE)
    # Define train_mask and test_mask
    # train_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    # test_mask = torch.tensor([False] * len(labels), dtype=torch.bool)
    # train_mask[indices] = True
    # test_mask[~train_mask] = True
    # val_mask = torch.tensor([False, False, True, False], dtype=torch.bool)
    # test_mask = torch.tensor([False, False, False, True], dtype=torch.bool)
    # graph_data = Data(x=node_features, edge_index=None, edge_weight=None,
    #                   y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # X_train, y_train = node_features[train_mask].cpu().numpy(), labels[train_mask].cpu().numpy()
    # X_val, y_val = node_features[val_mask].cpu().numpy(), labels[val_mask].cpu().numpy()
    # X_test, y_test = node_features[test_mask].cpu().numpy(), labels[test_mask].cpu().numpy()
    # X_shared_test, y_shared_test = X_test, y_test
    # evaluate_ML2(X_train.reshape((X_train.shape[0], -1)), y_train,
    #              X_val.reshape((X_val.shape[0], -1)), y_val,
    #              X_test.reshape((X_test.shape[0], -1)), y_test,
    #              X_shared_test.reshape((X_test.shape[0], -1)), y_shared_test, verbose=10)

    train_info['cnn']['data'] = (node_features, labels, train_mask, val_mask, test_mask)
    # print('Graph_data: ')
    # print(f'\tX_train: {graph_data.x[graph_data.train_mask].shape}, y_train: '
    #       f'{collections.Counter(graph_data.y[graph_data.train_mask].tolist())}, (local data + generated data)')
    # print(f'\tX_val: {graph_data.x[graph_data.val_mask].shape}, y_val: '
    #       f'{collections.Counter(graph_data.y[graph_data.val_mask].tolist())}')
    # print(f'\tX_test: {graph_data.x[graph_data.test_mask].shape}, y_test: '
    #       f'{collections.Counter(graph_data.y[graph_data.test_mask].tolist())}')
    # Use to generate/predict edges between nodes
    # local_lp = CNNLinkPredictor(input_dim, 32)
    # print('Train Link_predictor...')
    # tmp_data = {'X': node_features, 'y': labels, 'edge_indices': edge_indices, 'edge_weight': edge_weight, }
    # train_link_predictor(local_lp, global_lp, tmp_data, train_info)


def train_cnn(local_cnn, global_cnn, train_info={}):
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

    X, y, train_mask, val_mask, test_mask = train_info['cnn']['data']

    print('Compute classes weights...')
    # Get indices of y
    y_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    indices = torch.tensor(y_indices).to(DEVICE)  # total labeled data
    new_y = y[indices]
    labeled_cnt = collections.Counter(new_y.tolist())
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
    epochs_client = 500
    losses = []
    val_losses = []
    best = {'epoch': -1, 'val_accuracy': -1.0, 'val_accs': [], 'val_losses': [], 'train_accs': [], 'train_losses': []}
    val_cnt = 0
    pre_val_loss = 0
    # # here, you need make sure weight aligned with class order.
    class_weight = torch.tensor(list(labeled_classes_weights.values()), dtype=torch.float).to(DEVICE)
    print(f'class_weight: {class_weight}')

    local_cnn = local_cnn.to(DEVICE)
    local_cnn.load_state_dict(global_cnn.state_dict())  # Initialize client_gm with the parameters of global_model
    # optimizer = optim.Adam(local_cnn.parameters(), lr=0.005)
    optimizer = optim.Adam(local_cnn.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    # optimizer = torch.optim.AdamW(local_cnn.parameters(), lr=0.001, weight_decay=5e-4)
    # criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean').to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    # node_features = node_features.view(node_features.shape[0], 1, 28, 28)
    for epoch in range(epochs_client):
        local_cnn.train()  #
        # epoch_model_loss = 0
        # _model_loss, _model_distill_loss = 0, 0
        # epoch_gan_loss = 0
        # _gan_recon_loss, _gan_kl_loss = 0, 0
        # graph_data.to(DEVICE)
        # data_size, data_dim = graph_data.x.shape
        # your local personal model
        outputs = local_cnn(X)
        # Loss calculation: Only for labeled nodes
        model_loss = criterion(outputs, y)
        # if epoch > 0 and model_loss.item() < 1e-6:
        #     break
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

        scheduler.step()  # adjust learning rate

        losses.append(model_loss.item())

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
        if epoch % 100 == 0:
            print(f"train_cnn epoch: {epoch}, local_cnn train loss: {model_loss.item():.4f}, "
                  f"val_loss: {val_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]}")

        # if stop_training:
        #     local_cnn.stop_training = True
        #     print(f'Early Stopping. Epoch: {epoch}, Loss: {model_loss:.4f}')
        #     break

    # exit()
    # train_info['cnn'] = {'graph_data': graph_data, "losses": losses}
    print('***best at epoch: ', best['epoch'], ' best val_accuracy: ', best['val_accuracy'])
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

    return None


def evaluate_train(cnn, graph_data, gen_start, generated_size, epoch, local_data):
    print(f'\n***epoch: ', epoch)

    cnn.eval()
    train_mask, val_mask, test_mask = graph_data.train_mask, graph_data.val_mask, graph_data.test_mask

    X_train, y_train = graph_data.x[train_mask], graph_data.y[train_mask]
    X_val, y_val = graph_data.x[val_mask], graph_data.y[val_mask]
    X_test, y_test = graph_data.x[test_mask], graph_data.y[test_mask]

    gen_mask = torch.tensor([False] * len(train_mask), dtype=torch.bool)
    gen_mask[gen_start:gen_start + generated_size] = True

    with torch.no_grad():
        output = cnn(graph_data)
        _, predicted_labels = torch.max(output, dim=1)
        # for debug purpose
        for data_type, mask_ in [('train', train_mask),
                                 ('val', val_mask),
                                 ('test', test_mask),
                                 ('generated', gen_mask)]:
            true_labels = graph_data.y

            predicted_labels_tmp = predicted_labels[mask_]
            true_labels_tmp = true_labels[mask_]
            y = true_labels_tmp.cpu().numpy()
            y_pred = predicted_labels_tmp.cpu().numpy()

            # Total samples and number of classes
            total_samples = len(y)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y]
            print(f'class_weights: {class_weights}')

            accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
            print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
            conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
            conf_matrix = conf_matrix.astype(int)
            print("Confusion Matrix:\n", conf_matrix)

        # for shared test
        all_data = local_data['all_data']
        # all_indices = all_data['indices']
        # edge_indices = all_data['edge_indices'].to(DEVICE)  # all edge_indices
        X, Y = all_data['X'].to(DEVICE), all_data['y'].to(DEVICE)

        # Shared test data
        shared_test_mask = all_data['test_mask'].to(DEVICE)
        X_shared_test = X[shared_test_mask].to(DEVICE)
        y_shared_test = Y[shared_test_mask].to(DEVICE)
        # X_test_indices = all_indices[shared_test_mask].to(DEVICE)
        print(f'X_test: {X_shared_test.size()}, {collections.Counter(y_shared_test.tolist())}')
        # edge_indices_test = all_data['edge_indices_test'].to(DEVICE)

        new_X = torch.cat((X, X_shared_test), dim=0)
        new_y = torch.cat((Y, y_shared_test), dim=0)

        graph_data = Data(x=new_X, y=new_y, edge_index=None, edge_weight=None)
        graph_data.to(DEVICE)

        output = cnn(graph_data)
        _, predicted_labels = torch.max(output, dim=1)

        true_labels = graph_data.y
        mask_ = torch.tensor([False] * len(true_labels), dtype=torch.bool)
        mask_[len(X):] = True
        predicted_labels_tmp = predicted_labels[mask_]
        true_labels_tmp = true_labels[mask_]
        y = true_labels_tmp.cpu().numpy()
        y_pred = predicted_labels_tmp.cpu().numpy()

        # Total samples and number of classes
        total_samples = len(y)
        # Compute class weights
        class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
        sample_weight = [class_weights[y_0.item()] for y_0 in y]
        print(f'class_weights: {class_weights}')

        data_type = 'shared_test'
        accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
        print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
        conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
        conf_matrix = conf_matrix.astype(int)
        print("Confusion Matrix:\n", conf_matrix)

        if epoch % 10 == 0:
            evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_shared_test, y_shared_test, verbose=10)


def aggregate_gans(clients_gans, clients_info, global_cgan, local_data, histories_server, server_epoch):
    aggregate_method = 'parameter'
    if aggregate_method == 'parameter':  # aggregate clients' parameters
        aggregate_with_krum(clients_gans, clients_info, global_cgan, DEVICE)
        # aggregate_with_attention(client_parameters_list, global_gan, DEVICE)  # update global_gan inplace
    else:
        # train a new model on generated data by clients' gans
        # global_gan.load_state_dict(global_gan.state_dict())
        optimizer = optim.Adam(global_cgan.parameters(), lr=0.01)
        # Define a scheduler
        scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

        losses = []
        for epoch in range(301):
            loss_epoch = 0
            for client_i, local_gan_state_dict in enumerate(clients_gans):
                label_cnts = clients_info[client_i]['label_cnts']
                # Initialize local_gan with global_gan
                local_gan = Generator(latent_dim=200)
                local_gan.load_state_dict(local_gan_state_dict)
                local_gan.to(DEVICE)

                z = []
                ohe_labels = []
                latent_dim = local_gan.decoder.latent_dim
                for l, size in label_cnts.items():
                    # generate latent vector from N(0, 1)
                    z_ = torch.randn(size, latent_dim).to(DEVICE)  # Sample latent vectors
                    ohe_labels_ = torch.zeros((size, len(LABELS))).to(DEVICE)  # One-hot encoding for class labels
                    ohe_labels_[:, l] = 1

                    if len(z) == 0:
                        z = z_.clone().detach()
                        ohe_labels = ohe_labels_.clone().detach()
                    else:
                        z = torch.cat((z, z_), dim=0)
                        ohe_labels = torch.cat((ohe_labels, ohe_labels_), dim=0)

                z = z.to(DEVICE)
                ohe_labels = ohe_labels.to(DEVICE)
                pseudo_logits = local_gan.decoder(z, ohe_labels)  # Reconstruct probabilities from latent space
                X_ = pseudo_logits

                # use the generated data X to train global_gan
                recon_logits, mu, logvar = global_cgan(X_, ohe_labels)
                loss, info = gan_loss_function(recon_logits, X_, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                scheduler.step()

                loss_epoch += loss.item()

            if epoch % 100 == 0:
                print(f'train global gan epoch: {epoch}, global  gan loss: {loss_epoch:.4f}, {info},'
                      f'LR: {scheduler.get_last_lr()}')
            losses.append(loss_epoch)

            # train_info['gan'] = {"losses": losses}

    torch.save(global_cgan.state_dict(), f'global_cgan_{server_epoch}.pth')

    ##########################################################################################################
    # check the generated data
    # sizes = {l: s for l, s in collections.Counter(local_data['shared_data']['y'].tolist()).items()}
    sizes = {l: 100 for l in LABELS}
    print(sizes)
    # generated new data
    generated_data = gen_data(global_cgan, sizes, similarity_method=None,
                              local_data=local_data)

    for l, vs in generated_data.items():
        vs = vs['X']
        vs = (vs + 1) * 127.5  # Convert [-1, 1] back to [0, 255]
        vs = vs.clamp(0, 255)  # Ensure values are within [0, 255]
        vs = vs.cpu().numpy().astype(int)
        plot_img(vs, l, show=True, fig_file=f'tmp/generated_{l}_{server_epoch}~.png')

    # ml_info = check_gen_data(generated_data, local_data)
    # histories_server.append(ml_info)

    # # generated new data
    # local_gans = {}
    # # for client 0, we only use class 0 and 3
    # # for client 1, we only use class 1 and 4
    # # for client 2, we only use class 2 and 5
    # # for client 3, we only use class 6
    # for c, ls in [(0, (0, 3)), (1, (1, 4)), (2, (2, 5)), (3, (6,))]:
    #     for l in ls:
    #         local_gans[l] = gans[c][l]
    #
    # torch.save(local_gans, global_gan_path)
    #
    # generated_data = gen_data(local_gans, sizes, similarity_method='cosine',
    #                           local_data=local_data)
    # ml_info = check_gen_data(generated_data, local_data)
    # histories_server.append(ml_info)


def aggregate_cnns(clients_cnns, clients_info, global_cnn, histories_server, epoch):
    print('*aggregate cnn...')
    aggregate_with_krum(clients_cnns, clients_info, global_cnn, DEVICE)  # update global_cnn inplace


@timer
def evaluate(local_cnn, local_data, DEVICE, global_cnn, test_type='test', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.

        client_result = {'client_gm': client_model.state_dict(), 'logits': None, 'losses': losses, 'info': client_info}
        client_data_ =  (graph_data_, feature_info, client_data_)
    """
    print('---------------------------------------------------------------')
    for model_type, model in [('global', global_cnn), ('local', local_cnn)]:
        # At time t, global model has not been updated yet, however, local_cnn is updated.
        # After training, the model can make predictions for both labeled and unlabeled nodes
        print(f'***Testing {model_type} model on {test_type}...')
        # cnn = local_cnn(input_dim=64, hidden_dim=32, output_dim=10)
        # cnn.load_state_dict(client_result['client_gm'])
        cnn = model
        cnn = cnn.to(DEVICE)

        # graph_data = train_info['cnn']['graph_data'].to(DEVICE)  # graph data
        # train_mask, val_mask, test_mask = graph_data.train_mask, graph_data.val_mask, graph_data.test_mask
        X, Y, train_mask, val_mask, test_mask = train_info['cnn']['data']

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

                # Total samples and number of classes
                total_samples = len(y)
                # Compute class weights
                class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
                sample_weight = [class_weights[y_0.item()] for y_0 in y]
                print(f'class_weights: {class_weights}')

                accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)

                train_info[f'{model_type}_{data_type}_accuracy'] = accuracy
                print(f"Accuracy on {data_type} data (only): {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
                # if 'all' in test_type:
                #     client_result['labeled_accuracy_all'] = accuracy
                # else:
                #     client_result['labeled_accuracy'] = accuracy
                # print(y, y_pred)
                conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
                conf_matrix = conf_matrix.astype(int)
                train_info[f'{model_type}_{data_type}_cm'] = conf_matrix
                print("Confusion Matrix:\n", conf_matrix)

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


@timer
def evaluate_shared_test(local_cnn, local_data, DEVICE, global_cnn, global_lp,
                         test_type='shared_test_data', client_id=0, train_info={}):
    """
        Evaluate how well each client's model performs on the test set.
    """
    print('---------------------------------------------------------------')
    all_data = local_data['shared_data']

    # shared_test
    X_test, y_test = all_data['X'].to(DEVICE), all_data['y'].to(DEVICE)
    X_test = (X_test / 255 - 0.5) * 2
    # X_test_indices = all_indices[shared_test_mask].to(DEVICE)
    print(f'X_test: {X_test.size()}, {collections.Counter(y_test.tolist())}')

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

            # predicted_labels = predicted_labels[graph_data.train_mask]
            # true_labels = true_labels[graph_data.train_mask]
            # predicted_labels = predicted_labels[graph_data.test_mask]
            # true_labels = true_labels[graph_data.test_mask]

            y = true_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()

            # Total samples and number of classes
            total_samples = len(y)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y]
            print(f'class_weights: {class_weights}')

            accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
            train_info[f'{model_type}_shared_accuracy'] = accuracy
            print(f"Accuracy on shared test data: {accuracy * 100:.2f}%, {collections.Counter(y.tolist())}")
            # if 'all' in test_type:
            #     client_result['labeled_accuracy_all'] = accuracy
            # else:
            #     client_result['labeled_accuracy'] = accuracy

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y, y_pred, sample_weight=sample_weight)
            conf_matrix = conf_matrix.astype(int)
            train_info[f'{model_type}_shared_cm'] = conf_matrix
            print("Confusion Matrix:")
            print(conf_matrix)
            # if 'all' in test_type:
            #     client_result['labeled_cm_all'] = conf_matrix
            # else:
            #     client_result['labeled_cm'] = conf_matrix

        print(f"Client {client_id} evaluation on {test_type} Accuracy: {accuracy * 100:.2f}%")

    return


def evaluate_ML(local_cnn, local_data, DEVICE, global_cnn, test_type, client_id, train_info, verbose=10):
    if verbose > 5:
        print('---------------------------------------------------------------')
        print('Evaluate Classical ML on each client...')
    ml_info = {}
    # # Local data without any generated data
    # local_X, local_y = local_data['X'], local_data['y']
    # train_mask, val_mask, test_mask = local_data['train_mask'], local_data['val_mask'], local_data['test_mask']
    # X_train, y_train = local_X[train_mask], local_y[train_mask]
    # X_val, y_val = local_X[val_mask], local_y[val_mask]
    # X_test, y_test = local_X[test_mask], local_y[test_mask]

    # global shared test set
    shared_test_mask = local_data['all_data']['test_mask']
    X_shared_test, y_shared_test = local_data['all_data']['X'][shared_test_mask].numpy(), local_data['all_data']['y'][
        shared_test_mask].numpy()

    # local data with generated data
    graph_data = train_info['cnn']['graph_data']
    X, y = graph_data.x.numpy(), graph_data.y.numpy()
    X_train, y_train = X[graph_data.train_mask], y[graph_data.train_mask]  # local X_train + generated data
    X_val, y_val = X[graph_data.val_mask], y[graph_data.val_mask]  # should be the same as X_val, y_val
    X_test, y_test = X[graph_data.test_mask], y[graph_data.test_mask]  # should be the same as X_test, y_test

    dim = X_train.shape[1]
    num_classes = len(set(y_train))
    if verbose > 5:
        print(f'Number of Features: {dim}, Number of Classes: {num_classes}')
        print(f'\tX_train: {X_train.shape}, y_train: '
              f'{collections.Counter(y_train.tolist())}')
        print(f'\tX_val: {X_val.shape}, y_val: '
              f'{collections.Counter(y_val.tolist())}')
        print(f'\tX_test: {X_test.shape}, y_test: '
              f'{collections.Counter(y_test.tolist())}')

        print(f'\tX_shared_test: {X_shared_test.shape}, y_test: '
              f'{collections.Counter(y_shared_test.tolist())}')

    from sklearn.tree import DecisionTreeClassifier
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    from sklearn.ensemble import GradientBoostingClassifier
    gd = GradientBoostingClassifier(random_state=42)

    from sklearn import svm
    svm = svm.SVC(random_state=42)

    # mlp = MLP(dim, 64, num_classes)
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, 'MLP': mlp}
    clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, }

    # all_data = client_data['all_data']
    # test_mask = all_data['test_mask']
    # X_shared_test = all_data['X'][test_mask]
    # y_shared_test = all_data['y'][test_mask]
    for clf_name, clf in clfs.items():
        if verbose > 5:
            print(f"\nTraining {clf_name}")
        # Train the classifier on the training data
        if clf_name == 'MLP':
            clf.fit(X_train, y_train, X_val, y_val)
        else:
            clf.fit(X_train, y_train)
        if verbose > 5:
            print(f"Testing {clf_name}")
        for test_type, X_, y_ in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test),
                                  ('shared_test', X_shared_test, y_shared_test)]:
            if verbose > 5:
                print(f'Testing on {test_type}')
            # Make predictions on the data
            y_pred_ = clf.predict(X_)

            # Total samples and number of classes
            total_samples = len(y_)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y_]
            print(f'class_weights: {class_weights}')

            # Calculate accuracy
            accuracy = accuracy_score(y_, y_pred_, sample_weight=sample_weight)
            if verbose > 5:
                print(f"Accuracy of {clf_name}: {accuracy * 100:.2f}%")
            # Compute confusion matrix
            cm = confusion_matrix(y_, y_pred_, sample_weight=sample_weight)
            cm = cm.astype(int)
            if verbose > 5:
                print(cm)
            ml_info[clf_name] = {test_type: {'accuracy': accuracy, 'cm': cm}}
    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()

    train_info['ml_info'] = ml_info


def evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test,
                 X_shared_test, y_shared_test, verbose=10):
    if verbose > 5:
        print('---------------------------------------------------------------')
        print('Evaluate Classical ML on each client...')
    ml_info = {}

    dim = X_train.shape[1]
    num_classes = len(set(y_train))
    if verbose > 5:
        print(f'Number of Features: {dim}, Number of Classes: {num_classes}')
        print(f'\tX_train: {X_train.shape}, y_train: '
              f'{collections.Counter(y_train.tolist())}')
        print(f'\tX_val: {X_val.shape}, y_val: '
              f'{collections.Counter(y_val.tolist())}')
        print(f'\tX_test: {X_test.shape}, y_test: '
              f'{collections.Counter(y_test.tolist())}')

        print(f'\tX_shared_test: {X_shared_test.shape}, y_shared_test: '
              f'{collections.Counter(y_shared_test.tolist())}')

        print(f'Total (without X_shared_val): X_train + X_val + X_test + X_shared_test = '
              f'{X_train.shape[0] + X_val.shape[0] + X_test.shape[0] + X_shared_test.shape[0]}')

    from sklearn.tree import DecisionTreeClassifier
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    from sklearn.ensemble import GradientBoostingClassifier
    gd = GradientBoostingClassifier(random_state=42)

    from sklearn import svm
    svm = svm.SVC(random_state=42)

    # mlp = MLP(dim, 64, num_classes)
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm, 'MLP': mlp}
    clfs = {'Random Forest': rf}

    # all_data = client_data['all_data']
    # test_mask = all_data['test_mask']
    # X_shared_test = all_data['X'][test_mask]
    # y_shared_test = all_data['y'][test_mask]
    for clf_name, clf in clfs.items():
        if verbose > 5:
            print(f"\nTraining {clf_name}")
        # Train the classifier on the training data
        if clf_name == 'MLP':
            clf.fit(X_train, y_train, X_val, y_val)
        else:
            clf.fit(X_train, y_train)
        if verbose > 5:
            print(f"Testing {clf_name}")
        ml_info[clf_name] = {}
        for test_type, X_, y_ in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test),
                                  ('shared_test', X_shared_test, y_shared_test)
                                  ]:
            if verbose > 5:
                print(f'Testing on {test_type}')
            # Make predictions on the data
            y_pred_ = clf.predict(X_)

            # Total samples and number of classes
            total_samples = len(y_)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y_]
            print(f'class_weights: {class_weights}')

            # Calculate accuracy
            accuracy = accuracy_score(y_, y_pred_, sample_weight=sample_weight)
            if verbose > 5:
                print(f"Accuracy of {clf_name}: {accuracy * 100:.2f}%")
            # Compute confusion matrix
            cm = confusion_matrix(y_, y_pred_, sample_weight=sample_weight)
            cm = cm.astype(int)
            if verbose > 5:
                print(cm)
            ml_info[clf_name][test_type] = {'accuracy': accuracy, 'cm': cm}
    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()

    return ml_info


def check_gen_data(generated_data, local_data):
    X = local_data['all_data']['X']
    y = local_data['all_data']['y']
    train_mask = local_data['all_data']['train_mask']
    val_mask = local_data['all_data']['val_mask']
    test_mask = local_data['all_data']['test_mask']

    # build classifier with true data
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # test on the generated data
    dim = X_train.shape[1]
    X_gen_test = np.zeros((0, dim))
    y_gen_test = np.zeros((0,), dtype=int)

    for l, vs in generated_data.items():
        X_gen_test = np.concatenate((X_gen_test, vs['X'].cpu()), axis=0)
        y_gen_test = np.concatenate((y_gen_test, vs['y']))

    print('X_train, y_train as training set')
    ml_info = evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_gen_test, y_gen_test, verbose=10)

    print('X_gen_test, y_gen_test as training set')
    ml_info2 = evaluate_ML2(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test, verbose=10)

    return ml_info


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
    elif method == 'gan':
        generator = model
        generator.eval()
        z_dim = generator.latent_dim
        with torch.no_grad():
            z = torch.randn(size, z_dim).to(DEVICE)
            synthetic_data = generator(z)
        embedding = synthetic_data
    elif method == 'cgan':
        generator = model
        generator.eval()
        z_dim = generator.latent_dim
        labels = torch.zeros((size, NUM_CLASSES)).to(DEVICE)
        labels[:, l] = 1
        with torch.no_grad():
            z = torch.randn(size, z_dim).to(DEVICE)
            synthetic_data = generator(z, labels)
            synthetic_data = synthetic_data.squeeze(1)  # Removes the second dimension (size 1)
        embedding = synthetic_data

    else:  # default one is autoencoder
        vae = model
        latent_dim = vae.latent_dim
        # generate latent vector from N(0, 1)
        z = torch.randn(size, latent_dim).to(DEVICE)  # Sample latent vectors
        # ohe_labels = torch.zeros((size, len(LABELS))).to(DEVICE)  # One-hot encoding for class labels
        # ohe_labels[:, l] = 1
        pseudo_logits = vae.decoder(z)  # Reconstruct probabilities from latent space
        embedding = pseudo_logits

    return embedding


def compare_gen_true_1d(generated_1d, true_1d):
    # # Resample the larger dataset to match the smaller dataset size
    # if len(gen_1d) > len(true_1d):
    #     gen_1d = np.random.choice(gen_1d, size=len(true_1d), replace=False)
    # else:
    #     true_1d = np.random.choice(true_1d, size=len(gen_1d), replace=False)

    from scipy.stats import ks_2samp

    # Example: Compare two datasets
    stat, p_value = ks_2samp(generated_1d, true_1d)
    if p_value < 0.05:
        # print("The datasets are from different distributions (reject null hypothesis).")
        flg = False
    else:
        # print("The datasets are likely from the same distribution.")
        flg = True

    from scipy.stats import entropy

    # Convert datasets into probability distributions
    hist1, _ = np.histogram(generated_1d, bins=20, density=True)
    hist2, _ = np.histogram(true_1d, bins=20, density=True)

    # Compute KL Divergence
    kl_div = entropy(hist1, hist2)
    # print(f"KL Divergence: {kl_div}")
    return flg


def compare_gen_true(generated_data, true_data):
    flgs = [compare_gen_true_1d(generated_data[:, d].cpu(), true_data[:, d]) for d in range(generated_data.shape[1])]
    res = collections.Counter(flgs)
    same = res[True]
    diff = res[False]
    print(f'same distribution: {same}, different distribution:{diff}')
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # # Visualization
    # sns.kdeplot(data1, label="Data1", fill=True)
    # sns.kdeplot(data2, label="Data2", fill=True)
    # plt.legend()
    # plt.show()


def plot_img(generated_imgs, l, show=True, fig_file=f'tmp/generated~.png'):
    if show:
        # real_data = X_train[y_train == l]
        fig, axes = plt.subplots(16, 10, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < 100:
                ax.imshow(generated_imgs[i], cmap='gray')
            # else:
            #     ax.imshow((real_data[i - 100]).astype(int), cmap='gray')
            ax.axis('off')
            # Draw a red horizontal line across the entire figure when i == 100
            if i == 100:
                # Add a red horizontal line spanning the entire width of the plot
                # This gives the y-position of the 100th image in the figure
                line_position = (160 - 100 - 2) / 160
                plt.plot([0, 1], [line_position, line_position], color='red', linewidth=2,
                         transform=fig.transFigure,
                         clip_on=False)
        plt.suptitle(f'Generated class {l}')
        plt.tight_layout()
        dir_path = os.path.dirname(os.path.abspath(fig_file))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(fig_file)
        plt.show()


def gen_data(cgan, sizes, similarity_method='cosine', local_data={}):
    data = {}
    for l, size in sizes.items():
        gan = cgan
        gan.to(DEVICE)
        pseudo_logits = _gen_models(gan, l, size, method='cgan')
        pseudo_logits = pseudo_logits.detach().to(DEVICE)

        # plot_img(pseudo_logits, l, show=True)

        features = pseudo_logits
        # features = F.sigmoid(pseudo_logits)
        # if similarity_method == 'cosine':
        #     mask = features > 0.5
        #     features[mask] = 1
        #     features[~mask] = 0

        data[l] = {'X': features, 'y': [l] * size}
        print(f'Generated data {features.cpu().numpy().shape} range for class {l}: '
              f'min: {min(features.cpu().numpy().flatten())}, '
              f'max: {max(features.cpu().numpy().flatten())}')

        # print(f'Generated data {features.cpu().numpy().shape} range for class {l}: '
        #       f'mean: {torch.mean(features, dim=0)}, '
        #       f'std: {torch.std(features, dim=0)}')
        # print(f'Generated class {l}:')
        # print_histgram(pseudo_logits.detach().numpy())

        # mask = local_data['all_data']['y'] == l
        # true_data = local_data['all_data']['X'][mask]
        # compare_gen_true(features, true_data)

    return data


def merge_data(data, local_data):
    new_data = {'X': (local_data['X'].to(DEVICE) / 255 - 0.5) * 2,  # [-1, 1]
                'y': local_data['y'].to(DEVICE),
                'is_generated': torch.tensor(len(local_data['y']) * [False]).to(
                    DEVICE)}  # Start with None for concatenation
    # tmp = {}
    for l, vs in data.items():
        size = len(vs['y'])
        new_data['X'] = torch.cat((new_data['X'], vs['X']), dim=0)
        new_data['y'] = torch.cat((new_data['y'], torch.tensor(vs['y'], dtype=torch.long).to(DEVICE)))
        new_data['is_generated'] = torch.cat((new_data['is_generated'], torch.tensor(size * [True]).to(DEVICE)))
    return new_data


def plot_data(X, y, train_mask, gen_size, train_info={}, local_data={}, global_vaes=[], X_test=None, y_test=None):
    fig, axes = plt.subplots(2, 2)

    ###############################################################################################
    from sklearn.decomposition import PCA
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    # X_ = X[train_mask]
    # y_ = y[train_mask]
    X_ = X[train_mask].cpu().numpy()
    y_ = y[train_mask].cpu().numpy()
    X_2d = pca.fit_transform(X_)

    X_train_2d = X_2d[:-gen_size]
    X_gen_2d = X_2d[-gen_size:]
    y_train = y_[:-gen_size]
    y_gen = y_[-gen_size:]

    # Plot training data
    classes = collections.Counter(y_)
    for label in np.unique(y_):
        if label in np.unique(y_train):
            if label in np.unique(y_gen):  # only part of data is generated.
                axes[0, 0].scatter(
                    np.concatenate([X_train_2d[y_train == label, 0], X_gen_2d[y_gen == label, 0]], axis=0),
                    np.concatenate([X_train_2d[y_train == label, 1], X_gen_2d[y_gen == label, 1]], axis=0),
                    label=f'Train Class {label}:{classes[label]}',
                    marker='o',
                    alpha=0.7
                )
            else:
                axes[0, 0].scatter(
                    X_train_2d[y_train == label, 0],
                    X_train_2d[y_train == label, 1],
                    label=f'Train Class {label}:{classes[label]}',
                    marker='o',
                    alpha=0.7
                )
        else:
            axes[0, 0].scatter(
                X_gen_2d[y_gen == label, 0],
                X_gen_2d[y_gen == label, 1],
                label=f'Gen Class {label}:{classes[label]}',
                marker='x',
                alpha=0.7
            )

    # # Plot training data
    # classes = collections.Counter(y_train)
    # for label in np.unique(y_train):
    #     axes[0, 0].scatter(
    #         X_train_2d[y_train == label, 0],
    #         X_train_2d[y_train == label, 1],
    #         label=f'Train Class {label}:{classes[label]}',
    #         marker='o',
    #         alpha=0.7
    #     )
    #
    # # Plot generated data
    # classes = collections.Counter(y_gen)
    # for label in np.unique(y_gen):
    #     axes[0, 0].scatter(
    #         X_gen_2d[y_gen == label, 0],
    #         X_gen_2d[y_gen == label, 1],
    #         label=f'Generated Class {label}:{classes[label]}',
    #         marker='x',
    #         alpha=0.7
    #     )

    if X_test is not None:
        # Project test data using the fitted PCA
        X_test_2d = pca.transform(X_test)

        # Plot test data
        classes = collections.Counter(y_test)
        for label in np.unique(y_test):
            axes[0, 0].scatter(
                X_test_2d[y_test == label, 0],
                X_test_2d[y_test == label, 1],
                label=f'Test Class {label}:{classes[label]}',
                marker='s',
                alpha=0.7
            )

    client_id = train_info['client_id']
    server_epoch = train_info['server_epoch']

    # axes[0, 0].set_xlabel('Principal Component 1')
    # axes[0, 0].set_ylabel('Principal Component 2')
    axes[0, 0].set_title(f'local + generated')
    axes[0, 0].legend(fontsize=5)

    ###############################################################################################
    # only generated data
    test_mask = local_data['all_data']['test_mask']
    sizes = {l: s for l, s in collections.Counter(local_data['all_data']['y'][test_mask].tolist()).items()}
    print(sizes)
    # generated new data
    generated_data = gen_data(global_vaes, sizes, similarity_method='cosine',
                              local_data=local_data)
    # test on the generated data
    dim = X.shape[1]
    X_gen_test = np.zeros((0, dim))
    y_gen_test = np.zeros((0,), dtype=int)
    for l, vs in generated_data.items():
        X_gen_test = np.concatenate((X_gen_test, vs['X'].cpu()), axis=0)
        y_gen_test = np.concatenate((y_gen_test, vs['y']))

    pca = PCA(n_components=2)
    X = X_gen_test
    y = y_gen_test
    X_2d = pca.fit_transform(X)

    # Plot training data
    classes = collections.Counter(y)
    for label in np.unique(y):
        axes[0, 1].scatter(
            X_2d[y == label, 0],
            X_2d[y == label, 1],
            label=f'Class {label}:{classes[label]}',
            marker='o',
            alpha=0.7
        )

    # axes[0, 1].set_xlabel('Principal Component 1')
    # axes[0, 1].set_ylabel('Principal Component 2')
    axes[0, 1].set_title(f'All generated data')
    axes[0, 1].legend(fontsize=5)

    ###############################################################################################
    all_data = local_data['all_data']
    shared_test_mask = all_data['test_mask']
    X = all_data['X'][shared_test_mask].numpy()
    y = all_data['y'][shared_test_mask].numpy()
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)

    X_2d = pca.fit_transform(X)

    # Plot training data
    classes = collections.Counter(y)
    for label in np.unique(y):
        axes[1, 0].scatter(
            X_2d[y == label, 0],
            X_2d[y == label, 1],
            label=f'Class {label}:{classes[label]}',
            marker='o',
            alpha=0.7
        )

    # axes[1, 0].set_xlabel('Principal Component 1')
    # axes[1, 0].set_ylabel('Principal Component 2')
    axes[1, 0].set_title(f'true test data')
    axes[1, 0].legend(fontsize=5)

    ###############################################################################################
    all_data = local_data['all_data']
    # shared_test_mask = all_data['test_mask']
    X = all_data['X'].numpy()
    y = all_data['y'].numpy()
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)

    X_2d = pca.fit_transform(X)

    # Plot training data
    classes = collections.Counter(y)
    for label in np.unique(y):
        axes[1, 1].scatter(
            X_2d[y == label, 0],
            X_2d[y == label, 1],
            label=f'Class {label}:{classes[label]}',
            marker='o',
            alpha=0.7
        )

    # axes[1, 1].set_xlabel('Principal Component 1')
    # axes[1, 1].set_ylabel('Principal Component 2')
    axes[1, 1].set_title(f'All true data')
    axes[1, 1].legend(fontsize=5)

    plt.suptitle(f'epoch:{server_epoch}, client_{client_id}')
    plt.tight_layout()
    # plt.grid(True)
    fig_file = f'{IN_DIR}/plots/client_{client_id}/epoch_{server_epoch}.png'
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
            # Total samples and number of classes
            total_samples = len(y_val)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_val.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y_val]
            # print(f'class_weights: {class_weights}')

            outputs_ = model(X_val)
            val_loss = criterion(outputs_, y_val)
            val_accuracy = accuracy_score(y_val, np.argmax(outputs_, axis=1), sample_weight=sample_weight)
        else:
            data = X_val  # here must be graph data
            outputs_ = model(data)
            _, predicted_labels = torch.max(outputs_, dim=1)
            # Loss calculation: Only for labeled nodes
            val_loss = criterion(outputs_[data.val_mask], data.y[data.val_mask])

            # Total samples and number of classes
            y_val = data.y[data.val_mask]
            total_samples = len(y_val)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_val.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y_val]
            # print(f'class_weights: {class_weights}')

            val_accuracy = accuracy_score(data.y[data.val_mask].tolist(),
                                          predicted_labels[data.val_mask].tolist(), sample_weight=sample_weight)

            train_loss = criterion(outputs_[data.train_mask], data.y[data.train_mask])

            # Total samples and number of classes
            y_train = data.y[data.train_mask]
            total_samples = len(y_train)
            # Compute class weights
            class_weights = {c: total_samples / count for c, count in collections.Counter(y_train.tolist()).items()}
            sample_weight = [class_weights[y_0.item()] for y_0 in y_train]
            # print(f'class_weights: {class_weights}')

            train_accuracy = accuracy_score(data.y[data.train_mask].tolist(),
                                            predicted_labels[data.train_mask].tolist(), sample_weight=sample_weight)
            # print(f"epoch: {epoch} Accuracy on val data: {accuracy * 100:.2f}%")

    best['train_accs'].append(train_accuracy)
    best['train_losses'].append(train_loss.cpu())

    best['val_accs'].append(val_accuracy)
    best['val_losses'].append(val_loss.cpu())

    if best['val_accuracy'] < val_accuracy:
        best['model'] = model.state_dict()
        best['epoch'] = epoch
        best['val_loss'] = val_loss
        best['val_accuracy'] = val_accuracy
        best['train_accuracy'] = train_accuracy

    if epoch == 0:
        pre_val_loss = val_loss
        return val_loss, pre_val_loss, val_cnt, stop_training

    if val_loss < pre_val_loss:
        pre_val_loss = val_loss
        val_cnt = 0

        if best['val_accuracy'] <= val_accuracy:  # note here is <=  not =
            best['model'] = model.state_dict()
            best['epoch'] = epoch
            best['val_loss'] = val_loss
            best['val_accuracy'] = val_accuracy
            best['train_accuracy'] = train_accuracy

    else:  # if val_loss > pre_val_loss, it means we should consider early stopping.
        val_cnt += 1
        if val_cnt >= patience:
            # training stops.
            stop_training = True
    return val_loss, pre_val_loss, val_cnt, stop_training


# gan loss function
def gan_loss_function(recon_x, x, mean, log_var, beta=0):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / (x.shape[0] * x.shape[1]) == reduction='mean'
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]
    # BCE = F.mse_loss(recon_x, x, reduction='mean')
    # KLD loss for the latent space
    # This assumes a unit Gaussian prior for the latent space
    # (Normal distribution prior, mean 0, std 1)
    # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # KL divergence between q(z|x) and p(z) (standard normal)
    # Normalized by the batch size for stability
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # This is the regularization term that encourages meaningful latent space
    # (also known as the latent space regularization term)
    # When beta > 1, the model is encouraged to use the latent space more effectively
    # It's controlled by the beta value in β-VAE
    # Latent loss (KL divergence)
    # You can adjust this term using 'beta' to scale the importance of the latent space regularization
    # The larger the beta, the more emphasis on KL divergence
    # If beta is too large, the model might ignore reconstruction and over-regularize
    # If beta is too small, the model might ignore latent space regularization
    # Hence, a reasonable balance is required.
    KLD = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())) / x.shape[0]
    return (BCE + beta * KLD), {'BCE': BCE.item(), 'KLD': KLD.item()}


def print_histgram(new_probs, value_type='probs'):
    print(f'***Print histgram of {value_type}, min:{min(new_probs)}, max: {max(new_probs)}***')
    # # Convert the probabilities to numpy for histogram calculation
    # new_probs = new_probs.detach().cpu().numpy()
    # Compute histogram
    hist, bin_edges = torch.histogram(torch.tensor(new_probs), bins=5)
    # Print histogram
    for i in range(len(hist)):
        print(f"\tBin {i}: {value_type} Range ({bin_edges[i]}, {bin_edges[i + 1]}), Frequency: {hist[i]}")


def print_histories(histories):
    num_server_epoches = len(histories)
    num_clients = len(histories[0])
    num_classes = len(histories[0][0]["gan"])
    print('num_server_epoches:', num_server_epoches, ' num_clients:', num_clients, ' num_classes:', num_classes)
    # for c in range(num_clients):
    #     print(f"\n\nclient {c}")
    #     for s in range(num_server_epoches):
    #         client = histories[s][c]
    #         local_gan = client['gan']
    #         local_cnn = client['cnn']
    #         print(f'\t*local gan:', local_gan.keys(), f' server_epoch: {s}')
    #         losses_ = [float(f"{v:.2f}") for v in local_gan['losses']]
    #         # print(f'\t\tlocal gan ({len(losses_)}): {losses_[:5]} ... {losses_[-5:]}')
    #         print(f'\tlocal gan ({len(losses_)}): [{", ".join(map(str, losses_[:5]))}, ..., '
    #               f'{", ".join(map(str, losses_[-5:]))}]')
    #         # print('\t*local cnn:', [f"{v:.2f}" for v in local_cnn['losses']])
    #         # labeled_acc = client['labeled_accuracy']
    #         # unlabeled_acc = client['unlabeled_accuracy']
    #         # shared_acc = client['shared_accuracy']
    #         # print(f'\t\tlabeled_acc:{labeled_acc:.2f}, unlabeled_acc:{unlabeled_acc:.2f},
    #         # shared_acc:{shared_acc:.2f}')

    c = num_clients
    for model_type in ['global', 'local']:
        print(f'\n***model_type: {model_type}***')
        ncols = 2
        nrows, r = divmod(c, ncols)
        nrows = nrows if r == 0 else nrows + 1
        fig, axes = plt.subplots(nrows, ncols)
        for c in range(num_clients):
            i, j = divmod(c, ncols)
            print(f"\nclient {c}")
            train_accs = []  # train
            val_accs = []
            unlabeled_accs = []  # test
            shared_accs = []
            for s in range(num_server_epoches):
                client = histories[s][c]
                # local_gan = client['gan']
                # local_cnn = client['cnn']
                # print(f'\t*local gan:', local_gan.keys(), f' server_epoch: {s}')
                # losses_ = [float(f"{v:.2f}") for v in local_gan['losses']]
                # print(f'\t\tlocal gan:', losses_)
                # # print('\t*local cnn:', [f"{v:.2f}" for v in local_cnn['losses']])
                train_acc = client[f'{model_type}_train_accuracy']
                val_acc = client[f'{model_type}_val_accuracy']
                test_acc = client[f'{model_type}_test_accuracy']
                shared_acc = client[f'{model_type}_shared_accuracy']
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
            axes[i, j].set_title(f'Client_{c}')
            axes[i, j].legend(fontsize='small')

        if model_type == 'global':
            title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}' + '}$' + f':{LABELING_RATE}'
        else:
            title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' + f':{LABELING_RATE}'
        plt.suptitle(title)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig_file = f'{IN_DIR}/{LABELING_RATE}/{model_type}_accuracy.png'
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300)
        plt.show()


def print_data(local_data):
    print('Local_data: ')
    X, y = local_data['X'], local_data['y']
    print(f'X: {X.shape}, y: '
          f'{collections.Counter(y.tolist())}, in which, ')
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
            axes[i, j].legend(fontsize='small')

        if model_type == 'global':
            title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}' + '}$' + f':{LABELING_RATE}'
        else:
            title = f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' + f':{LABELING_RATE}'
        plt.suptitle(title)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        fig_file = f'{IN_DIR}/{LABELING_RATE}/{model_type}_accuracy.png'
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300)
        plt.show()


def normalize(X):
    return (X / 255.0 - 0.5) * 2  # [-1, 1]


def clients_training(epoch, global_cgan, global_cnn):
    # update clients
    clients_gans = {}
    clients_cnns = {}
    clients_info = {}  # extra information (e.g., number of samples) of clients that can be used in aggregation
    history = {}

    train_dataset = datasets.MNIST(root="./data", train=True, transform=None, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.tolist()
    shared_data = {"X": torch.tensor(normalize(X_test)).float(), 'y': torch.tensor(y_test)}

    # Extract data and targets
    X = train_dataset.data  # Tensor of shape (60000, 28, 28)
    y = train_dataset.targets  # Tensor of shape (60000,)
    X = normalize(X.numpy())  # [-1, 1]
    y = y.numpy()

    for l in LABELS:
        # in each four clients, the first three are honest clients, and the last one is Byzantine
        mask = y == l
        X_label, y_label = X[mask], y[mask]
        # each client has s images
        m = X_label.shape[0]
        n_honest_clients_in_each_group = 2
        n_attackers_in_each_group = 1
        n_clients_in_each_group = n_honest_clients_in_each_group + n_attackers_in_each_group
        s = m // n_honest_clients_in_each_group

        random_state = 42 * l
        torch.manual_seed(random_state)
        indices = torch.randperm(m)  # Randomly shuffle
        for i in range(
                n_clients_in_each_group):  # in each 4 clients, the first 3 are honest clients and the last one is Byzantine
            client_id = l * n_clients_in_each_group + i
            c = client_id
            client_type = 'Honest' if i < n_honest_clients_in_each_group else 'Byzantine'
            print(f"\n\n***server_epoch:{epoch}, client_{c}: {client_type}...")
            # might be used in server
            train_info = {"client_type": client_type, "gan": {}, "cnn": {}, 'client_id': c, 'server_epoch': epoch}
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Load data...')
            if i < n_honest_clients_in_each_group:
                # client_data_file = f'{IN_DIR}/c_{c}-{prefix}-data.pth'
                # local_data = torch.load(client_data_file, weights_only=True)

                indices_sub = indices[i * s:(i + 1) * s]  # and pick the first 10 indices
                X_sub = X_label[indices_sub]
                y_sub = y_label[indices_sub]
                num_samples = len(X_sub)

                # Create indices for train/test split
                indices_sub = np.arange(num_samples)
                train_indices, test_indices = train_test_split(indices_sub, test_size=1 - LABELING_RATE,
                                                               shuffle=True, random_state=random_state)
                train_indices, val_indices = train_test_split(train_indices, test_size=0.1, shuffle=True,
                                                              random_state=random_state)
                train_mask = np.full(num_samples, False)
                val_mask = np.full(num_samples, False)
                test_mask = np.full(num_samples, False)
                train_mask[train_indices] = True
                val_mask[val_indices] = True
                test_mask[test_indices] = True

                local_data = {'client_type': client_type,
                              'X': torch.tensor(X_sub), 'y': torch.tensor(y_sub),
                              'train_mask': torch.tensor(train_mask, dtype=torch.bool),
                              'val_mask': torch.tensor(val_mask, dtype=torch.bool),
                              'test_mask': torch.tensor(test_mask, dtype=torch.bool),
                              'shared_data': shared_data}

                label_cnts = collections.Counter(local_data['y'].tolist())
                clients_info[c] = {'label_cnts': label_cnts, 'size': num_samples}
                print(f'client_{c} data:', label_cnts)
                print_data(local_data)

                #####################################################################
                if epoch % GAN_UPDATE_FREQUENCY == 0 and GAN_UPDATE_FLG:
                    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    local_cgan = Generator(latent_dim=100, num_classes=NUM_CLASSES)
                    print('Train cgan...')
                    train_gan(local_cgan, global_cgan, local_data, train_info)
                    clients_gans[c] = local_cgan.state_dict()

                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                gen_local_data(global_cgan, local_data, train_info)
                print('Train CNN...')
                # local_cnn = CNN(input_dim=input_dim, hidden_dim=hidden_dim_cnn, output_dim=num_classes)
                local_cnn = CNN(num_classes=NUM_CLASSES)
                train_cnn(local_cnn, global_cnn, train_info)
                clients_cnns[c] = local_cnn.state_dict()

            else:
                # for each 4 clients, the last one is Byzantine. attackers don't need to train local_cnn
                # local_data = {'X': [], 'y': []}
                local_data = {'client_type': client_type,
                              'X': torch.tensor(X_sub) + 1000, 'y': torch.tensor(y_sub),
                              'train_mask': torch.tensor(train_mask, dtype=torch.bool),
                              'val_mask': torch.tensor(val_mask, dtype=torch.bool),
                              'test_mask': torch.tensor(test_mask, dtype=torch.bool),
                              'shared_data': shared_data}
                label_cnts = collections.Counter(local_data['y'].tolist())
                clients_info[c] = {'label_cnts': label_cnts, 'size': len(local_data['y'])}
                local_cnn = CNN(num_classes=NUM_CLASSES).to(DEVICE)
                # Assign large values to all parameters
                BIG_NUMBER = 1e6  # Example: Set all weights and biases to 1,000,000
                for param in local_cnn.parameters():
                    param.data.fill_(BIG_NUMBER)  # Assign big number to each parameter
                clients_cnns[c] = local_cnn.state_dict()
                train_info['cnn']['data'] = (local_data['X'].float().to(DEVICE), local_data['y'].to(DEVICE),
                                             local_data["train_mask"], local_data["val_mask"],
                                             local_data["test_mask"])

            print('Evaluate CNNs...')
            evaluate(local_cnn, None, DEVICE, global_cnn,
                     test_type='Client data', client_id=c, train_info=train_info)
            evaluate_shared_test(local_cnn, local_data, DEVICE, global_cnn, None,
                                 test_type='Shared test data', client_id=c, train_info=train_info)

            # if epoch % 100 == 0 or epoch+1 == SERVER_EPOCHS:
            #     evaluate_ML(local_cnn, local_data, DEVICE, global_cnn,
            #                 test_type='Classical ML', client_id=c, train_info=train_info)

            history[c] = train_info

    return clients_gans, clients_cnns, clients_info, history


@timer
def main():
    print(f"\n***************************** Global Models *************************************")
    global_cgan_path = '../cgan_10000_0.5_1_0.0001.pt'
    if os.path.exists(global_cgan_path):
        print(f'load {global_cgan_path}...')
        global_cgan = torch.load(global_cgan_path, weights_only=False,
                                 map_location=torch.device(DEVICE))
    else:
        global_cgan = Generator(latent_dim=100, num_classes=NUM_CLASSES)
    print(global_cgan)

    # global_cnn = CNN(input_dim=input_dim, hidden_dim=hidden_dim_cnn, output_dim=num_classes)
    global_cnn = CNN(num_classes=NUM_CLASSES)
    print(global_cnn)

    debug = True
    if debug:
        histories = {'clients': [], 'server': []}
        for epoch in range(SERVER_EPOCHS):

            print(f"\n***************************** {epoch}: Client Training *************************************")
            clients_gans, clients_cnns, clients_info, history = clients_training(epoch, global_cgan, global_cnn)
            histories['clients'].append(history)

            print(f"\n***************************** {epoch}: Sever Aggregation *************************************")
            # if not os.path.exists(global_cgan_path):
            if epoch % GAN_UPDATE_FREQUENCY == 0 and GAN_UPDATE_FLG:
                aggregate_gans(clients_gans, clients_info, global_cgan, None, histories['server'], epoch)

            aggregate_cnns(clients_cnns, clients_info, global_cnn, histories['server'], epoch)

    prefix = f'-n_{SERVER_EPOCHS}'
    history_file = f'{IN_DIR}/histories_gan_{prefix}.pth'
    print(f'saving histories to {history_file}')
    # with open(history_file, 'wb') as f:
    #     pickle.dump(histories, f)
    torch.save(histories, history_file)

    try:
        print_histories(histories['clients'])
    except Exception as e:
        print('error: ', e)
    print_histories_server(histories['server'])


if __name__ == '__main__':
    IN_DIR = '../fl/mnist'
    LABELS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    # LABELS = {0, 1}
    NUM_CLASSES = len(LABELS)
    GAN_UPDATE_FREQUENCY = 10
    GAN_UPDATE_FLG = True
    print(f'IN_DIR: {IN_DIR}, '
          f'num_classes: {NUM_CLASSES}, where classes: {LABELS}')

    main()
