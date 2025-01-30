"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare/auto_labeling
    $PYTHONPATH=. python3 cgan_fl_generate.py


"""
import argparse
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

from utils import timer

print(os.path.abspath(os.getcwd()))

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set print options for 2 decimal places
torch.set_printoptions(precision=1, sci_mode=False)

DATA = 'mnist'
if DATA == 'cora':
    LABELs = {0, 1, 2, 3, 4, 5, 6}
elif DATA == 'mnist':
    LABELs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
else:
    raise ValueError


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedGNN")

    # Add arguments to be parsed
    parser.add_argument('-r', '--label_rate', type=float, required=False, default=1.0,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-l', '--hidden_dimension', type=int, required=False, default=1,
                        help="Class label.")
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=5000,
                        help="The number of epochs (integer).")
    parser.add_argument('-p', '--patience', type=float, required=False, default=1e-3,
                        help="The patience.")
    # parser.add_argument('-a', '--vae_epochs', type=int, required=False, default=10,
    #                     help="vae epochs.")
    # parser.add_argument('-b', '--vae_beta', type=float, required=False, default=1.0,
    #                     help="vae loss beta.")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

# Access the arguments
ALPHA = args.label_rate
NUM_LAYERs = args.hidden_dimension
EPOCHs = args.server_epochs
LR = args.patience
# For testing, print the parsed parameters
# print(f"label_rate: {label_rate}")
# print(f"server_epochs: {server_epochs}")
print(args)
print(f'ALPHA: {ALPHA}, NUM_LAYERs: {NUM_LAYERs}, EPOCHs: {EPOCHs}, LR: {LR}')


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
    # clfs = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gd, 'SVM': svm}

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


#
# # Generator
# class Generator(nn.Module):
#     def __init__(self, latent_dim=10, hidden_dim=128, output_dim=-1):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#             # nn.Tanh()  # Outputs between -1 and 1
#             nn.Sigmoid()  # Outputs between 0 and 1
#         )
#         self.latent_dim = latent_dim
#
#     def forward(self, z):
#         output = self.model(z)
#         # binary_output = (output >= 0.5).float()  # Apply thresholding to get binary outputs
#         # output = F.gumbel_softmax(output, tau=1, hard=True)  # Differentiable approximation
#         return output
#
#
# # Discriminator
# class Discriminator(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()  # Outputs probability
#         )
#
#     def forward(self, x):
#         return self.model(x)


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

    def forward(self, z, c, num_layers=NUM_LAYERs):
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


def train_gan(X_train, y_train, X_val, y_val, X_test, y_test):

    # gans = {l: Generator() for l in LABELs}

    # for l, local_gan in gans.items():
    local_gan = Generator()
    X, y = X_train, y_train

    # label_mask = y == l
    # # if sum(label_mask) == 0:
    # #     continue
    #
    # print(f'training gan for class {l}...')
    # X = X[label_mask]
    # y = y[label_mask]

    # # random select 100 sample for training
    # m = len(X) // 100  # for each client, we only use a subset of data to compute gradient
    # if m < 10:
    #     print(m, len(X))
    #     m = 10

    X = (torch.tensor(X, dtype=torch.float) / 255 - 0.5) * 2  # [-1, 1]
    y = torch.tensor(y, dtype=torch.int)

    onehot_labels = torch.zeros((len(y), len(LABELs))).to(device)
    for i in LABELs:
        mask = y.numpy() == i
        onehot_labels[mask, i] = 1

    # Only update available local labels, i.e., not all the local_gans will be updated.
    # local_labels = set(y.tolist())
    print(f'local labels: {collections.Counter(y.tolist())}, with {len(y)} samples.')

    # Initialize the models, optimizers, and loss function
    # z_dim = 100  # Dimension of random noise
    # data_dim = X.shape[1]  # Number of features (e.g., Cora node features)
    # lr = 0.0002

    # generator = Generator(input_dim=z_dim, output_dim=data_dim).to(device)
    generator = local_gan
    generator = generator.to(device)
    # local_gan.load_state_dict(global_gans[l].state_dict())
    z_dim = generator.latent_dim

    discriminator = Discriminator().to(device)

    # lr = 1e-3
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
    g_loss = 1000
    m = 100  # subset
    for epoch in range(EPOCHs):
        # ---- Train Discriminator ----
        discriminator.train()

        indices = torch.randperm(len(X))[:m]  # Randomly shuffle and pick the first 10 indices
        real_data = X[indices].float().to(device)
        labels = onehot_labels[indices]

        # real_data = X.clone().detach().float().to(device) / 255  # Replace with your local data (class-specific)
        real_data = real_data.to(device)
        real_data = real_data.view(-1, 1, 28, 28)  # Ensure shape is (batch_size, 1, 28, 28)

        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Generate synthetic data
        z = torch.randn(batch_size, z_dim).to(device)
        fake_data = generator(z, labels).detach()  # Freeze Generator when training Discriminator
        # print(fake_data.shape,flush=True)
        # print(epoch, collections.Counter(labels.cpu().numpy().argmax(axis=1).tolist()))

        # Discriminator Loss
        real_loss = adversarial_loss(discriminator(real_data, labels), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_data, labels), fake_labels)
        d_loss = (real_loss + fake_loss)/2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        scheduler_D.step()

        STEP = 10

        generator.train()
        z = torch.randn(batch_size, z_dim).to(device)
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
            z = torch.randn(batch_size, z_dim).to(device)
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
                z = torch.randn(100, 100).to(device)
                c = torch.zeros((100, 10)).to(device)
                for i in range(10):
                    c[i * 10:(i + 1) * 10, i] = 1

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
                        ax.imshow(((real_data[i - 100, 0].cpu().numpy()+1) * 127.5).astype(int), cmap='gray')

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

    # gans[l] = generator

    # return gans
    return local_gan


@timer
def main():
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data(data='mnist')
    import torchvision
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=None, download=True)
    X, Y = train_dataset.data.numpy(), train_dataset.targets.numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(len(y_train), collections.Counter(y_train.tolist()))
    print(len(y_val), collections.Counter(y_val.tolist()))
    print(len(y_test), collections.Counter(y_test.tolist()))

    gan_file = f'cgan_{EPOCHs}_{ALPHA}_{NUM_LAYERs}_{LR}.pt'
    gan_file = 'cgan_10000_0.5_1_0.0001.pt'
    # gan_file = 'cgan_10000.pt'
    if os.path.exists(gan_file):
        print(f'load {gan_file}...')
        cgan = torch.load(gan_file, weights_only=False, map_location=torch.device(device))
    else:
        cgan = train_gan(X_train, y_train, X_val, y_val, X_test, y_test)
        torch.save(cgan, gan_file)

    # Generate data and evaluate model performance
    sizes = {l: s for l, s in collections.Counter(y_train.tolist()).items()}
    print('sizes: ', sizes)
    generated_data = {}
    for l, size in sizes.items():
        print(f'generating class {l} with size: {size}...', )
        generator = cgan.to(device)

        # Sets the model to evaluation mode, which affects certain layers like BatchNorm and Dropout,
        # ensuring they behave correctly during inference.
        generator.eval()

        z_dim = generator.latent_dim

        # torch.no_grad() disables gradient tracking, which reduces memory usage and speeds up inference.
        with torch.no_grad():

            z = torch.randn(size, z_dim).to(device)
            c = torch.zeros([size, 10]).to(device)
            c[:, l] = 1

            generated_imgs = generator(z, c)
            generated_imgs = (generated_imgs + 1) * 127.5  # Convert [-1, 1] back to [0, 255]
            generated_imgs = generated_imgs.clamp(0, 255)  # Ensure values are within [0, 255]
            generated_imgs = generated_imgs.cpu().numpy().astype(int)

            show = True
            if show:
                real_data = X_train[y_train == l]
                fig, axes = plt.subplots(16, 10, figsize=(8, 8))
                for i, ax in enumerate(axes.flatten()):
                    if i < 100:
                        ax.imshow(generated_imgs[i, 0], cmap='gray')
                    else:
                        ax.imshow((real_data[i - 100]).astype(int), cmap='gray')
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
                fig_file = f'tmp/generated_{l}~.png'
                dir_path = os.path.dirname(os.path.abspath(fig_file))
                os.makedirs(dir_path, exist_ok=True)
                plt.savefig(fig_file)
                plt.show()

            generated_imgs = generated_imgs.squeeze(1)  # Removes the second dimension (size 1)

        generated_data[l] = {'X': generated_imgs, 'y': [l] * size}

    # Test on the generated data
    X_gen_test = np.zeros((0, 28, 28))
    y_gen_test = np.zeros((0,), dtype=int)
    for l, vs in generated_data.items():
        X_gen_test = np.concatenate((X_gen_test, vs['X']), axis=0)
        y_gen_test = np.concatenate((y_gen_test, vs['y']))

    print('\n\nX_train, y_train as training set')
    ml_info = evaluate_ML2(X_train.reshape(len(X_train), -1), y_train,
                           X_val.reshape(len(X_val), -1), y_val,
                           X_test.reshape(len(X_test), -1), y_test,
                           X_gen_test.reshape(len(X_gen_test), -1), y_gen_test, verbose=10)

    print('\n\nX_gen_test, y_gen_test as training set')
    ml_info2 = evaluate_ML2(X_gen_test.reshape(len(X_gen_test), -1), y_gen_test,
                            X_train.reshape(len(X_train), -1), y_train,
                            X_val.reshape(len(X_val), -1), y_val,
                            X_test.reshape(len(X_test), -1), y_test, verbose=10)


if __name__ == '__main__':
    main()
