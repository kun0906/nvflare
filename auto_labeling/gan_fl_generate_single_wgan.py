"""
    $ssh kunyang@slogin-01.superpod.smu.edu
    $srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
    $srun -A kunyang_nvflare_py31012_0001 -t 260 -G 1 --pty $SHELL
    $module load conda
    $conda activate nvflare-3.10
    $cd nvflare/auto_labeling
    $PYTHONPATH=. python3 gan_fl_generate_wgan.py


"""
import argparse
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

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
    parser.add_argument('-r', '--label_rate', type=float, required=False, default=1e-3,
                        help="label rate, how much labeled data in local data.")
    parser.add_argument('-l', '--hidden_dimension', type=int, required=False, default=0,
                        help="Class label.")
    parser.add_argument('-n', '--server_epochs', type=int, required=False, default=100000,
                        help="The number of epochs (integer).")
    parser.add_argument('-p', '--patience', type=float, required=False, default=1.0,
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
EPOCHs = args.server_epochs
l = args.hidden_dimension
# num_layers = int(args.patience)
# For testing, print the parsed parameters
# print(f"label_rate: {label_rate}")
# print(f"server_epochs: {server_epochs}")
print(args)


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


def check_gen_data(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test):
    print('\n\nX_train, y_train as training set')
    ml_info = evaluate_ML2(X_train, y_train, X_val, y_val, X_test, y_test, X_gen_test, y_gen_test, verbose=10)

    print('\n\nX_gen_test, y_gen_test as training set')
    ml_info2 = evaluate_ML2(X_gen_test, y_gen_test, X_train, y_train, X_val, y_val, X_test, y_test, verbose=10)

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

#
# class Generator(nn.Module):
#     def __init__(self, latent_dim=100):
#         super(Generator, self).__init__()
#         # Fully connected layer to project z into a 128 * 7 * 7 tensor
#         self.fc1 = nn.Linear(latent_dim, 128 * 7 * 7)
#
#         # Transposed convolutional layers to upsample to 28x28
#         self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 14x14
#         self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample to 28x28
#         self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)  # Keep 28x28
#         self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)  # Keep 28x28
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.latent_dim = latent_dim
#
#     def forward(self, z):
#         x = self.fc1(z)
#         x = x.view(-1, 128, 7, 7)  # Reshape into feature map (batch_size, channels, height, width)
#
#         x = self.relu(self.conv1(x))  # Upsample to 14x14
#         x = self.relu(self.conv2(x))  # Upsample to 28x28
#         x = self.relu(self.conv3(x))  # Keep 28x28
#         x = self.tanh(self.conv4(x))  # Output is in the range [-1, 1]
#
#         return x
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # From 1 channel to 16 channels
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
#         self.fc1 = nn.Linear(64 * 3 * 3, 512)  # Adjust the dimensions after the convolution layers
#         self.fc2 = nn.Linear(512, 1)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # Ensure input has the correct shape (batch_size, 1, 28, 28)
#         x = self.leaky_relu(self.conv1(x))
#         x = self.leaky_relu(self.conv2(x))
#         x = self.leaky_relu(self.conv3(x))
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
#         x = self.leaky_relu(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return x


class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """

    def __init__(self, in_channel=1, input_size=784, condition_size=10, num_classes=1):
        super(Discriminator, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size + condition_size, 784),
            nn.LeakyReLU(0.2),
        )
        self.conv = nn.Sequential(
            # 28 -> 14
            nn.Conv2d(in_channel, 64, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 14 -> 7
            nn.Conv2d(64, 32, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 7 -> 4
            nn.Conv2d(32, 16, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Sequential(
            # reshape input, 128 -> 1
            nn.Linear(16, 1),
        )

    def forward(self, x, c=None):
        # x: (N, 1, 28, 28), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float()  # may not need
        v = torch.cat((x, c), 1)  # v: (N, 794)
        y_ = self.transform(v)  # (N, 784)
        y_ = y_.view(y_.shape[0], 1, 28, 28)  # (N, 1, 28, 28)
        y_ = self.conv(y_)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_


class Generator(nn.Module):
    """
        Convolutional Generator for MNIST
    """

    def __init__(self, latent_dim=100, condition_size=10):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_size, 4 * 4 * 128),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            # input: 4 by 4, output: 7 by 7
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(256),
            nn.ReLU(),
            # input: 7 by 7, output: 14 by 14
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(128),
            nn.ReLU(),
            # input: 14 by 14, output: 28 by 28
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        self.latent_dim = latent_dim

    def forward(self, x, c):
        # x: (N, 100), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float()  # may not need
        v = torch.cat((x, c), 1)  # v: (N, 110)
        y_ = self.fc(v)
        y_ = y_.view(y_.size(0), 128, 4, 4)
        y_ = self.conv(y_)  # (N, 28, 28)
        return y_


# MMD Loss
# def mmd_loss(real_features, fake_features):
#     x_kernel = real_features @ real_features.T
#     y_kernel = fake_features @ fake_features.T
#     xy_kernel = real_features @ fake_features.T
#     mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
#     return mmd


# Gradient Penalty
def compute_gradient_penalty(critic, real_data, fake_data, device):
    from torch.autograd import grad

    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)

    critic_output = critic(interpolates)
    gradients = grad(
        outputs=critic_output,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,  # Retain the computation graph
        retain_graph=True,  # Needed to avoid RuntimeError
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_gan(X_train, y_train, X_val, y_val, X_test, y_test):
    # epochs = 50000

    gans_file = f'gans_{EPOCHs}.pt'
    # if os.path.exists(gans_file):
    #     return torch.load(gans_file)

    # gans = {l: Generator() for l in LABELs}
    #
    # local_gan =
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

    # random select 100 sample for training
    m = len(X) // 100  # for each client, we only use a subset of data to compute gradient
    if m < 10:
        print(m, len(X))
        m = 10
    indices = torch.randperm(len(X))[:m]  # Randomly shuffle and pick the first 10 indices
    X = X[indices]
    y = y[indices]

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.int)

    # Only update available local labels, i.e., not all the local_gans will be updated.
    # local_labels = set(y.tolist())
    print(f'local labels: {collections.Counter(y.tolist())}, with {len(y)} samples.')

    # Initialize the models, optimizers, and loss function
    # z_dim = 100  # Dimension of random noise
    data_dim = X.shape[1]  # Number of features (e.g., Cora node features)
    # lr = 0.0002

    # generator = Generator(input_dim=z_dim, output_dim=data_dim).to(device)
    generator = local_gan
    generator = generator.to(device)
    # local_gan.load_state_dict(global_gans[l].state_dict())
    z_dim = generator.latent_dim

    discriminator = Discriminator().to(device)

    lr = 1e-3
    # optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0., 0.9))
    scheduler_G = StepLR(optimizer_G, step_size=500, gamma=0.85)

    # optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=5e-5)  # L2
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0., 0.9))
    scheduler_D = StepLR(optimizer_D, step_size=500, gamma=0.85)
    # optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    # optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # adversarial_loss = nn.BCELoss(reduction='mean')  # Binary Cross-Entropy Loss

    labels = torch.zeros((len(y), len(LABELs))).to(device)
    for i in LABELs:
        mask = y.numpy() == i
        labels[mask, i] = 1

    # Training loop
    losses = []
    for epoch in range(EPOCHs):
        # ---- Train Discriminator ----
        discriminator.train()
        optimizer_D.zero_grad()

        real_data = X.clone().detach().float().to(device) / 255  # Replace with your local data (class-specific)
        real_data = real_data.to(device)
        real_data = real_data.view(-1, 1, 28, 28)  # Ensure shape is (batch_size, 1, 28, 28)

        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Generate synthetic data
        z = torch.randn(batch_size, z_dim).to(device)
        fake_data = generator(z, labels)  # Freeze Generator when training Discriminator
        # print(fake_data.shape,flush=True)

        # # Gradient Penalty (e.g. gradients w.r.t x_penalty)
        # D_labels = torch.ones([batch_size, 1]).to(device)  # Discriminator Label to real
        # D_fakes = torch.zeros([batch_size, 1]).to(device)  # Discriminator Label to fake
        # eps = torch.rand(batch_size, 1, 1, 1).to(device)  # x shape: (64, 1, 28, 28)
        # x_penalty = eps * real_data + (1 - eps) * fake_data
        # x_penalty = x_penalty.view(x_penalty.size(0), -1)
        # p_outputs = discriminator(x_penalty, labels)
        # xp_grad = autograd.grad(outputs=p_outputs, inputs=x_penalty, grad_outputs=D_labels,
        #                         create_graph=True, retain_graph=True, only_inputs=True)
        # p_coeff = 10  # lambda
        # grad_penalty = p_coeff * torch.mean(torch.pow(torch.norm(xp_grad[0], 2, 1) - 1, 2))
        grad_penalty=0
        # # Wasserstein loss
        real_validity = discriminator(real_data, labels)
        fake_validity = discriminator(fake_data, labels)
        # gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, device)
        # lambda_gp = 10  # Gradient penalty coefficient
        wasserstein_distance = torch.mean(fake_validity) - torch.mean(real_validity)
        d_loss = wasserstein_distance + grad_penalty
        # d_loss = wasserstein_distance
        # WGAN-GP
        d_loss.backward()
        optimizer_D.step()

        scheduler_D.step()

        # Clip discriminator weights
        clip_value = 0.01  # Weight clipping range
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # # Train Generator every n_critic steps
        # if i % n_critic == 0:
        #     optimizer_G.zero_grad()
        #     fake_validity = discriminator(fake_imgs)
        #     g_loss = -torch.mean(fake_validity)
        #     g_loss.backward()
        #     optimizer_G.step()

        if epoch % 2 == 0:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            # ---- Train Generator ----
            # we don't need to freeze the discriminator because the optimizer for the discriminator
            # (optimizer_D) is not called.
            # This ensures that no updates are made to the discriminator's parameters,
            # even if gradients are computed during the backward pass.
            generator.train()
            z = torch.randn(batch_size, z_dim).to(device)
            generated_data = generator(z, labels)

            # Generator Loss (Discriminator should classify fake data as real)
            # g_loss = adversarial_loss(discriminator(generated_data), real_labels)

            fake_validity = discriminator(generated_data, labels)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

        scheduler_G.step()
        # ---- Logging ----
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} |  "
                  f"LR_D: {scheduler_D.get_last_lr()}, LR_G: {scheduler_G.get_last_lr()}")
        losses.append(g_loss.item())

        # Display some generated images after each epoch
        if epoch % 100 == 0:

            with torch.no_grad():
                z = torch.randn(batch_size, z_dim).to(device)
                c = torch.zeros([batch_size, 10]).to(device)
                t = 0
                for i in range(0, batch_size, 10):
                    c[i:i + 10, t % 10] = 1
                    t += 1

                generated_imgs = generator(z, c)
                generated_imgs = (generated_imgs + 1) * 127.5  # Convert [-1, 1] back to [0, 255]
                generated_imgs = generated_imgs.clamp(0, 255)  # Ensure values are within [0, 255]
                generated_imgs = generated_imgs.cpu().numpy().astype(int)

                fig, axes = plt.subplots(16, 10, figsize=(8, 8))
                for i, ax in enumerate(axes.flatten()):
                    if i < 100:
                        ax.imshow(generated_imgs[i, 0], cmap='gray')
                    else:
                        ax.imshow((real_data[i, 0].cpu().numpy() * 255).astype(int), cmap='gray')
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f'tmp/gen_{epoch}.png')
                plt.show()

    # gans[l] = generator
    torch.save(local_gan, f'cwgan_{EPOCHs}.pt')

    # return gans
    return local_gan


def main():
    # # X_train, y_train, X_val, y_val, X_test, y_test = load_data(data='mnist')
    # # gans = {l: f'gan_{l}_1000001.pt' for l in LABELs}
    # gans = {l: f'wgan_{l}_{EPOCHs}.pt' for l in LABELs}
    # check = True
    # for l, f in gans.items():
    #     if os.path.exists(f):
    #         continue
    #     check = False

    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    X, Y = train_dataset.data.numpy(), train_dataset.targets.numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(len(y_train), collections.Counter(y_train.tolist()))
    print(len(y_val), collections.Counter(y_val.tolist()))
    print(len(y_test), collections.Counter(y_test.tolist()))

    cgan = train_gan(X_train, y_train, X_val, y_val, X_test, y_test)

    sizes = {l: s for l, s in collections.Counter(y_train.tolist()).items()}

    print('sizes: ', sizes)
    generated_data = {}

    gan_file = f'cwgan_{EPOCHs}.pt'
    print(f'load {gan_file}...')

    gan = torch.load(gan_file, weights_only=None, map_location=torch.device('cpu'))

    for l, size in sizes.items():

        generator = gan.to(device)
        generator.eval()
        z_dim = generator.latent_dim
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
                fig, axes = plt.subplots(16, 8, figsize=(8, 8))
                for i, ax in enumerate(axes.flatten()):
                    ax.imshow(generated_imgs[i, 0], cmap='gray')
                    # if i < 64:
                    #     ax.imshow(generated_imgs[i, 0], cmap='gray')
                    # else:
                    #     ax.imshow((real_data[i, 0].cpu().numpy() * 255).astype(int), cmap='gray')
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f'tmp/generated_{l}~.png')
                plt.show()

            generated_imgs = generated_imgs.squeeze(1)  # Removes the second dimension (size 1)

        generated_data[l] = {'X': generated_imgs, 'y': [l] * size}

        # test on the generated data
    # dim = X_train.shape[1]
    X_gen_test = np.zeros((0, 28, 28))
    y_gen_test = np.zeros((0,), dtype=int)

    for l, vs in generated_data.items():
        X_gen_test = np.concatenate((X_gen_test, vs['X']), axis=0)
        y_gen_test = np.concatenate((y_gen_test, vs['y']))

    check_gen_data(X_gen_test.reshape(len(X_gen_test), -1), y_gen_test,
                   X_train.reshape(len(X_train), -1), y_train,
                   X_val.reshape(len(X_val), -1), y_val,
                   X_test.reshape(len(X_test), -1), y_test)


if __name__ == '__main__':
    main()
