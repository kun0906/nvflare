import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Minibatch Discrimination
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, num_kernels):
        super(MinibatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, num_kernels))

    def forward(self, x):
        M = torch.einsum("bi,ijk->bjk", x, self.T)
        M1 = M.unsqueeze(0)  # Shape: (1, batch_size, num_kernels, out_features)
        M2 = M.unsqueeze(1)  # Shape: (batch_size, 1, num_kernels, out_features)
        # c = torch.exp(-torch.norm(M1 - M2, p=2, dim=3))  # RBF Kernel

        c = torch.exp(-torch.clamp(torch.norm(M1 - M2, p=2, dim=3), max=10))

        minibatch_features = torch.sum(c, dim=1)
        return torch.cat([x, minibatch_features], dim=1)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, num_kernels):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        # self.minibatch_discrimination = MinibatchDiscrimination(64, 32, num_kernels)
        # self.output = nn.Linear(64 + 32, 1)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        features = self.fc(x)
        # features = self.minibatch_discrimination(features)
        validity = torch.sigmoid(self.output(features))
        return validity, features


# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, z):
        return self.model(z)


# Load Cora dataset
def load_cora():
    import networkx as nx
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root="cora_data", name="Cora")
    data = dataset[0]
    X = data.x.numpy()
    y = data.y.numpy()
    return X, y


import torch


def gaussian_kernel(x, y, sigma=1.0):
    """
    Computes the Gaussian kernel between two sets of vectors.
    Args:
        x (torch.Tensor): Input tensor of shape (N, D), where N is the batch size, D is the feature dimension.
        y (torch.Tensor): Input tensor of shape (M, D), where M is the batch size, D is the feature dimension.
        sigma (float): Bandwidth of the Gaussian kernel.
    Returns:
        torch.Tensor: Kernel matrix of shape (N, M).
    """
    diff = x[:, None, :] - y[None, :, :]  # Shape: (N, M, D)
    dist_sq = torch.sum(diff ** 2, dim=2)  # Squared Euclidean distance
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))  # Gaussian kernel
    return kernel


def mmd_loss(real_features, fake_features, sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss.
    Args:
        real_features (torch.Tensor): Real feature tensor of shape (N, D).
        fake_features (torch.Tensor): Fake feature tensor of shape (M, D).
        sigma (float): Bandwidth of the Gaussian kernel.
    Returns:
        torch.Tensor: Scalar MMD loss value.
    """
    # Compute the kernel matrices
    real_kernel = gaussian_kernel(real_features, real_features, sigma)  # (N, N)
    fake_kernel = gaussian_kernel(fake_features, fake_features, sigma)  # (M, M)
    cross_kernel = gaussian_kernel(real_features, fake_features, sigma)  # (N, M)

    # Compute the MMD loss
    mmd = real_kernel.mean() + fake_kernel.mean() - 2 * cross_kernel.mean()
    return mmd


# Gradient Penalty
def gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1).to(device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# Main Training Loop
z_dim = 10
input_dim, num_kernels = 1433, 16
epochs, batch_size, lambda_gp = 10000, 64, 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim, input_dim).to(device)
discriminator = Discriminator(input_dim, num_kernels).to(device)
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Load Data
X, y = load_cora()
X = torch.tensor(X).float().to(device)
y = torch.tensor(y).int().to(device)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training loop
for epoch in range(epochs):
    # ---- Train Discriminator ----
    discriminator.train()
    idx = torch.randperm(X.shape[0])[:batch_size]  # Randomly select a batch
    real_data = X[idx].float().to(device)  # Replace with Cora data
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Generate fake data
    z = torch.randn(batch_size, z_dim).to(device)
    fake_data = generator(z).detach()

    # Discriminator loss
    real_validity, real_features = discriminator(real_data)
    fake_validity, fake_features = discriminator(fake_data)

    real_loss = adversarial_loss(real_validity, real_labels)
    fake_loss = adversarial_loss(fake_validity, fake_labels)
    gp = gradient_penalty(discriminator, real_data, fake_data)
    domain_loss = mmd_loss(real_features, fake_features)

    d_loss = real_loss + fake_loss + lambda_gp * gp + 0.1 * domain_loss
    optimizer_D.zero_grad()
    d_loss.backward(retain_graph=True)
    optimizer_D.step()

    # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # Adjust max_norm as necessary

    # ---- Train Generator ----
    generator.train()

    z = torch.randn(batch_size, z_dim).to(device)
    generated_data = generator(z)

    # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # Adjust max_

    validity, gen_features = discriminator(generated_data)

    g_loss = adversarial_loss(validity, real_labels) + 0.1 * mmd_loss(real_features, gen_features)

    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

    # scheduler_G.step()
    # scheduler_D.step()

    # ---- Logging ----
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Evaluate with Random Forest
z = torch.randn(len(X_test), z_dim).to(device)
synthetic_data = generator(z).detach().cpu().numpy()

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(synthetic_data)
synthetic_accuracy = accuracy_score(y_test, y_pred)

print(f"Synthetic Data -> Test Accuracy: {synthetic_accuracy:.4f}")
