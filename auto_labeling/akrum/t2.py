import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.set_printoptions(precision=1, suppress=True)
VERBOSE = 1
import torch
import numpy as np
import matplotlib.pyplot as plt
import ragg.robust_aggregation as rag

# Function to compute geometric median with uniform weights (susceptible to Byzantine failure)
def geometric_median_with_byzantine(clients_updates, clients_weights, init_value=None, max_iters=100, tol=1e-6,
                                    verbose=1):
    N, D = clients_updates.shape
    assert clients_weights.shape == (N,), "Weights must be of shape (N,) where N is the number of updates."

    # Use uniform weights, making the algorithm more susceptible to the Byzantine client
    clients_weights = torch.ones_like(clients_weights) / N  # Equal weight for all clients

    if init_value is not None:
        estimated_center = init_value
    else:
        # Initial estimate: weighted average
        estimated_center = torch.sum(clients_updates * clients_weights.view(-1, 1), dim=0)

    for iteration in range(max_iters):
        distances = torch.norm(clients_updates - estimated_center, dim=1)
        distances = torch.clamp(distances, min=tol)  # Avoid division by zero

        inv_distances = 1.0 / distances
        weighted_inv_distances = clients_weights * inv_distances

        weighted_update = torch.sum(weighted_inv_distances.view(-1, 1) * clients_updates, dim=0) / torch.sum(
            weighted_inv_distances)

        diff = torch.norm(weighted_update - estimated_center).item()
        # Convergence check
        if diff < tol:
            print(diff)
            break
        # if iteration%50==0:
        #     print(iteration, diff)

        estimated_center = weighted_update
    # print(f"diff: {diff}")
    return weighted_update, estimated_center


# Create some 2D data for the example
np.random.seed(42)
#
# # Define 4 normal points and 1 Byzantine point
# normal_points = np.random.randn(4, 2)  # 4 points in 2D
#
# byzantine_point = np.array([[10, 10]])  # Outlier (Byzantine client)
# clients_updates = np.vstack([normal_points, byzantine_point])

# # Define normal points, clustered closely together around the true center
# normal_points = np.array([[0.6, 0.5], [0.65, 0.55], [0.55, 0.45], [0.7, 0.6]])  # Closer points
#
# # Define a very far away Byzantine point
# byzantine_point = np.array([[50, 50]])  # A very distant outlier

for location in [10]:    #[10, 20, 30, 50, 100, 150, 200, 300, 400, 500, 1000]:   # int(1e+6)
    n = location
    f = (n - 3) // 2
    # f = n//4
    h = n - f
    d = 5
    print(n, h, f, flush=True)
    # normal_points = np.random.randn(h, d)  # 4 points in 2D
    cov = np.eye(d)*0.1
    normal_points = np.random.multivariate_normal(np.zeros((d,)), cov, size=h//2)
    normal_points2 = np.random.multivariate_normal(np.ones((d,))*0.5, cov, size=h-h//2)
    normal_points = np.concatenate((normal_points, normal_points2), axis=0)

    # Create Byzantine points by shifting their location and adding more variation
    # byzantine_point = np.random.randn(f, d) + location * np.random.randn(f, d)*100  # Added randomness to location
    cov = np.eye(d)*0.01
    # cov[0, :] = 10
    # cov[0,0] = 1e+20
    # cov = np.random.randn(d,d)+10000
    byzantine_point = np.random.multivariate_normal(np.ones((d,)) * 5.0, cov, size=f)
    # byzantine_point2 = np.random.multivariate_normal(np.ones((d,))+(1e+4), cov, size=f-1)
    # byzantine_point = np.concatenate((byzantine_point, byzantine_point2), axis=0)
    # mu=np.ones((d,)) * 1e+1
    # mu[0] = 1000
    # byzantine_point = np.random.multivariate_normal(mu, cov, size=f)

    # # Create a random rotation matrix (2D in this case)
    # # theta = np.random.uniform(0, 2 * np.pi)  # Random rotation angle between 0 and 2*pi
    # theta = 0.3
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # # Rotate Byzantine points
    # rotated_byzantine_points = byzantine_point.dot(rotation_matrix)
    # # Scale the rotated Byzantine points so they are far from the normal points
    # scale_factor = 2000
    # byzantine_point = rotated_byzantine_points * scale_factor

    # Combine normal points and Byzantine client
    clients_updates = np.vstack([normal_points, byzantine_point])

    # Set initial weights (with equal weights for simplicity)
    clients_weights = np.ones(n)
    true_center = np.mean(normal_points, axis=0)

    # if byzantine is very far away from normal points, which lead geometric median more iteration to converge.
    # # Run the geometric median with the Byzantine point
    # geometric_median_result, final_center = geometric_median_with_byzantine(
    #     torch.tensor(clients_updates, dtype=torch.float32),
    #     torch.tensor(clients_weights, dtype=torch.float32),
    #     max_iters=100, tol=1e-6, verbose=1)

    points = torch.tensor(clients_updates, dtype=torch.float32)
    weights = torch.tensor(clients_weights, dtype=torch.float32)

    # if normal point with larger cov, cw_median will fail
    # final_center, clients_type = rag.cw_median(points, weights, verbose=VERBOSE)

    # if normal point with larger cov, medoid will fail
    final_center, clients_type = rag.medoid(points, weights, verbose=VERBOSE)
    print(f'n:{n}, medoid: ', final_center[:5], torch.norm(final_center - true_center).item(),  np.where(clients_type == 'Chosen Update')[0])

    # if normal point with larger cov, medoid will fail
    final_center2, clients_type2 = rag.adaptive_krum(points, weights, verbose=VERBOSE)
    print(f'n:{n}, adaptive_krum:', final_center2[:5], torch.norm(final_center2 - true_center).item(), np.where(clients_type2 == 'Chosen Update')[0])

    # print(f"true_center: {true_center}, final_center: {final_center}")
    # print(f'location: {location}, l2: {torch.norm(final_center - true_center).item()}')

# Plotting the result
plt.figure(figsize=(8, 6))
plt.scatter(clients_updates[:, 0], clients_updates[:, 1], c='blue', label='Client Updates')
plt.scatter(true_center[0], true_center[1], c='green', marker='x', label='True Center')
plt.scatter(final_center[0], final_center[1], c='red', marker='x', label='Geometric Median', s=10)
plt.title("Geometric Median with a Byzantine Client")
plt.legend()
plt.grid(True)
plt.show()

exit(0)

# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt

# Define the range of 'a' from 0 to pi/2

alpha_values = np.linspace(0, np.pi / 2, 100)
y_values = 1 - np.sin(alpha_values)

# Plot the function
plt.figure(figsize=(6, 4))
plt.plot(alpha_values, y_values, label=r'$1 - \sin(a)$', color='b')
plt.xlabel(r'$a$ (radians)')
plt.ylabel(r'$1 - \sin(a)$')
plt.title(r'Plot of $1 - \sin(\alpha)$ for $0 \leq \alpha < \frac{\pi}{2}$')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(True)
plt.show()

exit(0)

# Generate 10 points on a circle of radius 1 centered at (0,0)
n = 10
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
circle_points = np.array([(np.cos(a), np.sin(a)) for a in angles])

# Add an outlier at (100,100)
outlier = np.array([[10, 10]] * (n - 2))

# Combine points
points = np.vstack((circle_points, outlier))
print(points)

# Compute Mean (average of all points)
mean_point = np.mean(points, axis=0)

# Compute Median (coordinate-wise median)
median_point = np.median(points, axis=0)

# Compute Medoid (point closest to all others in terms of total distance)
dist_matrix = cdist(points, points, metric='euclidean')
total_distances = np.sum(dist_matrix, axis=1)
medoid_index = np.argmin(total_distances)
medoid_point = points[medoid_index]

# Compute Krum (Selects a point closest to its nearest neighbors, ignoring the farthest)
k = len(points) - len(outlier) - 2  # k = total_n -f-2  = 19 - 9 - 2 = 8
print(f'k:{k}')
krum_scores = np.sum(np.sort(dist_matrix, axis=1)[:, 1:k + 1], axis=1)  # Ignore self-distance (0)
print(krum_scores)
krum_index = np.argmin(krum_scores)
krum_point = points[krum_index]


#
# def adaptive_weighted_mean(points, beta_values):
#     center_guess = np.median(points, axis=0)  # Initial guess
#     distances = np.linalg.norm(points - center_guess, axis=1)
#
#     plt.figure(figsize=(8, 6))
#
#     for beta in beta_values:
#         weights = np.exp(-beta * distances)  # Exponential decay for far points
#         weights /= np.sum(weights)  # Normalize
#         plt.plot(points[:, 0], weights, 'o', label=f'β={beta}')
#
#     plt.xlabel("Distance from Median")
#     plt.ylabel("Weight")
#     plt.title("Adaptive Weighted Mean: Weights vs. Distance for Different β")
#     plt.legend()
#     plt.grid()
#     plt.show()
#
# # Test with different β values
# beta_values = [0.1, 1, 5, 10, 20]
# adaptive_weighted_mean(points, beta_values)
# exit(0)

# Compute Adaptive Weighted Mean
def adaptive_weighted_mean(points, beta=1):
    center_guess = np.median(points, axis=0)  # Initial guess
    distances = np.linalg.norm(points - center_guess, axis=1)
    weights = np.exp(-beta * distances)  # Exponential decay for far points
    weights /= np.sum(weights)
    print(weights)
    return np.sum(points * weights[:, np.newaxis], axis=0)


adaptive_mean = adaptive_weighted_mean(points)


# Compute Geometric Median using Weiszfeld’s algorithm
def geometric_median(X, eps=1e-5):
    y = np.mean(X, axis=0)  # Initial guess
    while True:
        distances = np.linalg.norm(X - y, axis=1)
        nonzero_distances = np.where(distances > eps, distances, eps)  # Avoid division by zero
        weights = 1 / nonzero_distances
        new_y = np.average(X, axis=0, weights=weights)
        if np.linalg.norm(y - new_y) < eps:
            return new_y
        y = new_y


geo_median = geometric_median(points)

# Plot results
plt.figure(figsize=(6, 6))
plt.scatter(circle_points[:, 0], circle_points[:, 1], label="Circle Points", color="blue", alpha=0.7)
plt.scatter(outlier[:, 0], outlier[:, 1], label="Outlier (100,100)", color="red", marker="x", s=150)
plt.scatter(*mean_point, color="purple", label="Mean", marker="o", s=100)
plt.scatter(*median_point, color="green", label="Median", marker="s", s=100)
plt.scatter(*medoid_point, color="orange", label="Medoid", marker="D", s=100)
plt.scatter(*krum_point, color="brown", label="Krum", marker="P", s=100)
plt.scatter(*geo_median, color="black", label="Geo_median", marker="o", s=100)
plt.scatter(*adaptive_mean, color="yellow", label="adaptive_mean", marker="s", s=100)

# plt.xlim(-2, 105)
# plt.ylim(-2, 105)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Comparison of Aggregation Methods")
plt.grid()
plt.show()

# Print results
print(mean_point, median_point, medoid_point, krum_point, geo_median, adaptive_mean)

#
# random_state=42
# torch.manual_seed(random_state)
# num_samples = 10
# indices = torch.randperm(num_samples)  # Randomly shuffle
# print(indices)
# indices = torch.randperm(num_samples)  # Randomly shuffle
# print(indices)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# from scipy.optimize import minimize
#
# # Generate 2D points along the line y = x
# np.random.seed(42)
# n = 20
# x_values = np.linspace(1, 10, n)
# points = np.vstack((x_values, x_values)).T  # Each point (x, x)
#
# # # Compute coordinate-wise median (marginal median)
# # coord_median = np.median(points, axis=0)
#
# # Compute geometric median using optimization
# def geometric_median(X, eps=1e-5):
#     """Compute the geometric median using Weiszfeld's algorithm."""
#     y = np.mean(X, axis=0)  # Initial guess: mean of points
#     while True:
#         distances = np.linalg.norm(X - y, axis=1)
#         nonzero_distances = np.where(distances > eps, distances, eps)  # Avoid division by zero
#         weights = 1 / nonzero_distances
#         new_y = np.average(X, axis=0, weights=weights)
#         if np.linalg.norm(y - new_y) < eps:
#             return new_y
#         y = new_y
#
# # geo_median = geometric_median(points)
#
# # Add some noise to y values
# noise = np.random.normal(scale=2000.0, size=n)
# points_noisy = np.vstack((x_values, x_values + noise)).T  # Slightly off the y=x line
#
# coord_median_noisy = np.median(points_noisy, axis=0)
# geo_median_noisy = geometric_median(points_noisy)
#
# # Plot
# plt.figure(figsize=(6, 6))
# plt.scatter(points_noisy[:, 0], points_noisy[:, 1], label="Noisy Data Points", color="blue")
# plt.scatter(*coord_median_noisy, color="red", label="Coordinate-wise Median", marker="x", s=150)
# plt.scatter(*geo_median_noisy, color="green", label="Geometric Median", marker="o", s=150)
#
# plt.plot(x_values, x_values, 'k--', label="y = x")
# plt.legend()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Geometric Median vs. Coordinate-wise Median (With Noise)")
# plt.grid()
# plt.show()
#

exit(0)

# import numpy as np
#
# NUM_HONEST_CLIENTS = 4
# for c in range(NUM_HONEST_CLIENTS):
#     # client_type = 'Honest'
#     # print(f"\n***server_epoch:{epoch}, client_{c}: {client_type}...")
#     X_c = [1] * 10
#     y_c = [0] * 10
#     np.random.seed(c)
#     if c % 4 == 0:  # 1/4 of honest clients has part of classes
#         mask_c = np.full(len(y_c), False)
#         # for l in [0, 1, 2, 3, 4]:
#         ls = []
#         for l in np.random.choice([0, 1, 2, 3, 4], size=2, replace=False):
#             mask_ = y_c == l
#             mask_c[mask_] = True
#             ls.append(l)
#         print(c, ls)
#         # mask_c = (y_c != (c%10))  # excluding one class for each client
#     elif c % 4 == 1:  # 1/4 of honest clients has part of classes
#         mask_c = np.full(len(y_c), False)
#         # for l in [5, 6, 7, 8, 9]:
#         ls = []
#         for l in np.random.choice([5, 6, 7, 8, 9], size=2, replace=False):
#             mask_ = y_c == l
#             mask_c[mask_] = True
#             ls.append(l)
#         print(c, ls)
#         #
#     else:  # 2/4 of honest clients has IID distributions
#         mask_c = np.full(len(y_c), True)
#
# exit(0)


file_path = 'histories_gan_r_0.1-n_1.pth'
data = torch.load(file_path, map_location="cpu")
print(data)


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
