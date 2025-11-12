import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, d, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 10),
            nn.ReLU(),
            nn.Linear(10, latent_dim)  # Latent space: One neuron
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.ReLU(),
            nn.Linear(10, d)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# Train Autoencoder Function

def train_autoencoder(model, data, epochs=11, lr=1e-5, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = data.shape[0]

    for epoch in range(epochs):
        # Shuffle data indices
        shuffled_indices = torch.randperm(n)
        data = data[shuffled_indices]
        epoch_loss = 0
        # Mini-batch training (optional)
        for i in range(0, n, batch_size):
            batch = data[i:i + batch_size]

            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, loss: {epoch_loss}')

    return model


# Iterative Robust Mean Estimation
def robust_mean_estimation(X, f):
    n, d = X.shape
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_tensor_good = X_tensor.clone()

    # Train autoencoder
    model = Autoencoder(d)
    max_iters = 20

    for _ in range(max_iters):
        model = train_autoencoder(model, X_tensor_good)
        # Compute reconstruction errors
        with torch.no_grad():
            reconstructed, _ = model(X_tensor)  # test on all points
            errors = torch.norm(X_tensor - reconstructed, dim=1)

        # Identify f points with the highest reconstruction error
        worst_indices = errors.argsort(descending=True)[:f]
        predict_labels = np.zeros(n)
        predict_labels[worst_indices] = 1
        # Remove bad points
        mask = torch.ones(n, dtype=torch.bool)
        mask[worst_indices] = False
        X_tensor_good = X_tensor[mask]
        m = X_tensor_good.shape[0]  # Update remaining sample count
        print(m, f,predict_labels)

    # Compute final robust mean
    print(predict_labels, X_tensor_good.shape)
    robust_mean = X_tensor_good.mean(dim=0).numpy()
    return robust_mean


def main():
    # Example Usage
    np.random.seed(42)
    # d = 2  # Dimensionality
    n = 20  # Total samples
    f = (n - 2) // 2 - 1  # Corrupted points
    # f = n//2 + 1
    true_center = np.asarray([10, 100])  # np.mean(honest_points, axis=0)
    honest_points = np.random.multivariate_normal(true_center, cov=[[5, 0], [0, 5]],
                                                  size=n - f)  # 50 honest points (Gaussian)
    empirical_mean = np.mean(honest_points, axis=0)
    print(true_center, n, n-f, f)

    byzantine_points = np.random.multivariate_normal([1, 5], cov=[[0.001, 0.0], [0, 0.001]],
                                                     size=f)  # 10 Byzantine points (outliers)
    X = np.vstack([honest_points, byzantine_points])

    # Compute robust mean
    robust_mu = robust_mean_estimation(X, f)
    print(f'True Mean: {true_center}')
    print(f'Empirical Mean: {empirical_mean}')
    print("Robust Mean Estimate:", robust_mu)


if __name__ == '__main__':
    main()
