"""Visualize NN

"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        return x


def train(model, X):
    with torch.no_grad():
        X = torch.relu(model.conv1(X))  # Get output of the first convolutional layer
        X = torch.relu(model.conv2(X))  # Get output of the second convolutional layer

    return X


def visualize_kernels(weights):
    n_filters = weights.shape[0]
    n_kernels = weights.shape[1]
    kernel_size = weights.shape[2]
    n_rows = n_filters
    n_cols = n_kernels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    # axes = axes.flatten()

    for i_filter in range(n_filters):
        for i_kernel_per_filter in range(n_kernels):
            ax = axes[i_filter][i_kernel_per_filter]
            ax.imshow(weights[i_filter][i_kernel_per_filter], cmap='viridis')
            if i_kernel_per_filter == 0:
                ax.set_ylabel(f'Kernel_{i_filter + 1}', fontsize=10)
            # ax.axis('off') will remove y_label
            ax.set_xticks([])
            ax.set_yticks([])

    # # Hide any unused subplots
    # for i in range( n_filters, len(axes)):
    #     axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(feature_maps):
    n_rows = feature_maps.shape[0]  # number of images
    n_features = feature_maps.shape[1]
    size = feature_maps.shape[2]

    n_cols = n_features
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    for i_img in range(n_rows):
        for i_feat_map in range(n_features):
            ax = axes[i_img][i_feat_map]
            ax.imshow(feature_maps[i_img, i_feat_map], cmap='viridis')
            if i_feat_map == 0:
                ax.set_ylabel(f'image_{i_img}', fontsize=10)
            ax.set_xlabel(f'feat_map_{i_feat_map}', fontsize=10)
            # ax.axis('off') will remove y_label
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    model = SimpleCNN()

    print(model)
    print(model.parameters())

    # Visualize kernels from the second convolutional layer
    visualize_kernels(model.conv2.weight.detach().cpu().numpy())

    # Create a sample input tensor (batch_size, channels, height, width)
    X = torch.randn(2, 3, 28, 28)
    X = train(model, X)
    # Visualize feature maps from the second convolutional layer
    visualize_feature_maps(X.detach().cpu().numpy())

    print()


if __name__ == '__main__':
    main()
