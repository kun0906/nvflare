"""
    1. ConditionalDiffusionModel
    2. Conditional GANs
    3. Conditional VAE
    4. Conditional Flow-based Models
    5. PixelCNN / PixelSNAIL
    6. ConditionalTransformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, timesteps, latent_dim, num_classes):
        super(ConditionalDiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes

        # Class embedding to condition the generation process
        self.class_embedding = nn.Embedding(num_classes, latent_dim)

        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(latent_dim + num_classes, 64, kernel_size=3, stride=1, padding=1),  # Conditionally input label
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # Assuming we generate 3-channel images
        )

    def forward(self, x, t, labels):
        # Embedding the labels
        label_embedding = self.class_embedding(labels)
        label_embedding = label_embedding.view(-1, self.latent_dim, 1, 1).expand(-1, self.latent_dim, x.size(2),
                                                                                 x.size(3))  # Expand to image size

        # Concatenate the noisy image with the label embedding
        x_cond = torch.cat([x, label_embedding], dim=1)

        # Pass through noise predictor
        noise_prediction = self.noise_predictor(x_cond)
        return noise_prediction


#
# # Instantiate the model
# timesteps = 1000
# latent_dim = 64
# num_classes = 10
# diff_model = ConditionalDiffusionModel(timesteps=timesteps, latent_dim=latent_dim, num_classes=num_classes)
#
# # Example usage
# x = torch.randn(32, 3, 64, 64)  # Noisy image (32 samples, 3 channels, 64x64 pixels)
# t = torch.randint(0, timesteps, (32,))  # Random timesteps for diffusion process
# labels = torch.randint(0, num_classes, (32,))  # Class labels
#
# output = diff_model(x, t, labels)
# print(output.shape)  # Output should have shape (32, 3, 64, 64)
#
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ConditionalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(ConditionalGNN, self).__init__()
        self.num_classes = num_classes
        self.gcn1 = GCNConv(input_dim + num_classes, hidden_dim)  # Conditional input
        self.gcn2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, class_labels):
        # One-hot encoding of class labels
        class_embedding = F.one_hot(class_labels, num_classes=self.num_classes).float()

        # Concatenate the class labels with node features
        x_cond = torch.cat([x, class_embedding], dim=-1)

        # Pass through the GCN layers
        x = F.relu(self.gcn1(x_cond, edge_index))
        x = self.gcn2(x, edge_index)
        return x


#
# # Example usage for Conditional GNN
# num_nodes = 100  # Example number of nodes
# input_dim = 10  # Initial node features
# hidden_dim = 32
# output_dim = 10  # Number of features for output (can be number of classes or other)
# num_classes = 10  # Class labels (10 classes)
#
# # Create a random graph (edge_index, features)
# edge_index = torch.randint(0, num_nodes, (2, 500))  # 500 edges
# node_features = torch.randn(num_nodes, input_dim)  # Node features
#
# # Define class labels for each node (for conditioning)
# class_labels = torch.randint(0, num_classes, (num_nodes,))
#
# # Instantiate model and forward pass
# gnn_model = ConditionalGNN(input_dim, hidden_dim, output_dim, num_classes)
# output = gnn_model(node_features, edge_index, class_labels)
#
# print(output.shape)  # Output shape should be (num_nodes, output_dim)


import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim, latent_dim, num_classes, output_dim):
        super(Generator, self).__init__()
        self.num_classes = num_classes

        # Class embedding to condition the generation process
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # The generator network
        self.fc1 = nn.Linear(noise_dim + latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z, labels):
        # Embed the labels
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(label_emb.size(0), -1)

        # Concatenate random noise and label embedding
        x = torch.cat([z, label_emb], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Assuming the output is in the range [-1, 1] for images
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # Class embedding to condition the discriminator
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # The discriminator network
        self.fc1 = nn.Linear(input_dim + latent_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        # Embed the labels
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(label_emb.size(0), -1)

        # Concatenate input data and label embedding
        x = torch.cat([x, label_emb], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


#
# # Example usage for Conditional GAN
# noise_dim = 100  # Latent space size (random noise)
# latent_dim = 10  # Latent space size for label embeddings
# num_classes = 10  # Number of classes (e.g., 10 for MNIST)
# input_dim = 784  # Input size (e.g., 28x28 image)
# output_dim = 784  # Output size (same as input size for MNIST)
#
# # Instantiate generator and discriminator
# generator = Generator(noise_dim, latent_dim, num_classes, output_dim)
# discriminator = Discriminator(input_dim, latent_dim, num_classes)
#
# # Example batch of random noise and labels
# batch_size = 32
# z = torch.randn(batch_size, noise_dim)  # Random noise
# labels = torch.randint(0, num_classes, (batch_size,))  # Class labels
#
# # Generate fake data
# generated_data = generator(z, labels)
#
# # Discriminator evaluation of generated data
# discriminator_output = discriminator(generated_data, labels)
#
# print(generated_data.shape)  # Output shape should be (batch_size, output_dim), e.g., (32, 784)
# print(discriminator_output.shape)  # Output shape should be (batch_size, 1)
#
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalTransformer(nn.Module):
    def __init__(self, timesteps, latent_dim, num_classes, image_size, num_channels=3):
        super(ConditionalTransformer, self).__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        # Class embedding to condition the generation process
        self.class_embedding = nn.Embedding(num_classes, latent_dim)

        # Transformer decoder layers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8, dim_feedforward=256)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        # Final output projection (to match image size)
        self.fc_out = nn.Linear(latent_dim, image_size * image_size * num_channels)

    def forward(self, x, t, labels):
        # Embedding the labels
        label_embedding = self.class_embedding(labels)
        label_embedding = label_embedding.view(-1, 1, self.latent_dim).expand(-1, x.size(1),
                                                                              self.latent_dim)  # Expand to match sequence length

        # Positional encoding (assuming sequence data)
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        pos_embedding = self.positional_encoding(pos, x.size(1))

        # Add class label and positional encoding
        x_cond = x + label_embedding + pos_embedding

        # Pass through the Transformer Decoder
        output = self.transformer_decoder(x_cond, x_cond)

        # Final output projection to image space
        output = self.fc_out(output).view(-1, self.num_channels, self.image_size, self.image_size)
        return output

    def positional_encoding(self, positions, size):
        pe = torch.zeros(size, self.latent_dim, device=positions.device)
        position = positions.unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.latent_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

#
# # Instantiate the model
# timesteps = 1000
# latent_dim = 64
# num_classes = 10
# image_size = 64
# num_channels = 3
# transformer_model = ConditionalTransformer(timesteps=timesteps, latent_dim=latent_dim, num_classes=num_classes,
#                                            image_size=image_size)
#
# # Example usage
# x = torch.randn(32, 10, latent_dim)  # Input sequence (32 samples, 10 timesteps, latent_dim features)
# t = torch.randint(0, timesteps, (32,))  # Random timesteps
# labels = torch.randint(0, num_classes, (32,))  # Class labels
#
# output = transformer_model(x, t, labels)
# print(output.shape)  # Output should have shape (32, 3, 64, 64)
