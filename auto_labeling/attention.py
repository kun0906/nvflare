""" Attention module


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class AttentionAggregation(nn.Module):
    def __init__(self, num_clients):
        super(AttentionAggregation, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_clients))  # Attention weights for each client

    def forward(self, client_params):
        """
        client_params: List of client model parameters (each client has a GNN's parameters)
        """
        # Calculate attention scores (softmax over client parameters)
        attention_scores = F.softmax(self.attention_weights, dim=0)

        # Get the shape of client_params and dynamically generate a shape for weights
        shape = list(client_params.shape)  # Get shape of client_params (e.g., [num_clients, 32, 18])
        shape = [1] * len(shape)  # Create a shape with all 1s (e.g., [1, 1, 1])
        shape[0] = -1  # Set the first dimension to -1 (it will be inferred based on the length of weights)
        # Reshape weights based on the calculated shape
        attention_scores = attention_scores.view(shape)  # This will reshape weights to (-1, 1, 1)
        # Weight the client parameters based on attention scores
        weighted_params = client_params * attention_scores  # Element-wise multiplication
        aggregated_params = weighted_params.sum(dim=0)
        return aggregated_params


def train_attention_weights(client_params, pseudo_ground_truth):
    """
    Train the attention weights to learn how to aggregate client parameters.
    """
    num_clients = len(client_params)
    # input_dim = client_params[0]
    num_epochs = 20
    lr = 1e-3
    attention_aggregator = AttentionAggregation(num_clients)

    # Define the optimizer for training attention weights
    optimizer = Adam(attention_aggregator.parameters(), lr=lr)
    # Dummy loss function (you can replace this with any task-specific loss function)
    criterion = nn.MSELoss()  # This could be any task-specific loss

    attention_aggregator.train()
    for epoch in range(num_epochs):
        # Get aggregated parameters using attention mechanism
        aggregated_params = attention_aggregator(client_params)

        # Combine self-consistency loss with the attention weights optimization
        loss = criterion(pseudo_ground_truth, aggregated_params)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"\tattention epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

    return attention_aggregator


def aggregate_with_attention(client_parameters_list, global_model, device=None):
    """
    Train the attention weights and then initialize the global model with aggregated parameters.
    """
    global_state_dict = {key: torch.zeros_like(value).to(device) for key, value in global_model.state_dict().items()}
    # Aggregate parameters for each layer
    for key in global_state_dict:
        print(f'update {key}')
        # global_state_dict[key] += client_state_dict[key].to(device)
        # Train the attention weights
        # give old global_params impact
        beta = 0.1
        client_key_params = torch.stack([vs[key] for vs in client_parameters_list])
        global_key_params = global_state_dict[key]
        # Weighted average of median and old_global_params
        coordinate_median, _ = torch.median(client_key_params, dim=0)
        pseudo_ground_true = (coordinate_median + beta * global_key_params) / (1 + beta)
        attention_aggregator = train_attention_weights(client_key_params, pseudo_ground_true)

        # After training, aggregate the client parameters
        attention_aggregator.eval()

        # Weighted average of median and old_global_params
        aggregated_params = attention_aggregator(client_key_params)
        global_state_dict[key] = (aggregated_params + beta * global_key_params) / (1 + beta)

    # Update the global model with the aggregated parameters
    global_model.load_state_dict(global_state_dict)

#
# """ LSTM + Attention for aggregation
#
#
# """
# import torch
# import torch.nn as nn
# from torch.optim import Adam
#
#
# class AttentionAggregation(nn.Module):
#     # Placeholder for the attention aggregator model.
#     def __init__(self, input_dim, num_clients):
#         super(AttentionAggregation, self).__init__()
#         self.attention_weights = nn.Parameter(torch.randn(num_clients, input_dim))  # Attention weights
#         self.fc = nn.Linear(input_dim, input_dim)  # Linear layer to transform the input
#
#     def forward(self, client_params):
#         # Compute the weighted sum of client parameters using attention weights
#         weighted_params = torch.stack([p * self.attention_weights[i] for i, p in enumerate(client_params)])
#         return self.fc(weighted_params.sum(dim=0))
#
#
# class LSTM(nn.Module):
#     # Placeholder for LSTM model for global parameter prediction
#     def __init__(self, input_dim):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_size=hidden_dim, num_layers=1)
#         self.fc = nn.Linear(hidden_dim, input_dim)
#
#     def forward(self, aggregated_params, previous_global_params):
#         # Combine aggregated parameters with previous global parameters
#         combined_input = torch.cat((aggregated_params, previous_global_params), dim=-1)
#         out, _ = self.lstm(combined_input.unsqueeze(0))  # LSTM expects input of shape (seq_len, batch, input_size)
#         return self.fc(out.squeeze(0))  # Predict global parameters
#
#
# def train_attention_weights(client_params, global_params, previous_global_params, input_dim=10, num_epochs=100,
#                             lr=1e-3):
#     """
#     Train the attention weights and LSTM for global model aggregation in two phases:
#     1. Train attention weights (fix the LSTM).
#     2. Train the LSTM (fix the attention aggregator).
#     """
#     num_clients = len(client_params)
#     attention_aggregator = AttentionAggregation(input_dim, num_clients)
#
#     # Initialize LSTM model
#     lstm = LSTM(input_dim)
#
#     # Phase 1: Train attention weights while fixing the LSTM
#     attention_aggregator.train()
#     for param in lstm.parameters():
#         param.requires_grad = False  # Freeze LSTM weights during this phase
#
#     optimizer_attention = Adam(attention_aggregator.parameters(), lr=lr)
#     criterion = nn.MSELoss()  # Loss function for optimization
#
#     for epoch in range(num_epochs):
#         # Get aggregated parameters using attention mechanism
#         aggregated_params = attention_aggregator(client_params)
#
#         # Get prediction of global parameters using the LSTM (fixed)
#         pred_global_params = lstm(aggregated_params, previous_global_params)
#
#         # Loss function comparing aggregated parameters to LSTM's predicted global parameters
#         loss = criterion(aggregated_params, pred_global_params)
#
#         # Backpropagation and optimization
#         optimizer_attention.zero_grad()
#         loss.backward()
#         optimizer_attention.step()
#
#         if epoch % 10 == 0:
#             print(f"Phase 1 - Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")
#
#     # Phase 2: Train LSTM while fixing the attention aggregator
#     for param in attention_aggregator.parameters():
#         param.requires_grad = False  # Freeze attention aggregator weights during this phase
#
#     # Re-initialize optimizer for the LSTM
#     optimizer_lstm = Adam(lstm.parameters(), lr=lr)
#
#     for epoch in range(num_epochs):
#         # Get aggregated parameters using the fixed attention mechanism
#         aggregated_params = attention_aggregator(client_params)  # Attention is frozen here
#
#         # Get LSTM's prediction of global parameters (LSTM is trainable here)
#         pred_global_params = lstm(aggregated_params, previous_global_params)
#
#         # Loss function comparing aggregated parameters to LSTM's predicted global parameters
#         loss = criterion(aggregated_params, pred_global_params)
#
#         # Backpropagation and optimization for LSTM
#         optimizer_lstm.zero_grad()
#         loss.backward()
#         optimizer_lstm.step()
#
#         if epoch % 10 == 0:
#             print(f"Phase 2 - Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")
#
#     return attention_aggregator, lstm
#
#
# # Example usage:
# client_params = [torch.randn(10) for _ in range(5)]  # Example client parameters
# global_params = torch.randn(10)  # Example global parameters
# previous_global_params = torch.randn(10)  # Example previous global parameters
#
# # Train attention weights and LSTM
# attention_aggregator, trained_lstm = train_attention_weights(client_params, global_params, previous_global_params)
#
