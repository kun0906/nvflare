""" only focus on text dataset

    pip install torch torchvision torch-geometric transformers


    Shakespeare Dataset:
    for text classification (e.g., tragedy, comedy)
    or Sentiment Analysis (e.g., positive, negative, neutral)
"""
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

# Load the Reddit dataset (large dataset with 2.6M posts and 2.3M edges)
dataset = Reddit(root='./data', name='Reddit')
data = dataset[0]  # Get the graph data object for Reddit

# Extract number of nodes and edges
num_nodes = data.num_nodes
num_edges = data.num_edges
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

# Initialize BERT model and tokenizer for feature extraction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

# Function to extract BERT embeddings for node text
def get_bert_embeddings(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Pass through the BERT model
    with torch.no_grad():
        outputs = model_bert(**inputs)
    # Mean pooling: Take the mean of all tokens' embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Here we assume the dataset has text data for each node (subreddit or user)
# For this example, we need to retrieve the actual Reddit posts or comment content
# In the Reddit dataset from PyTorch Geometric, the text data may be stored in `data.text` or another attribute
# If data.text is not available, you will need to find where the text data is stored or preprocess it.

# Example: Assuming `data.text` holds the posts/comments for each node (subreddit/user)
# Ensure that the number of nodes matches the number of text entries
node_texts = [data.text[i] for i in range(num_nodes)]  # This assumes `data.text` exists with node text content

# Check if node_texts matches the number of nodes
assert len(node_texts) == num_nodes, f"Expected {num_nodes} node texts, but found {len(node_texts)}"

# Extract embeddings for each node in the graph
node_embeddings = get_bert_embeddings(node_texts)

# Assign the extracted embeddings as the node features (data.x)
data.x = node_embeddings  # This is the feature matrix for the nodes

# Example: Define a simple GCN for node classification (or other tasks)
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Instantiate the GCN model (for example usage)
input_dim = node_embeddings.size(1)  # Number of features (embedding size)
hidden_dim = 64  # Hidden layer size
output_dim = 5  # Assume 5 categories for node classification (adjust accordingly)
model = GCN(input_dim, hidden_dim, output_dim)

# Example: Setting up training parameters and training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Train the model (simplified example for demonstration)
def train(model, data, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Only train on train_mask
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Example: Running training (ensure to have train_mask and labels in data)
train(model, data, optimizer, criterion)

# Now you have processed all the nodes, extracted their features using BERT, and trained a simple GCN.

