"""  Shakespeare Dataset:
    for text classification (e.g., tragedy, comedy)
    or Sentiment Analysis (e.g., positive, negative, neutral)

    # Upgrade to the latest compatible version
    pip install --upgrade torchtext

    # Or install a specific version that is known to be compatible with your PyTorch version
   pip install torchtext==0.15.0



    https://github.com/TalwalkarLab/leaf/blob/master/data/shakespeare/preprocess/get_data.sh

    https://www.gutenberg.org/files/100/old/old/1994-01-100.zip
    unzip 1994-01-100.zip
    rm 1994-01-100.zip
    mv 100.txt raw_data.txt

    download https://github.com/TalwalkarLab/leaf/blob/master/data/shakespeare/preprocess/preprocess_shakespeare.py
    % python3 preprocess_shakespeare.py raw_data.txt output_dir
    python3 preprocess_shakespeare.py raw_data.txt .


"""

import requests
import re
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from sklearn.model_selection import train_test_split

# Download the Shakespeare dataset (if not already downloaded)
url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
response = requests.get(url)

# Save the text content to a file
with open('shakespeare_works.txt', 'w') as f:
    f.write(response.text)

# Load the dataset (assuming it's text-based, line-by-line)
with open('shakespeare_works.txt', 'r') as file:
    lines = file.readlines()

# Example of how to label lines (in reality, this will be more complex, based on acts and scenes)
def label_line(line):
    if 'comedy' in line.lower():
        return 'Comedy'
    elif 'tragedy' in line.lower():
        return 'Tragedy'
    elif 'history' in line.lower():
        return 'History'
    elif 'romance' in line.lower():
        return 'Romance'
    else:
        return 'Other'


# Process the lines and label them
data = []
for line in lines:
    label = label_line(line)
    data.append({'text': line.strip(), 'label': label})

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Tokenization and Vectorization using torchtext
tokenizer = get_tokenizer('basic_english')

# Example of how to tokenize and create vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Splitting dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create vocab
train_data = train_df['text']
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Encode the text using the vocab
def encode_text(text):
    return [vocab[token] for token in tokenizer(text)]

train_df['encoded'] = train_df['text'].apply(encode_text)
test_df['encoded'] = test_df['text'].apply(encode_text)

# Now convert the labels to numerical format
label_map = {'Comedy': 0, 'Tragedy': 1, 'History': 2, 'Romance': 3, 'Other': 4}
train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# Convert to list of tuples (text, label)
train_data = list(zip(train_df['encoded'], train_df['label']))
test_data = list(zip(test_df['encoded'], test_df['label']))

# Define a simple collate_fn for padding sequences
def collate_batch(batch):
    label_list, text_list = zip(*batch)

    # Print out the structure of each text item
    for i, text in enumerate(text_list):
        print(f"Text {i}: {text}, Type: {type(text)}")  # Debugging line to print structure

    # Ensure text is a list or sequence and filter out invalid entries
    # text_list = [torch.tensor(text) for text in text_list if isinstance(text, (list, str)) and len(text) > 0]
    text_list = [torch.tensor([text]) for text in text_list if isinstance(text, int)]

    # Print out the filtered text_list
    print("Filtered text_list:", text_list)  # Debugging line to check filtered contents

    if len(text_list) == 0:
        raise ValueError("text_list is empty after filtering. Check the tokenization or data processing steps.")

    # Ensure labels are correctly formatted as a tensor
    label_tensor = torch.tensor(label_list, dtype=torch.int64)

    # Use pad_sequence for text tensors with padding
    text_tensor = torch.nn.utils.rnn.pad_sequence(text_list, padding_value=0, batch_first=True)

    return text_tensor, label_tensor


# Create DataLoader
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_batch)

# Define the Text Classifier model
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.mean(dim=1)  # Simple mean pooling
        x = F.relu(self.fc1(embedded))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define input, hidden, and output dimensions
input_dim = len(vocab)
hidden_dim = 128
output_dim = len(label_map)

# Instantiate the model
model = TextClassifier(input_dim, hidden_dim, output_dim)

# Loss function and optimizer
optimizer = Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in train_dataloader:
        text, labels = batch

        optimizer.zero_grad()

        # Forward pass
        output = model(text, None)

        # Compute loss
        loss = criterion(output, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Prediction
        predictions = torch.argmax(output, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += len(labels)

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}, Accuracy: {correct_predictions / total_predictions}')

# Evaluate the model
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in test_dataloader:
        text, labels = batch

        output = model(text, None)

        predictions = torch.argmax(output, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += len(labels)

print(f'Test Accuracy: {correct_predictions / total_predictions}')
