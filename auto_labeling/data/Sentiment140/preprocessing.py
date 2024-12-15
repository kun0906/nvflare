"""
https://www.kaggle.com/datasets/kazanova/sentiment140/data
"""
import collections
import os  # accessing directory structure
import pickle

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import BertTokenizer, BertModel

from auto_labeling.utils import timer

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

@timer
def tweet2vec(tweet):
    # # Tokenize tweet and get BERT embeddings
    # inputs = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # outputs = model(**inputs)
    #
    # # Get the embeddings for the [CLS] token (first token)
    # embedding = outputs.last_hidden_state[0][0].detach().numpy()
    def sliding_window_tokenize(tweet, tokenizer, window_size=512):
        tokens = tokenizer.encode(tweet, truncation=True, padding=False)  # Encoding the tweet
        if len(tokens) > window_size:
            print(f'the tweet has {len(tokens)} tokens.')
        embeddings = []

        for i in range(0, len(tokens), window_size):
            chunk = tokens[i:i + window_size]
            # Tokenizer returns a dictionary, make sure you include padding and truncation
            inputs = tokenizer.prepare_for_model(
                chunk,
                truncation=True,
                padding='max_length',  # You can also use True for dynamic padding
                max_length=window_size,
                return_tensors='pt'  # Tensors for PyTorch
            )

            # Ensure input tensor is correctly formatted with a batch dimension (2D)
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)  # Add batch dimension
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)  # Add batch dimension

            outputs = model(**inputs)
            chunk_embedding = outputs.last_hidden_state[0][0].detach().numpy()  # [CLS] token
            embeddings.append(chunk_embedding)

        # You can average or combine embeddings from different chunks if needed
        return np.mean(embeddings, axis=0)  # Example: average embeddings

    embedding = sliding_window_tokenize(tweet, tokenizer)
    # print(embedding)
    # print(embedding.shape)        # (768, )
    return embedding

@timer
def preprocessing():
    in_file = 'training.1600000.processed.noemoticon.csv'
    # with open(in_file, 'rb') as file:
    #     result = chardet.detect(file.read())
    #     print(result)

    nrows = None  # specify 'None' if want to read whole file
    # training.1600000.processed.noemoticon.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    # The most common encoding for CSV files is ISO-8859-1 (also known as latin1) or utf-16.
    df = pd.read_csv(in_file, delimiter=',', nrows=nrows, encoding='ISO-8859-1', header=None)
    df.dataframeName = 'training.1600000.processed.noemoticon.csv'
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns', df.columns.tolist())

    labels, ids, dates, flags, users, texts = df['target'].tolist(), df['id'].tolist(), df['date'].tolist(), df[
        'flag'].tolist(), df['user'].tolist(), df['text'].tolist()
    print(collections.Counter(labels))
    data = {}

    for i, user in enumerate(users):
        text, label = texts[i], labels[i]
        item = (text, label)
        if user not in data:
            data[user] = [item]
        else:
            data[user].append(item)

    vs = [(k, len(v)) for k, v in data.items()]
    print(len(data), sorted(vs, key=lambda kv: kv[1], reverse=True)[:10])

    num_clients = 10
    in_dir = 'data'
    os.makedirs(in_dir, exist_ok=True)
    for c in range(num_clients):
        users_ = np.random.choice(list(data.keys()), size=100, replace=False)  # each client has 10 users' data
        client_data = [(k, data[k]) for k in users_]
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)

    for c in range(num_clients):
        print(f'\nclient {c}...')
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        client_data_file = f'{in_dir}/{c}.pkl'
        X = []
        Y = []
        Y_names = []
        for user_name_, vs in client_data:
            for tweet, y in vs:  # some client may have multiple tweets
                x = tweet2vec(tweet)
                X.append(x)
                Y.append(y)
                Y_names.append(user_name_)

        client_data = (X, Y, Y_names)
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)


def check_client_data():
    num_clients = 10
    in_dir = 'data'
    for c in range(num_clients):
        client_data_file = f'{in_dir}/{c}.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        print(c, client_data)


if __name__ == '__main__':
    preprocessing()
    check_client_data()
