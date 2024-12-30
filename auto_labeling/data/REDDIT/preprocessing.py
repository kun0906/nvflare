""" Reddit Dataset
    from datasets import load_dataset
    ds = load_dataset("icedwind/reddit_dataset_92")

    https://huggingface.co/datasets/icedwind/reddit_dataset_92

    Data Fields
    text (string): The main content of the Reddit post or comment.
    label (string): Sentiment or topic category of the content, i.e., negative vs. positive.
    dataType (string): Indicates whether the entry is a post or a comment.
    communityName (string): The name of the subreddit where the content was posted.
    datetime (string): The date when the content was posted or commented.
    username_encoded (string): An encoded version of the username to maintain user privacy.
    url_encoded (string): An encoded version of any URLs included in the content.

    cd data/REDDIT
    PYTHONPATH=. python3 preprocessing.py
"""

import os  # accessing directory structure
import pickle
import time

import numpy as np  # linear algebra
from transformers import BertTokenizer, BertModel

# Timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# @timer
def text2vec(text):
    # # Tokenize tweet and get BERT embeddings
    # inputs = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # outputs = model(**inputs)
    #
    # # Get the embeddings for the [CLS] token (first token)
    # embedding = outputs.last_hidden_state[0][0].detach().numpy()
    def sliding_window_tokenize(text, tokenizer, window_size=512):
        tokens = tokenizer.encode(text, truncation=True, padding=True)  # Encoding the tweet
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

    embedding = sliding_window_tokenize(text, tokenizer)
    # print(embedding)
    # print(embedding.shape)        # (768, )
    return embedding


@timer
def texts2vec(texts):
    # Tokenize and pad the batch of texts
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Pass the input tokens to the model
    outputs = model(**inputs)

    # Extract [CLS] token embeddings for all texts in the batch
    # outputs.last_hidden_state has shape (batch_size, sequence_length, hidden_size)
    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()  # (batch_size, 768)

    # Optional: Print the shape of the embeddings
    print(embedding.shape)  # Expected: (len(texts), 768)

    # print(embedding.shape)        # (768, )
    return embedding


@timer
def preprocessing():
    in_file = f'{in_dir}/reddit_dataset_1.pkl'
    os.makedirs(in_dir, exist_ok=True)
    if not os.path.exists(in_file):
        # pip install datasets
        from datasets import load_dataset
        ds = load_dataset("icedwind/reddit_dataset_92")  # ~/.cache/huggingface/datasets/
        """
            text (string): The main content of the Reddit post or comment.
            label (string): Sentiment or topic category of the content.
            dataType (string): Indicates whether the entry is a post or a comment.
            communityName (string): The name of the subreddit where the content was posted.
            datetime (string): The date when the content was posted or commented.
            username_encoded (string): An encoded version of the username to maintain user privacy.
            url_encoded (string): An encoded version of any URLs included in the content.
        """
        N, d = ds['train'].shape
        print(ds['train'].shape, ds['train'].column_names)
        # print([ds['train'][col][:2] for col in ds['train'].column_names]) # takes too long time, why?
        nrows = int(0.01 * N)  # 1% percent data
        print('nrows', nrows)
        labels = ds['train']['label'][:nrows]
        texts = ds['train']['text'][:nrows]
        data = (texts, labels)

        with open(in_file, 'wb') as f:
            pickle.dump(data, f)

    else:
        with open(in_file, 'rb') as f:
            texts, labels = pickle.load(f)

    # print(collections.Counter(labels)[:10])
    data = {}
    for i, label in enumerate(labels):
        text = texts[i]
        if len(text.split(' ')) < 5: continue  # ignore the text less than 5 words.
        if label not in data:
            data[label] = [text]
        else:
            data[label].append(text)

    top_res = sorted(data.items(), key=lambda kv: len(kv[1]), reverse=True)[:num_clients]
    top_labels = [l for l, vs in top_res]
    top_cnts = [len(vs) for l, vs in top_res]
    print(top_labels, top_cnts)
    max_len = max(len(text.split(' ')) for text in texts)
    print(f'max text length is {max_len}')
    # labels_set = [l for l, c in top_labels]

    for c, l in enumerate(top_labels):
        vs = data[l]
        replace = False if len(vs) >= 1000 else True
        print(l, len(vs), replace)
        client_texts = np.random.choice(list(vs), size=1000, replace=replace)
        y = [l] * len(client_texts)
        client_data = (list(client_texts), y)
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)

    for c in range(num_clients):
        print(f'\nclient {c}...')
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        client_data_file = f'{in_dir}/{c}.pkl'
        client_texts, ys = client_data
        # X = texts2vec(client_texts)
        X = np.array([text2vec(text) for text in client_texts]) # memory concern
        Y = [c] * len(client_texts)
        Y_names = ys

        client_data = (X, Y, Y_names)
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)


def check_client_data():
    for c in range(num_clients):
        client_data_file = f'{in_dir}/{c}.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        print(c, client_data)


if __name__ == '__main__':
    in_dir = 'data'
    num_clients = 10
    # preprocessing()
    check_client_data()
