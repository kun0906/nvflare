"""   Shakespeare's Plays: Dialogues & Characters
    Explore the Genius of Shakespeare Through His Plays and Characters

    (e.g., tragedy, comedy)

    https://www.kaggle.com/datasets/guslovesmath/shakespeare-plays-dataset

    genre: The genre of the play (Comedy, History, Tragedy).

    The columns:
    play_name: The name of the play.
    genre: The genre of the play (Comedy, History, Tragedy).
    character: The character who delivers the line.
    act: Act number in the play.
    scene: Scene number in the act.
    sentence: Line number in the scene.
    text: The text of the dialogue.
    sex: The gender of the character, reflecting Shakespeare's diverse cast.



"""

import collections
import os  # accessing directory structure
import pickle

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import BertTokenizer, BertModel

from ragg.utils import timer

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


@timer
def text2vec(text):
    # # Tokenize tweet and get BERT embeddings
    # inputs = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # outputs = model(**inputs)
    #
    # # Get the embeddings for the [CLS] token (first token)
    # embedding = outputs.last_hidden_state[0][0].detach().numpy()
    def sliding_window_tokenize(text, tokenizer, window_size=512):
        tokens = tokenizer.encode(text, truncation=True, padding=False)  # Encoding the tweet
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
    in_file = 'shakespeare_plays.csv'
    """
        play_name: The name of the play.
        genre: The genre of the play (Comedy, History, Tragedy).
        character: The character who delivers the line.
        act: Act number in the play.
        scene: Scene number in the act.
        sentence: Line number in the scene.
        text: The text of the dialogue.
        sex: The gender of the character, reflecting Shakespeare's diverse cast.
    """
    nrows = None  # specify 'None' if want to read whole file
    # The most common encoding for CSV files is ISO-8859-1 (also known as latin1) or utf-16.
    df = pd.read_csv(in_file, delimiter=',', nrows=nrows, encoding='ISO-8859-1', header=None)
    df.columns = ['index', 'play_name', 'genre', 'character', 'act', 'scene', 'sentence', 'text', 'sex']
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns', df.columns.tolist())

    labels, texts = df['genre'].tolist(), df['text'].tolist()
    print(collections.Counter(labels))
    max_len = max(len(text.split(' ')) for text in texts)
    print(f'max text length is {max_len}')
    labels_set = set(labels)
    labels_set.remove("genre")

    data = {}
    for i, label in enumerate(labels):
        text = texts[i]
        if label not in data:
            data[label] = [text]
        else:
            data[label].append(text)

    vs = [(k, len(v)) for k, v in data.items()]
    print(len(data), sorted(vs, key=lambda kv: kv[1], reverse=True)[:10])

    # num_clients = 9  # each 3 clients drawn samples from 1 class
    os.makedirs(in_dir, exist_ok=True)
    for c in range(num_clients):
        if c % 3 == 0:
            key = 'Comedy'
        elif c % 3 == 1:
            key = 'Tragedy'
        else:  # c%3 == 2
            key = 'History'
        vs = data[key]
        label = key
        client_texts = np.random.choice(list(vs), size=1000, replace=False)
        y = [label] * len(client_texts)
        client_data = (list(client_texts), y)
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'wb') as f:
            pickle.dump(client_data, f)

    label2int = {'Comedy': 0, 'Tragedy': 1, 'History': 2}
    for c in range(num_clients):
        print(f'\nclient {c}...')
        client_data_file = f'{in_dir}/{c}_raw.pkl'
        with open(client_data_file, 'rb') as f:
            client_data = pickle.load(f)

        client_data_file = f'{in_dir}/{c}.pkl'
        X = []
        Y = []
        Y_names = []
        client_texts, ys = client_data
        X = texts2vec(client_texts)
        Y = [label2int[l] for l in ys]
        Y_names = ys
        # for text, y in zip(client_texts, y):
        #     x = text2vec(text)  # For each text, we preprocess it separately in case some of them are too long (> 512).
        #     X.append(x)
        #     Y.append(label2int[y])
        #     Y_names.append(y)  #

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
    num_clients = 9
    preprocessing()
    check_client_data()
