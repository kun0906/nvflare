"""
    https://www.kaggle.com/datasets/kazanova/sentiment140/data


    This is the sentiment140 dataset.
    It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment .
    It contains the following 6 fields:

    target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
    ids: The id of the tweet ( 2087)
    date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
    flag: The query (lyx). If there is no query, then this value is NO_QUERY.
    user: the user that tweeted (robotickilldozr)
    text: the text of the tweet (Lyx is cool)
    The official link regarding the dataset with resources about how it was generated is here
    The official paper detailing the approach is here

    According to the creators of the dataset:

    "Our approach was unique because our training data was automatically created, as opposed to having humans manual annotate tweets. In our approach, we assume that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. We used the Twitter Search API to collect these tweets by using keyword search"

    citation: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.


    $PYTHONPATH=.:nsf python3 nsf/preprocessing_sentiment140.py

"""
import collections
import os  # accessing directory structure
import pickle

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from transformers import BertTokenizer, BertModel

from utils import timer

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
model = BertModel.from_pretrained('bert-base-uncased')

# Ensure the model and input tensors use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)  # Move model to GPU


def tweets2vecs(tweets, batch_size=5000):
    all_embeddings = []

    for i in range(0, len(tweets), batch_size):
        # if i % 10000 == 0:
        #     print(f'{i}/{len(tweets)}', flush=True)
        print(f'{i}/{len(tweets)}, {i / len(tweets) * 100:.2f}', flush=True)
        batch = tweets[i:i + batch_size]  # Slice batch
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # [CLS] token
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)  # Combine all batch results


# @timer
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
    in_file = 'data/Sentiment140/training.1600000.processed.noemoticon.csv'
    # with open(in_file, 'rb') as file:
    #     result = chardet.detect(file.read())
    #     print(result)

    # nrows = 1000  # specify 'None' if want to read whole file
    nrows = None  # specify 'None' if want to read whole file
    # training.1600000.processed.noemoticon.csv may have more rows in reality,
    # but we are only loading/previewing the first 1000 rows
    # The most common encoding for CSV files is ISO-8859-1 (also known as latin1) or utf-16.
    df = pd.read_csv(in_file, delimiter=',', nrows=nrows, encoding='ISO-8859-1', header=None)
    df.dataframeName = 'training.1600000.processed.noemoticon.csv'
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns', df.columns.tolist())

    labels, ids, dates, flags, users, texts = df['target'].tolist(), df['id'].tolist(), df['date'].tolist(), df[
        'flag'].tolist(), df['user'].tolist(), df['text'].tolist()
    print(collections.Counter(labels))
    # data = {}
    # print(f'Number of Users: {len(users)}')    # {collections.Counter(users)}
    # for i, user in enumerate(users):
    #     text, label = texts[i], labels[i]
    #     item = text
    #     if user not in data:
    #         data[user] = [item]
    #     else:
    #         data[user].append(item)
    # vs = [(k, len(v)) for k, v in data.items()]
    # print(len(data), ', top 10 User distribution: ', sorted(vs, key=lambda kv: kv[1], reverse=True)[:10])
    print('text length distribution: ', collections.Counter([len(t) for t in texts]))

    # csv_file = f'{in_file}_bert.csv'
    # with open(csv_file, 'w') as f:
    #     for i in range(nRow):
    #         if i % 5000 == 0:
    #             print(f'{i}/{nRow}')
    #         target, id, date, flag, user, text = df.iloc[i]
    #         x = tweet2vec(text)
    #         x_string = ', '.join([f"{v:.5f}" for v in x])
    #         f.write(f'{x_string}, {target}\n')

    xs = tweets2vecs(texts)

    # csv_file = f'{in_file}_bert.csv'
    # with open(csv_file, 'w') as f:
    #     for i, (x, l) in enumerate(zip(xs, labels)):
    #         if i % 5000 == 0:
    #             print(f'{i}/{nRow}')
    #         x_string = ','.join([f"{v:.5f}" for v in x])
    #         f.write(f'{x_string}, {l}\n')

    csv_file = f'{in_file}_bert.csv'
    buffer_size = 50000
    with open(csv_file, 'w') as f:
        buffer = []
        for i, (x, l) in enumerate(zip(xs, labels)):
            x_string = ','.join([f"{v:.5f}" for v in x])
            if l == 0:
                pass
            elif l == 4:
                l = 1
            else:
                raise ValueError(l)
            buffer.append(f'{x_string}, {l}\n')

            # Write in batches of `buffer_size`
            if len(buffer) >= buffer_size:
                print(f'{i}/{len(xs)}, {i/len(xs)*100:.2f}', flush=True)
                f.writelines(buffer)
                buffer = []  # Reset buffer

        # Write any remaining lines
        if buffer:
            f.writelines(buffer)

    return csv_file


if __name__ == '__main__':
    preprocessing()
