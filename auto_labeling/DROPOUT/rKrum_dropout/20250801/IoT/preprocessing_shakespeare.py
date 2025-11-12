"""
    Accuracy is ~0.48 only use texts

     Shakespeare's Plays: Dialogues & Characters
    Explore the Genius of Shakespeare Through His Plays and Characters

    (e.g., tragedy, comedy)

    https://www.kaggle.com/datasets/guslovesmath/shakespeare-plays-dataset

    This comprehensive dataset includes every line from all of William Shakespeare’s plays, categorized by play, genre,
    character, and more. It is an invaluable resource for those interested in literary analysis,
    natural language processing, and the historical study of one of the most significant figures in
    English literature. The dataset consists of 108,093 rows and 9 columns, capturing lines from various plays by
    William Shakespeare. Here’s a breakdown of the dataset structure and its contents:


    Columns

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


    $PYTHONPATH=.:nsf python3 nsf/preprocessing_shakespeare.py

"""
import collections

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel

from ragg.utils import timer

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
        print(f'{i}/{len(tweets)}, {i / len(tweets) * 100:.2f}%', flush=True)
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
    in_file = 'data/SHAKESPEARE/shakespeare_plays.csv'
    # with open(in_file, 'rb') as file:
    #     result = chardet.detect(file.read())
    #     print(result)

    # nrows = 1000  # specify 'None' if want to read whole file
    nrows = None  # specify 'None' if want to read whole file
    # training.1600000.processed.noemoticon.csv may have more rows in reality,
    # but we are only loading/previewing the first 1000 rows
    # The most common encoding for CSV files is ISO-8859-1 (also known as latin1) or utf-16.
    df = pd.read_csv(in_file, delimiter=',', nrows=nrows, encoding='ISO-8859-1', header=0)
    df.dataframeName = 'shakespeare_plays.csv'
    df.columns = ['index', 'play_name', 'genre', 'character', 'act', 'scene', 'sentence', 'text', 'sex']
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns', df.columns.tolist())

    # labels, ids, dates, flags, users, texts = df['target'].tolist(), df['id'].tolist(), df['date'].tolist(), df[
    #     'flag'].tolist(), df['user'].tolist(), df['text'].tolist()
    labels, texts = df['genre'].tolist(), df['text'].tolist()
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
            if l == 'Comedy':
                l = 0
            elif l == 'Tragedy':
                l = 1
            elif l == 'History':
                l = 2
            else:
                raise ValueError(l)
            buffer.append(f'{x_string}, {l}\n')

            # Write in batches of `buffer_size`
            if len(buffer) >= buffer_size:
                print(f'{i}/{len(xs)}, {i / len(xs) * 100:.2f}', flush=True)
                f.writelines(buffer)
                buffer = []  # Reset buffer

        # Write any remaining lines
        if buffer:
            f.writelines(buffer)

    return csv_file


if __name__ == '__main__':
    # csv_file = preprocessing()

    reduce_dimension = False
    if reduce_dimension:
        csv_file = 'data/SHAKESPEARE/shakespeare_plays.csv_bert.csv'
        reduced_csv_file = f'{csv_file}_reduced.csv'
        df = pd.read_csv(csv_file, dtype=float, header=None)
        X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values

        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X)

        # Create a new DataFrame with reduced features + labels
        df_reduced = pd.DataFrame(X_reduced)
        df_reduced[len(df_reduced.columns)] = y  # Append labels as last column

        # Save the reduced dataset
        reduced_csv_file = f"{csv_file}_reduced.csv"
        df_reduced.to_csv(reduced_csv_file, index=False, header=False)

        print(f"Saved reduced dataset to {reduced_csv_file}")
