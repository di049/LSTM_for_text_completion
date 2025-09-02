import re
from transformers import AutoTokenizer
import pickle

def load_data(data_path='data/tweets.txt'):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()

    data_split = data.split('\n')

    return data_split


def make_clean_data(data_path='data/tweets.txt', re_pattern='@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'):

    data_split = load_data(data_path)

    clean_data = ''

    for tweet in data_split:
        clean_tweet = re.sub(re_pattern, ' ', tweet.lower()).strip()
        clean_data += clean_tweet + '\n'

    with open('data/tweets_processed.txt', 'w', encoding='utf-8') as f:
        f.write(clean_data)


def tokenize_data(data_path='data/tweets_processed.txt', tokenizer_name='distilbert/distilgpt2'):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data_split = load_data(data_path)

    tokenized_data = []

    for tweet in data_split:
        tokenized_data.append(tokenizer.encode(tweet))

    with open('data/tweets_processed_tokenized.pkl', 'wb') as f:
        pickle.dump(tokenized_data, f)

#    return tokenized_data

def train_test_val_split(data_path='data/tweets_processed_tokenized.pkl'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    test = data[:round(len(data) * 0.1)]
    val = data[round(len(data) * 0.1) : (2 * round(len(data) * 0.1))]
    train = data[(2 * round(len(data) * 0.1)):]

    return train, test, val

