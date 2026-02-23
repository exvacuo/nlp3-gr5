from __future__ import annotations
from typing import Any
import re
import pandas as pd
import numpy as np
from collections import Counter

PAD = "<pad>"
UNK = "<unk>"


def clean_text(text: str) -> str:
    """
    Clean the input text by converting it to lowercase, removing non-alphanumeric characters, and stripping leading/trailing whitespace.
    
    :param text: The input text string to be cleaned
    :return: A cleaned version of the input text string
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize the input text by splitting it into individual words based on whitespace.
    
    :param text: The input text string to be tokenized
    :return: A list of tokens (words) extracted from the input text string
    """
    return text.split()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by cleaning the text in the "description" column and tokenizing it into a new "tokens" column.

    :param df: A pandas DataFrame containing a "description" column with text data to be preprocessed
    :return: A new DataFrame with the original "description" column cleaned and a new "tokens" column containing lists of tokens
    """
    df["description"] = df["description"].apply(clean_text)
    df["tokens"] = df["description"].apply(tokenize_text)
    return df


def build_vocab(token_lists: list[list[str]], max_tokens: int = 20000) -> dict[str, int]:
    """
    Build a vocabulary dictionary mapping tokens to unique integer indices based on the most common tokens in the input list of token lists, with special tokens for padding and unknown words.
    
    :param token_lists: A list of lists of tokens (words) from the dataset
    :param max_tokens: The maximum number of tokens to include in the vocabulary (including special tokens)
    :return: A dictionary mapping tokens to unique integer indices, with special tokens for padding and unknown words"""
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    most_common = counter.most_common(max_tokens - 2)

    vocab = {PAD: 0, UNK: 1}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i

    return vocab


def encode_and_pad(token_lists: list[list[str]], vocab: dict[str, int], max_length: int = 128) -> np.ndarray:
    """
    Encode the input list of token lists into a 2D numpy array of integer indices based on the provided vocabulary, and pad or truncate each sequence to a specified maximum length.
    
    :param token_lists: A list of lists of tokens (words) from the dataset
    :param vocab: A dictionary mapping tokens to unique integer indices
    :param max_length: The maximum length of each sequence after encoding and padding/truncating
    :return: A 2D numpy array of shape (num_samples, max_length) containing integer indices for each token in the input sequences, with padding applied as needed
    """
    encoded = []

    for tokens in token_lists:
        ids = [vocab.get(token, vocab[UNK]) for token in tokens]
        ids = ids[:max_length]

        if len(ids) < max_length:
            ids += [vocab[PAD]] * (max_length - len(ids))

        encoded.append(ids)

    return np.array(encoded)


def feature_engineering(
    dataset: pd.DataFrame,
    column_name: str,
    *,
    max_tokens: int = 20000,
    output_sequence_length: int = 128,
    vocab: dict | None = None,
):
    token_lists = dataset["tokens"].tolist()

    if vocab is None:
        vocab = build_vocab(token_lists, max_tokens)

    X = encode_and_pad(token_lists, vocab, output_sequence_length)

    return X, vocab
