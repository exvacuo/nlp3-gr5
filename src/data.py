from __future__ import annotations
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the AG News dataset from Hugging Face and return the train and test splits as pandas DataFrames.

    :return: A tuple containing the train and test DataFrames.
    """
    # Prefer `datasets` over `pandas.read_json(hf://...)`.
    # The HF hub often serves JSONL files compressed (gzip); `datasets` handles
    # this reliably, while pandas may try to decode a compressed stream as UTF-8.
    try:
        ds = load_dataset("sh0416/ag_news")
        train_ds = ds["train"]
        test_ds = ds["test"]
    except Exception:
        # Fallback for repos that only contain raw files.
        data_files = {"train": "train.jsonl", "test": "test.jsonl"}
        ds = load_dataset("json", data_files=data_files, repo_id="sh0416/ag_news")
        train_ds = ds["train"]
        test_ds = ds["test"]

    train = train_ds.to_pandas()
    test = test_ds.to_pandas()

    # Normalize common column naming differences.
    for df in (train, test):
        if "description" not in df.columns and "text" in df.columns:
            df.rename(columns={"text": "description"}, inplace=True)
    return train, test

def split_dataset(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the original train set into a new train set and a dev set.

    :param train: The original train DataFrame.
    :return: A tuple containing the new train and dev DataFrames.
    """

    x_all = train["description"].tolist()
    y_all = train["label"].tolist()

    x_train, x_dev, y_train, y_dev = train_test_split(
        x_all,
        y_all,
        test_size=0.1,
        random_state=1337,
        stratify=y_all,
        shuffle=True,
    )

    new_train = pd.DataFrame({"description": x_train, "label": y_train})
    dev = pd.DataFrame({"description": x_dev, "label": y_dev})

    return new_train, dev