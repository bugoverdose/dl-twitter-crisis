#!/usr/bin/env python3
"""
- Loads multiple .tsv files from a folder, concatenates them, and maps labels to binary.
- Supports two architectures via --arch: "transformer" (HuggingFace) and "cnn" (TextCNN).
"""
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_text(text: str) -> str:
    """
    Cleans CrisisBench text by:
    - Lowercasing
    - Removing URLs, hashtags, numbers, punctuation, symbols, and non-ASCII chars
    - Replacing removed characters with whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|pic\.twitter\.com/\S+", " ", text)

    # Remove hashtags and mentions (keep the word if desired: replace '#word' with 'word')
    text = re.sub(r"#", "", text)
    text = re.sub(r"@\w+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation and symbols (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove invisible and non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_crisisbench(filepath: str) -> pd.DataFrame:
    """
    Loads a single .tsv CrisisBench split file.
    Expected columns: id, event, source, text, lang, lang_conf, class_label
    Drops language columns and applies text preprocessing.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)
    expected_cols = {"id", "event", "source", "text", "lang", "lang_conf", "class_label"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{filepath} is missing expected columns: {missing}")

    # Drop unwanted columns
    df = df.drop(columns=["lang", "lang_conf", "id"])

    # Clean up
    df["text"] = df["text"].astype(str).str.strip().map(clean_text)
    df["class_label"] = df["class_label"].astype(str).str.strip()
    df = df.replace({"": np.nan}).dropna(subset=["text", "class_label"]).reset_index(drop=True)

    return df

def main():
    set_seed(42)

    # load data
    train_df = load_crisisbench("./data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_train.tsv")
    val_df = load_crisisbench("./data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_dev.tsv")
    test_df = load_crisisbench("./data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_test.tsv")

    print(train_df.head())
    print(val_df.head())
    print(test_df.head())

if __name__ == "__main__":
    main()
