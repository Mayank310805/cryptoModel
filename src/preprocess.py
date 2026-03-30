"""
preprocess.py
-------------
Text preprocessing utilities for the LSTM crypto classifier.
Reproduces the same char-level tokenization used in crypto_model.py
so that user-supplied plaintext/ciphertext can be fed to the LSTM.
"""
import os
import pandas as pd
import numpy as np


MAX_LEN = 200


def load_vocab(dataset_path: str):
    """
    Rebuild the character vocabulary from dataset.csv.
    Returns char_to_idx dict (same construction as in crypto_model.py).
    """
    df = pd.read_csv(dataset_path)
    df["combined"] = df["Plaintext"].astype(str) + " " + df["Ciphertext"].astype(str)
    all_text = "".join(df["combined"].values)
    vocab = sorted(list(set(all_text)))
    char_to_idx = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = padding
    return char_to_idx


def encode_text(text: str, char_to_idx: dict, max_len: int = MAX_LEN):
    """
    Encode a raw string into a fixed-length integer sequence (padded/truncated).
    """
    encoded = [char_to_idx.get(ch, 0) for ch in text]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [0] * (max_len - len(encoded))
    return encoded


def prepare_input(plaintext: str, ciphertext: str, char_to_idx: dict):
    """
    Combine plaintext+ciphertext then encode, matching training format.
    Returns numpy array of shape (1, MAX_LEN).
    """
    combined = plaintext + " " + ciphertext
    encoded = encode_text(combined, char_to_idx)
    return np.array([encoded], dtype=np.int64)
