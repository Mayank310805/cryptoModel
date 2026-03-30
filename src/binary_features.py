"""
binary_features.py
------------------
Feature extraction for uploaded firmware binary files.
Produces the same 259-dimensional feature vector used to train crypto_resnet_model.pth:
  - 256 byte-frequency bins (normalised)
  - entropy
  - zero_ratio
  - file_size
Also performs a lightweight crypto constant signature scan.
"""
import os
import math
import numpy as np


# -------------------------------------------------------
# Known crypto byte signatures (first bytes of constants)
# -------------------------------------------------------
SIGNATURES = {
    "AES": [
        bytes([0x63, 0x7c, 0x77, 0x7b, 0xf2]),   # AES S-box start
        bytes([0x52, 0x09, 0x6a, 0xd5, 0x30]),   # AES S-box mid
    ],
    "DES": [
        bytes([0x3b, 0x4a, 0x9c, 0x5f]),          # DES PC-1 table fragment
        bytes([0x0e, 0x11, 0x1b, 0x00, 0x0e]),   # DES PC-2 table fragment
    ],
    "Blowfish": [
        bytes([0x24, 0x3f, 0x6a, 0x88, 0x85]),   # Blowfish P-array (Pi digits)
        bytes([0xa3, 0x08, 0xd3, 0x13, 0x19]),
    ],
    "RC4": [
        bytes([0x00, 0x01, 0x02, 0x03, 0x04,
               0x05, 0x06, 0x07, 0x08, 0x09]),   # RC4 KSA initialisation pattern
    ],
    "ChaCha20": [
        b"expand 32-byte k",                      # ChaCha20 constant string
        b"expand 16-byte k",
    ],
    "RSA": [
        bytes([0x30, 0x82]),                      # DER/ASN.1 RSA key header
        b"-----BEGIN RSA",
        b"-----BEGIN PUBLIC KEY",
    ],
}


def _calculate_entropy(byte_data: np.ndarray) -> float:
    """Shannon entropy of a byte array."""
    if len(byte_data) == 0:
        return 0.0
    counts = np.bincount(byte_data, minlength=256)
    probs = counts / len(byte_data)
    entropy = -np.sum([p * math.log2(p) for p in probs if p > 0])
    return float(entropy)


def _scan_signatures(raw_bytes: bytes):
    """
    Scan for known crypto constants.
    Returns (matched_algo_name | None).
    """
    for algo, sigs in SIGNATURES.items():
        for sig in sigs:
            if sig in raw_bytes:
                return algo
    return None


def extract_features(file_path: str):
    """
    Extract 259-dimensional feature vector from a binary file.

    Returns
    -------
    features : list of float  (length 259)
    crypto_flag : bool        True if high-entropy or signature matched
    signature_algo : str|None  Name of the matched algorithm, or None
    """
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    byte_data = np.frombuffer(raw_bytes, dtype=np.uint8)

    if len(byte_data) == 0:
        features = [0.0] * 259
        return features, False, None

    # 256 normalised byte-frequency bins
    byte_freq = (np.bincount(byte_data, minlength=256) / len(byte_data)).tolist()

    entropy = _calculate_entropy(byte_data)
    zero_ratio = float(np.sum(byte_data == 0) / len(byte_data))
    file_size = float(len(byte_data))

    features = byte_freq + [entropy, zero_ratio, file_size]   # 259 values

    # Crypto detection heuristics
    signature_algo = _scan_signatures(raw_bytes)
    crypto_flag = (entropy >= 7.0) or (signature_algo is not None)

    return features, crypto_flag, signature_algo
