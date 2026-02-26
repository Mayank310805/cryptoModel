import os
import numpy as np
import pandas as pd
import math

base_folder = "generated_bins"

def calculate_entropy(data):
    if len(data) == 0:
        return 0
    byte_counts = np.bincount(data, minlength=256)
    probabilities = byte_counts / len(data)
    entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])
    return entropy

feature_rows = []

print("Extracting features...")

for algorithm in os.listdir(base_folder):
    algo_path = os.path.join(base_folder, algorithm)

    for file in os.listdir(algo_path):
        file_path = os.path.join(algo_path, file)

        with open(file_path, "rb") as f:
            byte_data = np.frombuffer(f.read(), dtype=np.uint8)

        entropy = calculate_entropy(byte_data)
        byte_freq = np.bincount(byte_data, minlength=256) / len(byte_data)
        zero_ratio = np.sum(byte_data == 0) / len(byte_data)
        file_size = len(byte_data)

        features = list(byte_freq)
        features.append(entropy)
        features.append(zero_ratio)
        features.append(file_size)
        features.append(algorithm)

        feature_rows.append(features)

# Create DataFrame
columns = [f"byte_{i}" for i in range(256)]
columns += ["entropy", "zero_ratio", "file_size", "label"]

df_features = pd.DataFrame(feature_rows, columns=columns)

df_features.to_csv("binary_features.csv", index=False)

print("Feature extraction complete.")
print("Saved as binary_features.csv")