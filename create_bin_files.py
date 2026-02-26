import pandas as pd
import os

# Load dataset
df = pd.read_csv("dataset.csv")

# Create base folder
base_folder = "generated_bins"
os.makedirs(base_folder, exist_ok=True)

print("Generating .bin files...")

for index, row in df.iterrows():
    algorithm = str(row["Algorithm"])
    ciphertext = str(row["Ciphertext"])

    # Create folder per algorithm
    algo_folder = os.path.join(base_folder, algorithm)
    os.makedirs(algo_folder, exist_ok=True)

    # Convert ciphertext string to bytes
    byte_data = ciphertext.encode("utf-8")

    # File name
    filename = f"sample_{index}.bin"
    file_path = os.path.join(algo_folder, filename)

    # Write binary file
    with open(file_path, "wb") as f:
        f.write(byte_data)

print("All .bin files created successfully!")