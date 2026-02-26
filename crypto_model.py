import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Combine plaintext and ciphertext
df["combined"] = df["Plaintext"].astype(str) + " " + df["Ciphertext"].astype(str)

texts = df["combined"].values
labels = df["Algorithm"].values

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# ==========================
# CHARACTER TOKENIZATION
# ==========================
all_text = "".join(texts)
vocab = sorted(list(set(all_text)))
char_to_idx = {ch: i+1 for i, ch in enumerate(vocab)}  # 0 = padding
vocab_size = len(char_to_idx) + 1

max_len = 200

def encode_text(text):
    encoded = [char_to_idx.get(ch, 0) for ch in text]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [0] * (max_len - len(encoded))
    return encoded

X = np.array([encode_text(t) for t in texts])
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ==========================
# LSTM MODEL
# ==========================
class CryptoLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(CryptoLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

model = CryptoLSTM(
    vocab_size=vocab_size,
    embed_dim=64,
    hidden_dim=128,
    num_classes=len(np.unique(y))
)

# ==========================
# TRAINING
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
batch_size = 64

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    total_loss = 0

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = X_train[indices]
        batch_y = y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ==========================
# EVALUATION
# ==========================
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

print("\nAccuracy:", accuracy_score(y_test, predicted))
print("\nClassification Report:\n")
print(classification_report(y_test, predicted, target_names=label_encoder.classes_))

torch.save(model.state_dict(), "crypto_lstm_model.pth")
print("\nModel saved as crypto_lstm_model.pth")