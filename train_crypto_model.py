# ============================================
# Crypto Algorithm Classification - ResNet Model
# Single File Training Script
# ============================================
# Deep Residual Neural Network
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

# ===============================
# 1. LOAD DATA
# ===============================

print("Loading dataset...")

df = pd.read_csv("binary_features.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("Dataset loaded successfully.")
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ===============================
# 2. DEFINE RESIDUAL NETWORK
# ===============================

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out


class CryptoResNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()

        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)

        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


model = CryptoResNet(X_train.shape[1], len(np.unique(y)))

# ===============================
# 3. TRAINING SETUP
# ===============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

epochs = 30
batch_size = 128

print("\nStarting training...\n")

# ===============================
# 4. TRAINING LOOP
# ===============================

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    total_loss = 0

    for i in range(0, X_train.size()[0], batch_size):
        idx = permutation[i:i+batch_size]
        batch_x = X_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")

# ===============================
# 5. EVALUATION
# ===============================

print("\nEvaluating model...")

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

accuracy = accuracy_score(y_test, predicted)

print("\nTest Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, predicted, target_names=label_encoder.classes_))

# ===============================
# 6. SAVE MODEL
# ===============================

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoder': label_encoder
}, "crypto_resnet_model.pth")

print("\nModel saved as crypto_resnet_model.pth")
print("Training complete.")