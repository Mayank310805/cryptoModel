import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
# LOAD DATA
# ===============================
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

print("Dataset Loaded Successfully")

# ===============================
# 1️⃣ RANDOM FOREST
# ===============================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_acc)

# ===============================
# 2️⃣ XGBOOST
# ===============================
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("XGBoost Accuracy:", xgb_acc)

# ===============================
# 3️⃣ NEURAL NETWORK (MLP)
# ===============================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

mlp = MLP(X_train.shape[1], len(np.unique(y)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

epochs = 20
batch_size = 128

for epoch in range(epochs):
    perm = torch.randperm(X_train_t.size()[0])
    total_loss = 0

    for i in range(0, X_train_t.size()[0], batch_size):
        idx = perm[i:i+batch_size]
        batch_x = X_train_t[idx]
        batch_y = y_train_t[idx]

        optimizer.zero_grad()
        outputs = mlp(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(X_test_t)
    _, mlp_pred = torch.max(mlp_pred, 1)

mlp_acc = accuracy_score(y_test, mlp_pred)
print("Neural Network Accuracy:", mlp_acc)

# ===============================
# RESULTS SUMMARY
# ===============================
print("\n===== FINAL COMPARISON =====")
print(f"Random Forest Accuracy : {rf_acc:.4f}")
print(f"XGBoost Accuracy       : {xgb_acc:.4f}")
print(f"Neural Network Accuracy: {mlp_acc:.4f}")

# Confusion Matrix (Best Model)
best_pred = rf_pred
best_name = "Random Forest"

if xgb_acc > rf_acc and xgb_acc > mlp_acc:
    best_pred = xgb_pred
    best_name = "XGBoost"
elif mlp_acc > rf_acc and mlp_acc > xgb_acc:
    best_pred = mlp_pred
    best_name = "Neural Network"

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()