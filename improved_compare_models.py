# ============================================================
# improved_compare_models.py
# Crypto Algorithm Classification - Enhanced Feature Set + Tuned Models
# ============================================================
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ============================================================
# 1. LOAD EXISTING FEATURES
# ============================================================
print("Loading base features from binary_features.csv ...")
df = pd.read_csv("binary_features.csv")

base_X = df.iloc[:, :-1].values   # 259 features
y_raw  = df.iloc[:,  -1].values

# ============================================================
# 2. ENRICH FEATURES FROM RAW BINARY FILES
# ============================================================
# We add derived statistics per-sample from the byte_0..byte_255 columns
# which represent the normalised byte frequency histogram.

print("Engineering richer features ...")

byte_cols = [f"byte_{i}" for i in range(256)]
byte_freq = df[byte_cols].values          # shape (N, 256)

# a) Mean / Std / Skewness of byte frequency distribution
freq_mean  = byte_freq.mean(axis=1, keepdims=True)              # how flat
freq_std   = byte_freq.std(axis=1, keepdims=True)               # deviation from flat
freq_max   = byte_freq.max(axis=1, keepdims=True)               # dominant byte
freq_min   = byte_freq.min(axis=1, keepdims=True)
freq_range = freq_max - freq_min                                  # spread

# Skewness of the frequency histogram
freq_skew = (((byte_freq - freq_mean) ** 3).mean(axis=1, keepdims=True) /
             (freq_std ** 3 + 1e-9))

# Kurtosis
freq_kurt = (((byte_freq - freq_mean) ** 4).mean(axis=1, keepdims=True) /
             (freq_std ** 4 + 1e-9))

# b) Chi-squared statistic vs uniform distribution (1/256 each bin)
uniform = 1.0 / 256.0
chi2 = ((byte_freq - uniform) ** 2 / (uniform + 1e-9)).sum(axis=1, keepdims=True)

# c) Number of zero-frequency bytes (bytes that never appear)
zero_freq_count = (byte_freq == 0).sum(axis=1, keepdims=True).astype(float)

# d) Byte range quartiles: Q1, Q2, Q3 of freq histogram
q1   = np.percentile(byte_freq, 25, axis=1, keepdims=True)
q2   = np.percentile(byte_freq, 50, axis=1, keepdims=True)
q3   = np.percentile(byte_freq, 75, axis=1, keepdims=True)
iqr  = q3 - q1

# e) Top-16 / bottom-16 byte freq sums (asymmetry between low and high bytes)
low_sum  = byte_freq[:, :16].sum(axis=1, keepdims=True)
high_sum = byte_freq[:, 240:].sum(axis=1, keepdims=True)
mid_sum  = byte_freq[:, 64:192].sum(axis=1, keepdims=True)

# f) Printable ASCII ratio (bytes 0x20-0x7E)
printable_sum = byte_freq[:, 0x20:0x7F].sum(axis=1, keepdims=True)

# g) Entropy of the byte distribution variance (already have global entropy)
#    Add local "entropy surprise": deviation of stored entropy from expected
#    uniform entropy (8.0 bits)
global_entropy = df[["entropy"]].values          # already in base_X
entropy_gap    = 8.0 - global_entropy             # closer to 0 → more random

# Concatenate all derived features
derived = np.hstack([
    freq_mean, freq_std, freq_skew, freq_kurt,
    freq_max, freq_min, freq_range,
    chi2,
    zero_freq_count,
    q1, q2, q3, iqr,
    low_sum, high_sum, mid_sum,
    printable_sum,
    entropy_gap,
])

# Final feature matrix
X_all = np.hstack([base_X, derived])
print(f"  Base features     : {base_X.shape[1]}")
print(f"  Derived features  : {derived.shape[1]}")
print(f"  Total features    : {X_all.shape[1]}")

# ============================================================
# 3. ENCODE & SPLIT
# ============================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

print(f"\nClasses ({num_classes}):", label_encoder.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}\n")

# ============================================================
# 4. RANDOM FOREST (balanced, more trees)
# ============================================================
print("Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"  Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

# ============================================================
# 5. XGBOOST (more estimators, better regularisation)
# ============================================================
print("Training XGBoost ...")
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)
xgb_pred = xgb.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)
print(f"  XGBoost Accuracy: {xgb_acc:.4f}")
print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

# ============================================================
# 6. DEEP MLP (BatchNorm + Dropout + LR scheduler)
# ============================================================
print("Training Deep Neural Network ...")

X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
X_te_t  = torch.tensor(X_test,  dtype=torch.float32)
y_tr_t  = torch.tensor(y_train, dtype=torch.long)
y_te_t  = torch.tensor(y_test,  dtype=torch.long)

# Weighted sampler for any residual imbalance
class_counts = np.bincount(y_train)
weights      = 1.0 / class_counts[y_train]
sampler      = WeightedRandomSampler(weights, len(weights))

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=256, sampler=sampler)


class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


mlp = DeepMLP(X_tr_t.shape[1], num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

epochs = 60

for epoch in range(epochs):
    mlp.train()
    total_loss = 0
    for batch_x, batch_y in train_dl:
        optimizer.zero_grad()
        loss = criterion(mlp(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        mlp.eval()
        with torch.no_grad():
            val_pred = mlp(X_te_t).argmax(dim=1)
        val_acc = accuracy_score(y_test, val_pred.numpy())
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  Loss: {total_loss:.3f}  Val Acc: {val_acc:.4f}")
        mlp.train()

mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(X_te_t).argmax(dim=1).numpy()
mlp_acc = accuracy_score(y_test, mlp_pred)
print(f"\n  Deep MLP Accuracy: {mlp_acc:.4f}")
print(classification_report(y_test, mlp_pred, target_names=label_encoder.classes_))

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "="*50)
print("       FINAL ACCURACY COMPARISON")
print("="*50)
print(f"  Random Forest   : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(f"  XGBoost         : {xgb_acc:.4f}  ({xgb_acc*100:.2f}%)")
print(f"  Deep MLP        : {mlp_acc:.4f}  ({mlp_acc*100:.2f}%)")
print("="*50)

# Best model confusion matrix
accs = {"Random Forest": (rf_acc, rf_pred),
        "XGBoost":       (xgb_acc, xgb_pred),
        "Deep MLP":      (mlp_acc, mlp_pred)}
best_name, (best_acc, best_pred) = max(accs.items(), key=lambda x: x[1][0])
print(f"\nBest model: {best_name} ({best_acc*100:.2f}%)")

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f"Confusion Matrix — {best_name} ({best_acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_improved.png", dpi=150)
plt.show()
print("\nConfusion matrix saved to confusion_matrix_improved.png")
