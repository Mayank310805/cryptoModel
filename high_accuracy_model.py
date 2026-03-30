# ============================================================
# high_accuracy_model.py
# Crypto Algorithm Classification - Proper Byte-Level Features
# Root-cause: ciphertexts in dataset.csv are base64-encoded strings.
# Fix: decode to real bytes first, then extract byte-level features.
# ============================================================
import os
import math
import base64
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# FEATURE EXTRACTOR (on real cipher bytes)
# ============================================================

def shannon_entropy(data: np.ndarray) -> float:
    if len(data) == 0:
        return 0.0
    counts = np.bincount(data, minlength=256)
    probs = counts / len(data)
    return float(-np.sum([p * math.log2(p) for p in probs if p > 0]))


def autocorrelation_at_lag(data: np.ndarray, lag: int) -> float:
    """Byte-level autocorrelation at a given lag."""
    if len(data) <= lag:
        return 0.0
    a = data[:-lag].astype(float)
    b = data[lag:].astype(float)
    mu = a.mean()
    diff_a = a - mu
    diff_b = b - mu
    denom = np.sqrt((diff_a ** 2).sum() * (diff_b ** 2).sum())
    return float(np.dot(diff_a, diff_b) / denom) if denom > 0 else 0.0


def block_periodicity(data: np.ndarray, block_size: int) -> float:
    """
    Mean absolute difference between non-overlapping consecutive blocks.
    Lower = more periodic (ECB block cipher pattern).
    """
    n_blocks = len(data) // block_size
    if n_blocks < 2:
        return 1.0
    blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size).astype(float)
    diffs = np.diff(blocks, axis=0)
    return float(np.abs(diffs).mean())


def byte_runs(data: np.ndarray):
    """Run-length stats: mean and std of run lengths."""
    if len(data) == 0:
        return 0.0, 0.0
    runs = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            runs.append(count)
            count = 1
    runs.append(count)
    runs = np.array(runs)
    return float(runs.mean()), float(runs.std())


def extract_features_from_bytes(raw_bytes: np.ndarray) -> list:
    """Extract 295-dim discriminative feature vector from raw cipher bytes."""
    n = len(raw_bytes)

    # ── 1. Byte frequency histogram (256)
    freq = np.bincount(raw_bytes, minlength=256) / (n + 1e-9)

    # ── 2. Global statistics (7)
    entropy = shannon_entropy(raw_bytes)
    zero_ratio = float(np.sum(raw_bytes == 0) / n)
    mean_byte = float(raw_bytes.mean())
    std_byte = float(raw_bytes.std())
    chi2 = float(((freq - 1/256) ** 2 / (1/256 + 1e-9)).sum())
    printable = float(freq[0x20:0x7F].sum())
    high_byte = float(freq[0x80:].sum())

    # ── 3. Output length & block-structure features (8)
    file_size = float(n)
    mod8  = float(n % 8)
    mod16 = float(n % 16)
    mod32 = float(n % 32)
    periodicity_8  = block_periodicity(raw_bytes, 8)
    periodicity_16 = block_periodicity(raw_bytes, 16)
    periodicity_32 = block_periodicity(raw_bytes, 32)
    periodicity_64 = block_periodicity(raw_bytes, 64)

    # ── 4. Autocorrelation at key lags (8)
    ac_lags = [1, 2, 4, 8, 16, 32, 64, 128]
    autocorrs = [autocorrelation_at_lag(raw_bytes, lag) for lag in ac_lags]

    # ── 5. Run-length stats (2)
    run_mean, run_std = byte_runs(raw_bytes)

    # ── 6. Quartile statistics of freq (4)
    q1, q2, q3 = np.percentile(freq, [25, 50, 75])
    iqr = q3 - q1

    # ── 7. Low/mid/high byte sums (3)
    low_sum  = float(freq[:64].sum())
    mid_sum  = float(freq[64:192].sum())
    hi_sum   = float(freq[192:].sum())

    # ── 8. Byte transition: same-byte sequential repeat ratio (1)
    repeat_ratio = float(np.sum(np.diff(raw_bytes.astype(int)) == 0) / max(n-1, 1))

    features = (
        list(freq)                                  # 256
        + [entropy, zero_ratio, mean_byte, std_byte,
           chi2, printable, high_byte]              # 7
        + [file_size, mod8, mod16, mod32,
           periodicity_8, periodicity_16,
           periodicity_32, periodicity_64]          # 8
        + autocorrs                                 # 8
        + [run_mean, run_std]                       # 2
        + [float(q1), float(q2), float(q3), float(iqr)]  # 4
        + [low_sum, mid_sum, hi_sum]               # 3
        + [repeat_ratio]                            # 1
    )
    return features   # Total: 289


def try_decode(ciphertext_str: str) -> np.ndarray:
    """Try base64 decode; fall back to raw UTF-8 bytes."""
    s = str(ciphertext_str).strip()
    try:
        raw = base64.b64decode(s + "==")   # pad just in case
        return np.frombuffer(raw, dtype=np.uint8)
    except Exception:
        return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)


# ============================================================
# LOAD & EXTRACT
# ============================================================
print("Loading dataset.csv ...")
df = pd.read_csv("dataset.csv")
print(f"  Rows: {len(df)}  |  Algorithms: {sorted(df['Algorithm'].unique())}")

print("\nExtracting byte-level features (decoding base64 ciphertext) ...")
feature_rows = []
labels = []

for _, row in df.iterrows():
    raw = try_decode(row["Ciphertext"])
    if len(raw) == 0:
        continue
    feature_rows.append(extract_features_from_bytes(raw))
    labels.append(row["Algorithm"])

X_all = np.array(feature_rows, dtype=np.float32)
y_raw = np.array(labels)

print(f"  Feature matrix: {X_all.shape}")
print(f"\nClass distribution:")
unique, counts = np.unique(y_raw, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u:12s}: {c}")


# ============================================================
# ENCODE, SCALE, SPLIT
# ============================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}\n")


# ============================================================
# RANDOM FOREST
# ============================================================
print("=" * 55)
print("Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"  Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred,
                            target_names=label_encoder.classes_))


# ============================================================
# XGBOOST
# ============================================================
print("=" * 55)
print("Training XGBoost ...")
xgb = XGBClassifier(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=9,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=2,
    gamma=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
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
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"  XGBoost Accuracy: {xgb_acc:.4f}")
print(classification_report(y_test, xgb_pred,
                            target_names=label_encoder.classes_))


# ============================================================
# DEEP MLP
# ============================================================
print("=" * 55)
print("Training Deep MLP ...")

X_tr_t = torch.tensor(X_train, dtype=torch.float32)
X_te_t = torch.tensor(X_test,  dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.long)
y_te_t = torch.tensor(y_test,  dtype=torch.long)

class_counts = np.bincount(y_train)
sample_weights = torch.tensor(1.0 / class_counts[y_train], dtype=torch.float)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=512, sampler=sampler)


class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 256),        nn.BatchNorm1d(256),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128),        nn.BatchNorm1d(128),  nn.GELU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)


mlp = DeepMLP(X_tr_t.shape[1], num_classes)
optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
criterion = nn.CrossEntropyLoss()

for epoch in range(80):
    mlp.train()
    total_loss = 0
    for bx, by in train_dl:
        optimizer.zero_grad()
        loss = criterion(mlp(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 20 == 0:
        mlp.eval()
        with torch.no_grad():
            val_acc = accuracy_score(y_test, mlp(X_te_t).argmax(1).numpy())
        print(f"  Epoch [{epoch+1:3d}/80]  Loss: {total_loss:.3f}  Val Acc: {val_acc:.4f}")
        mlp.train()

mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(X_te_t).argmax(1).numpy()
mlp_acc = accuracy_score(y_test, mlp_pred)
print(f"\n  Deep MLP Accuracy: {mlp_acc:.4f}")
print(classification_report(y_test, mlp_pred,
                            target_names=label_encoder.classes_))


# ============================================================
# SUMMARY + CONFUSION MATRIX
# ============================================================
print("\n" + "=" * 55)
print("          FINAL ACCURACY COMPARISON")
print("=" * 55)
print(f"  Random Forest : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(f"  XGBoost       : {xgb_acc:.4f}  ({xgb_acc*100:.2f}%)")
print(f"  Deep MLP      : {mlp_acc:.4f}  ({mlp_acc*100:.2f}%)")
print("=" * 55)

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
plt.savefig("confusion_matrix_highaccuracy.png", dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix_highaccuracy.png")
