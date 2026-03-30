import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from PIL import Image

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))

from binary_features import extract_features
from preprocess import load_vocab, prepare_input

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Firmware Detector",
    page_icon="🔒",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 17px;
    background-color: #060d1a;
    color: #cfd8e3;
}

/* ── Main title ── */
.main-title {
    font-size: 44px;
    font-weight: 700;
    background: linear-gradient(135deg, #00e6e6 0%, #00c3ff 50%, #7b61ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
}

.subtitle {
    font-size: 17px;
    color: #8899aa;
    margin-bottom: 28px;
}

/* ── Section header ── */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #00c3ff;
    margin-top: 30px;
    margin-bottom: 12px;
}

/* ── Feature cards ── */
.feature-card {
    background: linear-gradient(145deg, #0c1220, #121a2b);
    border-radius: 14px;
    padding: 26px 22px;
    margin-bottom: 16px;
    border: 1px solid #1f2a40;
    min-height: 160px;
    transition: all 0.3s ease;
}
.feature-card:hover {
    border-color: #00c8ff;
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,200,255,0.15);
}
.feature-icon {
    font-size: 32px;
    margin-bottom: 10px;
}
.feature-title {
    font-size: 18px;
    font-weight: 700;
    color: #4dd6ff;
    margin-bottom: 8px;
}
.feature-desc {
    font-size: 15px;
    color: #8fa3b8;
    line-height: 1.6;
}

/* ── Algorithm cards ── */
.alg-card {
    background: linear-gradient(145deg, #0a1020, #101828);
    border: 1px solid #1a2540;
    border-radius: 14px;
    padding: 20px 18px;
    margin-bottom: 14px;
    transition: all 0.3s ease;
}
.alg-card:hover {
    border-color: #7b61ff;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(123,97,255,0.15);
}
.alg-title {
    font-size: 20px;
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 8px;
}
.alg-tag {
    display: inline-block;
    background: #1e2d47;
    color: #93c5fd;
    padding: 3px 10px;
    border-radius: 8px;
    margin-right: 6px;
    font-size: 12px;
    font-weight: 600;
}
.alg-desc {
    margin-top: 10px;
    font-size: 14px;
    color: #8fa3b8;
    line-height: 1.55;
}

/* ── Metric boxes ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0c1220, #121a2b);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1f2a40;
}

/* ── Divider ── */
hr { border-color: #1a2540; margin: 30px 0; }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #00c3ff, #7b61ff);
    color: #fff;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 16px;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model Definitions  (must match training scripts exactly)
# ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        return self.relu(out)


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
        return self.output(x)


class CryptoLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


# ─────────────────────────────────────────────────────────────
# Cached Resource Loaders
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_resnet_model():
    """Load CryptoResNet from crypto_resnet_model.pth.
    Dynamically reads input_dim from the saved weight shape so the app
    works regardless of which feature set was used during training.
    """
    model_path = os.path.join(current_dir, "crypto_resnet_model.pth")
    if not os.path.exists(model_path):
        return None, None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    scaler = checkpoint["scaler"]
    label_encoder = checkpoint["label_encoder"]
    num_classes = len(label_encoder.classes_)

    # Auto-detect input_dim from the saved weight matrix
    input_dim = checkpoint["model_state_dict"]["input_layer.weight"].shape[1]

    model = CryptoResNet(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, label_encoder


@st.cache_resource(show_spinner=False)
def load_lstm_model():
    """Load CryptoLSTM from crypto_lstm_model.pth, rebuilding vocab from dataset.csv."""
    model_path = os.path.join(current_dir, "crypto_lstm_model.pth")
    dataset_path = os.path.join(current_dir, "dataset.csv")

    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        return None, None, None


    # Rebuild vocab exactly as in crypto_model.py training
    char_to_idx = load_vocab(dataset_path)
    vocab_size = len(char_to_idx) + 1

    # Rebuild label list from dataset
    df = pd.read_csv(dataset_path)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df["Algorithm"].values)
    num_classes = len(le.classes_)

    model = CryptoLSTM(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_classes=num_classes
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, char_to_idx, le


def run_resnet(features_list, model, scaler, label_encoder):
    """Run ResNet inference. Truncates/pads feature vector to match the model's input_dim."""
    # Determine model's expected input_dim
    expected_dim = next(model.parameters()).shape[1] if hasattr(next(model.parameters()), 'shape') else len(features_list)
    # Safer: read from first linear layer
    expected_dim = model.input_layer.in_features

    feat = list(features_list)
    if len(feat) < expected_dim:
        feat += [0.0] * (expected_dim - len(feat))   # pad with zeros
    feat = feat[:expected_dim]                         # truncate if longer

    X = np.array([feat], dtype=np.float32)
    X_scaled = scaler.transform(X)
    tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = np.argmax(probs)
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    return pred_label, confidence


def run_lstm(plaintext, ciphertext, model, char_to_idx, label_encoder):
    """Run LSTM inference on plaintext+ciphertext strings. Returns (label, confidence%)."""
    encoded = prepare_input(plaintext, ciphertext, char_to_idx)
    tensor = torch.tensor(encoded, dtype=torch.long)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = np.argmax(probs)
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    return pred_label, confidence


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🔐 AI/ML Cryptographic Firmware Analyzer</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Detect cryptographic primitives in firmware binaries using machine learning and entropy analysis.</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# Hero Image
# ─────────────────────────────────────────────────────────────
image_path = os.path.join(current_dir, "home_img.png")
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="AI/ML-Based Identification of Cryptographic Primitives in Firmware", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Key Features
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Key Features</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Entropy Analysis</div>
        <div class="feature-desc">
            Detects encrypted or compressed binary regions by computing Shannon entropy
            across byte distributions in uploaded firmware.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">Deep Learning Classification</div>
        <div class="feature-desc">
            ResNet &amp; LSTM models trained on 259-dimensional binary features predict
            crypto algorithms like AES, DES, ChaCha20 with high accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔎</div>
        <div class="feature-title">Signature Detection</div>
        <div class="feature-desc">
            Scans for known cryptographic constants — AES S-box, Blowfish Pi-digits,
            ChaCha20 nonce strings, RSA DER headers, and more.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Supported Algorithms
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🧩 Supported Algorithms</div>', unsafe_allow_html=True)

row1 = st.columns(3)
with row1[0]:
    st.markdown("""
    <div class="alg-card">
        <div class="alg-title">AES</div>
        <span class="alg-tag">Symmetric</span>
        <span class="alg-tag">128 / 192 / 256-bit</span>
        <div class="alg-desc">Modern encryption standard for secure communication and storage.</div>
    </div>""", unsafe_allow_html=True)
with row1[1]:
    st.markdown("""
    <div class="alg-card">
        <div class="alg-title">DES</div>
        <span class="alg-tag">Legacy</span>
        <span class="alg-tag">56-bit</span>
        <div class="alg-desc">Older encryption standard, now deprecated. Still found in legacy systems.</div>
    </div>""", unsafe_allow_html=True)
with row1[2]:
    st.markdown("""
    <div class="alg-card">
        <div class="alg-title">3DES</div>
        <span class="alg-tag">Triple Encryption</span>
        <div class="alg-desc">Enhanced DES variant used historically in banking and payment systems.</div>
    </div>""", unsafe_allow_html=True)

row2 = st.columns(3)
with row2[0]:
    st.markdown("""
    <div class="alg-card">
        <div class="alg-title">Blowfish</div>
        <span class="alg-tag">Fast Cipher</span>
        <div class="alg-desc">Fast symmetric cipher with pi-digit key schedule, used in file encryption tools.</div>
    </div>""", unsafe_allow_html=True)
with row2[1]:
    st.markdown("""
    <div class="alg-card">
        <div class="alg-title">RC4</div>
        <span class="alg-tag">Stream Cipher</span>
        <span class="alg-tag">⚠ Broken</span>
        <div class="alg-desc">Previously used in SSL and WEP. Now considered cryptographically insecure.</div>
    </div>""", unsafe_allow_html=True)
with row2[2]:
    st.markdown("""
    <div class="alg-card">
        <div class="alg-title">ChaCha20</div>
        <span class="alg-tag">Modern Stream Cipher</span>
        <div class="alg-desc">High-performance cipher adopted in TLS 1.3, WireGuard VPN, and QUIC.</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Text-Based Algorithm Detection (LSTM)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🔑 Detect Algorithm from Plaintext / Ciphertext</div>', unsafe_allow_html=True)
st.markdown("""
Provide a plaintext and its corresponding ciphertext.  
The LSTM model will predict the encryption algorithm used.
""")

col_plain, col_cipher = st.columns(2)
with col_plain:
    plain_input = st.text_area("Plaintext", placeholder="Enter original plaintext…", height=120)
with col_cipher:
    cipher_input = st.text_area("Ciphertext", placeholder="Enter ciphertext (hex or base64)…", height=120)

if st.button("🔍 Predict Algorithm from Text"):
    if plain_input and cipher_input:
        with st.spinner("Loading LSTM model…"):
            lstm_model, char_to_idx, lstm_le = load_lstm_model()

        if lstm_model is None:
            st.error("⚠️ LSTM model or dataset.csv not found. Please ensure `crypto_lstm_model.pth` and `dataset.csv` are in the project root.")
        else:
            try:
                with st.spinner("Running inference…"):
                    pred_label, confidence = run_lstm(plain_input, cipher_input, lstm_model, char_to_idx, lstm_le)

                st.success(f"🤖 Predicted Algorithm: **{pred_label}**")
                st.progress(int(confidence), text=f"Confidence: {confidence:.2f}%")

                with st.expander("About this prediction"):
                    st.markdown(f"""
                    - **Model**: CryptoLSTM (char-level sequence classifier)  
                    - **Input length**: {len(plain_input + cipher_input)} characters  
                    - **Predicted class**: `{pred_label}`  
                    - **Confidence**: `{confidence:.2f}%`
                    """)
            except Exception as e:
                st.error(f"Inference error: {e}")
    else:
        st.warning("⚠️ Please provide both plaintext and ciphertext.")

# ─────────────────────────────────────────────────────────────
# Binary Firmware Analysis (ResNet)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📂 Analyze Firmware Binary</div>', unsafe_allow_html=True)
st.markdown("""
Upload a firmware binary file. The ResNet model analyzes byte-frequency distributions
and entropy patterns to detect and classify cryptographic content.
""")

uploaded_file = st.file_uploader(
    "Upload firmware binary",
    type=["bin", "exe", "hex", "dat", "elf", "img"]
)

if uploaded_file is not None:
    temp_dir = os.path.join(current_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File uploaded: `{uploaded_file.name}` ({uploaded_file.size:,} bytes)")

    try:
        with st.spinner("Loading ResNet model…"):
            resnet_model, scaler, resnet_le = load_resnet_model()

        if resnet_model is None:
            st.error("⚠️ `crypto_resnet_model.pth` not found. Please train the ResNet model first.")
        else:
            with st.spinner("Extracting binary features and analyzing firmware…"):
                features, crypto_flag, signature_algo = extract_features(file_path)

            st.markdown("---")
            st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)

            # Metrics
            byte_data = np.array(features[:256])
            entropy_val = features[256]
            zero_ratio = features[257]
            file_size = int(features[258])
            avg_byte = float(np.average(np.arange(256), weights=byte_data + 1e-9))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📁 File Size", f"{file_size:,} bytes")
            m2.metric("📊 Shannon Entropy", f"{entropy_val:.4f}")
            m3.metric("⚖️ Avg Byte Value", f"{avg_byte:.1f}")
            m4.metric("⬜ Zero-Byte Ratio", f"{zero_ratio:.2%}")

            # Entropy indicator
            if entropy_val >= 7.0:
                st.warning(f"⚠️ High entropy detected ({entropy_val:.4f} ≥ 7.0) — likely encrypted/compressed content.")
            elif entropy_val >= 5.0:
                st.info(f"ℹ️ Moderate entropy ({entropy_val:.4f}) — possible partial crypto content.")
            else:
                st.success(f"✅ Low entropy ({entropy_val:.4f}) — file appears unencrypted.")

            if crypto_flag:
                st.error("🔐 Cryptographic Content Detected")

                try:
                    pred_label, confidence = run_resnet(features, resnet_model, scaler, resnet_le)
                    st.markdown(f"### 🤖 Predicted Algorithm: **{pred_label}**")
                    st.progress(int(confidence), text=f"Model Confidence: {confidence:.2f}%")
                except Exception as e:
                    st.error(f"ResNet inference error: {e}")

                if signature_algo:
                    st.info(f"🔎 Signature Match: **{signature_algo}** constants found in binary.")
                else:
                    st.caption("No specific crypto constant signatures matched.")

            else:
                st.success("✅ No cryptographic patterns detected in this binary.")

            # Feature detail expander
            with st.expander("📋 Full Feature Vector Details"):
                st.markdown("**Byte Frequency Distribution (top 10 most frequent bytes)**")
                freq_arr = np.array(features[:256])
                top10_idx = np.argsort(freq_arr)[::-1][:10]
                freq_df = pd.DataFrame({
                    "Byte (hex)": [f"0x{i:02X}" for i in top10_idx],
                    "Byte (dec)": top10_idx,
                    "Frequency": [f"{freq_arr[i]:.4f}" for i in top10_idx]
                })
                st.dataframe(freq_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Analysis error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#445566; font-size:13px; padding: 12px 0;">
    AI/ML Cryptographic Firmware Analyzer &nbsp;·&nbsp;
    Built with PyTorch &amp; Streamlit &nbsp;·&nbsp;
    ResNet + LSTM Dual-Model Architecture
</div>
""", unsafe_allow_html=True)
