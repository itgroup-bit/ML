
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Dataset configuration ────────────────────────────────────────────────────
NUM_CLIENTS   = 5          # total federated clients
CLIENT_ID     = 0          # partition we will train on locally
NUM_SAMPLES   = 2_000      # samples per client
NUM_FEATURES  = 78         # tabular NF-style features (like CICIDS-2017)
NUM_CLASSES   = 5          # BENIGN + 4 attack types
SEQ_LEN       = 8          # time-steps per LSTM window
OVERLAP       = 4          # sliding-window overlap

# ── Traffic class labels ─────────────────────────────────────────────────────
CLASS_NAMES = ["BENIGN", "DoS", "PortScan", "BruteForce", "Infiltration"]

# ── Simulate a single-client partition ──────────────────────────────────────
# Imbalanced class distribution (BENIGN dominates – realistic)
class_weights = [0.60, 0.15, 0.12, 0.08, 0.05]
labels_raw    = np.random.choice(NUM_CLASSES, size=NUM_SAMPLES, p=class_weights)

# Feature matrix: random floats with class-specific signal
features_raw = np.random.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
for c in range(NUM_CLASSES):
    mask = labels_raw == c
    features_raw[mask] += c * 0.3   # class-dependent shift

# ── Normalize features (standard scaling per feature) ───────────────────────
feat_mean = features_raw.mean(axis=0, keepdims=True)
feat_std  = features_raw.std(axis=0, keepdims=True) + 1e-8
features_norm = (features_raw - feat_mean) / feat_std

# ── Build sliding-window sequences  ─────────────────────────────────────────
step  = SEQ_LEN - OVERLAP
seqs, seq_labels = [], []

for start in range(0, NUM_SAMPLES - SEQ_LEN + 1, step):
    seqs.append(features_norm[start : start + SEQ_LEN])
    # label = majority class in the window
    window_lbl = labels_raw[start : start + SEQ_LEN]
    seq_labels.append(np.bincount(window_lbl, minlength=NUM_CLASSES).argmax())

client_X = np.stack(seqs).astype(np.float32)   # (N_seq, SEQ_LEN, NUM_FEATURES)
client_y = np.array(seq_labels, dtype=np.int64) # (N_seq,)

# ── Quick sanity summary ─────────────────────────────────────────────────────
print(f"Client {CLIENT_ID} partition ready")
print(f"  Sequences  : {client_X.shape}  →  (samples, seq_len, features)")
print(f"  Labels     : {client_y.shape}")
print(f"  Class dist : { {CLASS_NAMES[i]: int((client_y == i).sum()) for i in range(NUM_CLASSES)} }")
