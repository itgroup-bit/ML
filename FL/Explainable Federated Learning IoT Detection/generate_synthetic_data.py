
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. Synthetic IoT traffic feature definitions
#    Inspired by TON_IoT / N-BaIoT / Bot-IoT
# ─────────────────────────────────────────────

ATTACK_CLASSES = [
    "Benign",           # normal traffic
    "DDoS",             # distributed denial-of-service
    "DoS",              # denial-of-service
    "Mirai",            # Mirai botnet (N-BaIoT style)
    "Scan",             # network scanning
    "Injection",        # SQL / command injection (TON_IoT)
    "Ransomware",       # ransomware (TON_IoT)
    "MitM",             # man-in-the-middle (Bot-IoT)
    "Theft",            # data exfiltration (Bot-IoT)
]

# Sample counts per class (imbalanced, realistic distribution)
CLASS_SIZES = {
    "Benign":      8000,
    "DDoS":        4000,
    "DoS":         3500,
    "Mirai":       3000,
    "Scan":        2000,
    "Injection":   1500,
    "Ransomware":  1000,
    "MitM":         800,
    "Theft":        700,
}

TOTAL_SAMPLES = sum(CLASS_SIZES.values())
N_FEATURES = 40  # tabular feature count

def make_class_features(label: str, n: int) -> np.ndarray:
    """
    Generate n synthetic tabular traffic feature vectors for a given class.
    Feature semantics mirror TON_IoT / N-BaIoT / Bot-IoT flow statistics:
    packet lengths, inter-arrival times, byte counts, flag ratios, etc.
    Each class has distinct mean/std offsets to be learnable.
    """
    rng = np.random.default_rng(seed=hash(label) % (2**31))
    offsets = {
        "Benign":     (0.5,  0.10),
        "DDoS":       (0.9,  0.05),  # high packet rate, low variance
        "DoS":        (0.85, 0.07),
        "Mirai":      (0.75, 0.12),
        "Scan":       (0.3,  0.15),  # low byte counts
        "Injection":  (0.6,  0.20),
        "Ransomware": (0.7,  0.18),
        "MitM":       (0.55, 0.13),
        "Theft":      (0.65, 0.16),
    }
    mu, sigma = offsets[label]
    X = rng.normal(loc=mu, scale=sigma, size=(n, N_FEATURES)).clip(0, 1)
    # Add class-specific marker features (first 5 dims)
    class_idx = ATTACK_CLASSES.index(label)
    X[:, class_idx % N_FEATURES] += rng.uniform(0.1, 0.3, size=n)
    return X

records = []
for cls, n in CLASS_SIZES.items():
    X = make_class_features(cls, n)
    labels = np.full(n, cls)
    df_cls = pd.DataFrame(X, columns=[f"feat_{i:02d}" for i in range(N_FEATURES)])
    df_cls["label"] = labels
    records.append(df_cls)

raw_df = pd.concat(records, ignore_index=True)

# Shuffle the dataset
raw_df = raw_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Synthetic IoT dataset generated")
print(f"   Total samples : {len(raw_df):,}")
print(f"   Features      : {N_FEATURES}")
print(f"   Classes       : {len(ATTACK_CLASSES)}")
print(f"\nClass distribution:")
print(raw_df["label"].value_counts().to_string())
