
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 2. Preprocessing: normalization, label encoding, train/test split
# ─────────────────────────────────────────────

feat_cols = [c for c in raw_df.columns if c.startswith("feat_")]

# --- Min-Max normalization across the full dataset ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(raw_df[feat_cols].values)

# --- Integer label encoding ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(raw_df["label"].values)

print(f"✅ Preprocessing complete")
print(f"   Feature matrix : {X_scaled.shape}")
print(f"   Label classes  : {list(label_encoder.classes_)}")
print(f"   Encoded range  : [{y_encoded.min()}, {y_encoded.max()}]")

# --- 80/20 stratified train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"\n   Train samples  : {len(X_train):,}")
print(f"   Test samples   : {len(X_test):,}")

# --- Class label mapping (integer ↔ string) ---
label_map = {i: cls for i, cls in enumerate(label_encoder.classes_)}
n_classes = len(label_encoder.classes_)

# Verify train label distribution (proportional to global distribution)
_train_counts = pd.Series(y_train).value_counts().sort_index()
print(f"\nTrain label counts (encoded):")
for enc_lbl, cnt in _train_counts.items():
    print(f"  {enc_lbl:2d} → {label_map[enc_lbl]:<12s}: {cnt:5d}")
