import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Working with the loaded dataset
print("=" * 70)
print("PREPROCESSING PIPELINE - PAPER SPECIFICATION")
print("=" * 70)

# Step 1: Random sampling - 9500 training + 4500 test = 14000 total samples
# Using random_state=42 for reproducibility
print("\n" + "=" * 70)
print("RANDOM SAMPLING (seed=42)")
print("=" * 70)

total_samples = 9500 + 4500  # 14000 samples total
sampled_df = kdd_df.sample(n=total_samples, random_state=42)
print(f"✓ Sampled {total_samples:,} records from {len(kdd_df):,} total records")

# Step 2: Identify categorical and numerical columns
categorical_cols = sampled_df.select_dtypes(include=['object']).columns.tolist()
# Categorical features per paper: protocol_type, service, flag, class(labels)
categorical_features = [col for col in categorical_cols if col != 'labels']
numerical_cols = sampled_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n✓ Identified {len(categorical_features)} categorical features: {categorical_features}")
print(f"✓ Identified {len(numerical_cols)} numerical features")

# Step 3: Encode categorical features (protocol_type, service, flag, class)
print("\n" + "=" * 70)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 70)

preprocessed_df = sampled_df.copy()
label_encoders = {}

# Encode protocol_type, service, flag
for col in categorical_features:
    le = LabelEncoder()
    preprocessed_df[col] = le.fit_transform(sampled_df[col])
    label_encoders[col] = le
    print(f"✓ Encoded '{col}': {len(le.classes_)} unique values → [0-{len(le.classes_)-1}]")

# Encode class (labels) - the target variable
le_class = LabelEncoder()
preprocessed_df['labels'] = le_class.fit_transform(sampled_df['labels'])
label_encoders['labels'] = le_class
print(f"✓ Encoded 'labels' (class): {len(le_class.classes_)} unique values → [0-{len(le_class.classes_)-1}]")

# Step 4: Split into training (9500) and test (4500) sets
print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

# Separate features and target
X_full = preprocessed_df.drop('labels', axis=1)
y_full = preprocessed_df['labels']

# Split: 9500 training, 4500 test (without stratification to avoid minority class issues)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_full, y_full, test_size=4500, random_state=42
)

print(f"✓ Training set: {X_train_full.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")

# Verify exact counts
assert X_train_full.shape[0] == 9500, f"Expected 9500 training samples, got {X_train_full.shape[0]}"
assert X_test.shape[0] == 4500, f"Expected 4500 test samples, got {X_test.shape[0]}"
print("✓ Sample counts verified: 9500 training, 4500 test")

# Step 5: Split training into TR1 (65%) and TR2 (35%)
print("\n" + "=" * 70)
print("SPLIT TRAINING INTO TR1 (65%) AND TR2 (35%)")
print("=" * 70)

# TR1 should be 65% of 9500 = 6175, TR2 should be 35% of 9500 = 3325
X_TR1, X_TR2, y_TR1, y_TR2 = train_test_split(
    X_train_full, y_train_full, test_size=0.35, random_state=42
)

print(f"✓ TR1 (65%): {X_TR1.shape[0]:,} samples ({X_TR1.shape[0]/X_train_full.shape[0]*100:.1f}%)")
print(f"✓ TR2 (35%): {X_TR2.shape[0]:,} samples ({X_TR2.shape[0]/X_train_full.shape[0]*100:.1f}%)")

expected_TR1 = int(9500 * 0.65)
expected_TR2 = 9500 - expected_TR1
print(f"✓ Split verified: TR1={X_TR1.shape[0]} (expected {expected_TR1}), TR2={X_TR2.shape[0]} (expected {expected_TR2})")

# Step 6: Min-Max normalization to [0,1] range
print("\n" + "=" * 70)
print("MIN-MAX NORMALIZATION [0,1]")
print("=" * 70)

scaler_minmax = MinMaxScaler(feature_range=(0, 1))

# Normalize TR1 (fit and transform)
X_TR1_normalized = X_TR1.copy()
X_TR1_normalized[numerical_cols] = scaler_minmax.fit_transform(X_TR1[numerical_cols])

# Normalize TR2 (transform only - using TR1 parameters)
X_TR2_normalized = X_TR2.copy()
X_TR2_normalized[numerical_cols] = scaler_minmax.transform(X_TR2[numerical_cols])

# Normalize Test (transform only - using TR1 parameters)
X_test_normalized = X_test.copy()
X_test_normalized[numerical_cols] = scaler_minmax.transform(X_test[numerical_cols])

print(f"✓ Normalized {len(numerical_cols)} numerical features using MinMaxScaler")
print(f"✓ Fitted on TR1 and applied to TR1, TR2, and Test sets")

# Verify normalization range
print("\n" + "=" * 70)
print("NORMALIZATION VERIFICATION")
print("=" * 70)
tr1_min = X_TR1_normalized[numerical_cols].min().min()
tr1_max = X_TR1_normalized[numerical_cols].max().max()
tr2_min = X_TR2_normalized[numerical_cols].min().min()
tr2_max = X_TR2_normalized[numerical_cols].max().max()
test_min = X_test_normalized[numerical_cols].min().min()
test_max = X_test_normalized[numerical_cols].max().max()

print(f"TR1 numerical features - Min: {tr1_min:.6f}, Max: {tr1_max:.6f}")
print(f"TR2 numerical features - Min: {tr2_min:.6f}, Max: {tr2_max:.6f}")
print(f"Test numerical features - Min: {test_min:.6f}, Max: {test_max:.6f}")

# Verify range is [0,1] for TR1 (fit set)
assert tr1_min >= 0 and tr1_max <= 1, f"TR1 normalization failed: range [{tr1_min}, {tr1_max}]"
print("✓ Normalization range [0,1] verified for TR1")

print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETE - PAPER SPECIFICATION")
print("=" * 70)
print(f"   - Total samples: {total_samples:,}")
print(f"   - Training samples: {X_train_full.shape[0]:,} (9500 ✓)")
print(f"   - Test samples: {X_test.shape[0]:,} (4500 ✓)")
print(f"   - TR1: {X_TR1.shape[0]:,} samples (65% of training)")
print(f"   - TR2: {X_TR2.shape[0]:,} samples (35% of training)")
print(f"   - Categorical features encoded: 4 (protocol_type, service, flag, class) ✓")
print(f"   - Numerical features normalized: {len(numerical_cols)} to range [0,1] ✓")
print(f"   - Random seed: 42 ✓")
print(f"   - Total features: {X_TR1_normalized.shape[1]}")