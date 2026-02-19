import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print("=" * 70)
print("DETAILED DATASET STATISTICS REPORT")
print("=" * 70)

# Class distribution
print("\n" + "=" * 70)
print("TARGET CLASS DISTRIBUTION")
print("=" * 70)
class_counts = y.value_counts()
class_percentages = (y.value_counts(normalize=True) * 100).round(2)
class_dist_df = pd.DataFrame({
    'Attack Type': class_counts.index,
    'Count': class_counts.values,
    'Percentage': class_percentages.values
}).sort_values('Count', ascending=False)

print(class_dist_df.to_string(index=False))
print(f"\nTotal unique attack types: {len(class_counts)}")

# Training and test set distributions
print("\n" + "=" * 70)
print("CLASS DISTRIBUTION IN SPLITS")
print("=" * 70)
train_dist = y_train.value_counts().sort_index()
test_dist = y_test.value_counts().sort_index()

# Get all unique classes from both sets
all_classes = sorted(set(train_dist.index) | set(test_dist.index))

# Create aligned lists
train_counts = [train_dist.get(cls, 0) for cls in all_classes]
test_counts = [test_dist.get(cls, 0) for cls in all_classes]
train_pcts = [(cnt / len(y_train) * 100) for cnt in train_counts]
test_pcts = [(cnt / len(y_test) * 100) for cnt in test_counts]

split_comparison = pd.DataFrame({
    'Attack Type': all_classes,
    'Train Count': train_counts,
    'Train %': [round(p, 2) for p in train_pcts],
    'Test Count': test_counts,
    'Test %': [round(p, 2) for p in test_pcts]
})
print(split_comparison.to_string(index=False))

# Numerical features statistics
print("\n" + "=" * 70)
print("NUMERICAL FEATURES STATISTICS (Original Data)")
print("=" * 70)
print(f"\nTraining Set Statistics:")
print(X_train[numerical_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].head(15))

print("\n" + "=" * 70)
print("NUMERICAL FEATURES STATISTICS (After Normalization)")
print("=" * 70)
print(f"\nTraining Set Statistics (Scaled):")
print(X_train_scaled[numerical_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].head(15))

# Categorical features statistics
print("\n" + "=" * 70)
print("CATEGORICAL FEATURES STATISTICS (After Encoding)")
print("=" * 70)
for cat_col in categorical_features:
    unique_vals = preprocessed_df[cat_col].nunique()
    print(f"\n{cat_col}:")
    print(f"  - Unique encoded values: {unique_vals}")
    print(f"  - Value range: [{preprocessed_df[cat_col].min()} - {preprocessed_df[cat_col].max()}]")
    print(f"  - Top 5 most common:")
    top_values = preprocessed_df[cat_col].value_counts().head(5)
    for val, count in top_values.items():
        print(f"      {val}: {count:,} ({count/len(preprocessed_df)*100:.1f}%)")

print("\n" + "=" * 70)
print("FEATURE CORRELATION WITH TARGET (Top 10)")
print("=" * 70)
# Encode target for correlation analysis
target_encoded = LabelEncoder().fit_transform(y)
correlation_df = pd.DataFrame(X, columns=X.columns)
correlation_df['target'] = target_encoded
correlation_matrix = correlation_df.corr()
correlations_with_target = correlation_matrix['target'].drop('target')
# Sort by absolute correlation descending and keep feature names as index
correlations = correlations_with_target.abs().sort_values(ascending=False)
print("\nTop 10 features most correlated with target:")
for i, (feat, corr_val) in enumerate(correlations.head(10).items(), 1):
    print(f"{i:2d}. {feat:35s} : {corr_val:.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ Dataset successfully preprocessed and split")
print(f"   - Original dataset: {len(kdd_df):,} samples, {len(kdd_df.columns)} features")
print(f"   - Features after preprocessing: {X_train_scaled.shape[1]}")
print(f"   - Training samples: {X_train_scaled.shape[0]:,} ({X_train_scaled.shape[0]/len(X)*100:.1f}%)")
print(f"   - Test samples: {X_test_scaled.shape[0]:,} ({X_test_scaled.shape[0]/len(X)*100:.1f}%)")
print(f"   - Target classes: {len(class_counts)}")
print(f"   - Missing values: 0")
print(f"   - Categorical features encoded: {len(categorical_features)}")
print(f"   - Numerical features normalized: {len(numerical_cols)}")