import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
kdd_df = pd.read_csv('kdd_train.csv')

# Display basic information
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"\nDataset Shape: {kdd_df.shape[0]:,} rows × {kdd_df.shape[1]} columns")
print(f"\nMemory Usage: {kdd_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 70)
print("COLUMN INFORMATION")
print("=" * 70)
print(f"\nData Types:")
print(kdd_df.dtypes.value_counts())

print("\n" + "=" * 70)
print("MISSING VALUES ANALYSIS")
print("=" * 70)
missing_stats = pd.DataFrame({
    'Column': kdd_df.columns,
    'Missing_Count': kdd_df.isnull().sum(),
    'Missing_Percentage': (kdd_df.isnull().sum() / len(kdd_df) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
if len(missing_stats) > 0:
    print(missing_stats.to_string(index=False))
else:
    print("✓ No missing values detected in the dataset")

print("\n" + "=" * 70)
print("SAMPLE DATA (First 5 Rows)")
print("=" * 70)
print(kdd_df.head())

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)
print(f"Total columns: {len(kdd_df.columns)}")
print("\nAll columns:")
for i, col in enumerate(kdd_df.columns, 1):
    print(f"{i:2d}. {col}")