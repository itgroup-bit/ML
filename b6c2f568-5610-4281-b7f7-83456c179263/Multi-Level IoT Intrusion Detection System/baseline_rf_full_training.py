import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("BASELINE RANDOM FOREST MODEL - FULL TRAINING DATA (TR1 + TR2)")
print("=" * 80)

# Combine TR1 and TR2 for full training data
print("\n✓ Combining TR1 and TR2 into full training set...")
X_train_combined = pd.concat([X_TR1_normalized, X_TR2_normalized], axis=0)
y_train_combined = pd.concat([y_TR1, y_TR2], axis=0)

print(f"  - TR1: {X_TR1_normalized.shape[0]:,} samples")
print(f"  - TR2: {X_TR2_normalized.shape[0]:,} samples")
print(f"  - Combined Training: {X_train_combined.shape[0]:,} samples")
print(f"  - Test Set: {X_test_normalized.shape[0]:,} samples")
print(f"  - Total Features: {X_train_combined.shape[1]}")

# Get class labels for interpretation
class_names = label_encoders['labels'].classes_
n_classes = len(class_names)
print(f"\n✓ {n_classes}-Class Classification:")
for i, class_name in enumerate(class_names):
    print(f"  - Class {i}: {class_name}")

# Train baseline Random Forest WITHOUT feature selection
print("\n" + "=" * 80)
print("TRAINING BASELINE RANDOM FOREST (All 41 Features)")
print("=" * 80)
print("Parameters: n_estimators=100, random_state=42, default hyperparameters")

baseline_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

baseline_rf.fit(X_train_combined, y_train_combined)
print(f"✓ Model trained on {X_train_combined.shape[0]:,} samples with {X_train_combined.shape[1]} features")

# Predict on test set
print("\n" + "=" * 80)
print("PREDICTION ON TEST SET")
print("=" * 80)

y_pred_baseline = baseline_rf.predict(X_test_normalized)
print(f"✓ Predictions generated for {len(y_pred_baseline):,} test samples")

# Check unique classes in test and predictions
unique_test = np.unique(y_test)
unique_pred = np.unique(y_pred_baseline)
print(f"\n✓ Unique classes in test set: {len(unique_test)}")
print(f"✓ Unique classes in predictions: {len(unique_pred)}")

# Compute confusion matrix
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm_baseline = confusion_matrix(y_test, y_pred_baseline)
print(f"\nConfusion Matrix shape: {cm_baseline.shape}")
print("Confusion Matrix (rows=actual, cols=predicted):")
print(cm_baseline)

# Create a formatted confusion matrix with class names - use only classes present in test set
classes_present_in_test = sorted(unique_test)
class_names_present = [class_names[i] for i in classes_present_in_test]

cm_df_baseline = pd.DataFrame(
    cm_baseline,
    index=[f"Actual_{class_names[i]}" for i in classes_present_in_test],
    columns=[f"Pred_{class_names[i]}" for i in classes_present_in_test]
)
print("\nConfusion Matrix with Class Labels:")
print(cm_df_baseline)

# Compute metrics per class (only for classes present in test set)
print("\n" + "=" * 80)
print("PER-CLASS METRICS")
print("=" * 80)

# Calculate True Positives, False Positives, False Negatives, True Negatives for each class
baseline_metrics_per_class = []

for idx, class_idx in enumerate(classes_present_in_test):
    class_name = class_names[class_idx]
    
    # True Positives: diagonal element
    TP = cm_baseline[idx, idx]
    
    # False Positives: sum of column idx minus TP
    FP = cm_baseline[:, idx].sum() - TP
    
    # False Negatives: sum of row idx minus TP
    FN = cm_baseline[idx, :].sum() - TP
    
    # True Negatives: sum of all elements minus TP, FP, FN
    TN = cm_baseline.sum() - TP - FP - FN
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall/Sensitivity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    baseline_metrics_per_class.append({
        'Class': class_name,
        'Class_Index': class_idx,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1-Score': f1
    })
    
    print(f"\n{class_name} (Class {class_idx}):")
    print(f"  TP: {TP:4d} | FP: {FP:4d} | FN: {FN:4d} | TN: {TN:4d}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}") 
    print(f"  F1-Score:    {f1:.4f}")

baseline_metrics_df = pd.DataFrame(baseline_metrics_per_class)

# Compute overall metrics
print("\n" + "=" * 80)
print("OVERALL METRICS")
print("=" * 80)

overall_accuracy = accuracy_score(y_test, y_pred_baseline)
overall_precision_macro = precision_score(y_test, y_pred_baseline, average='macro', zero_division=0)
overall_sensitivity_macro = recall_score(y_test, y_pred_baseline, average='macro', zero_division=0)
overall_f1_macro = f1_score(y_test, y_pred_baseline, average='macro', zero_division=0)

# Calculate macro average specificity
macro_specificity = baseline_metrics_df['Specificity'].mean()

print(f"\nOverall Accuracy:              {overall_accuracy:.4f}")
print(f"Overall Precision (macro):     {overall_precision_macro:.4f}")
print(f"Overall Sensitivity (macro):   {overall_sensitivity_macro:.4f}")
print(f"Overall Specificity (macro):   {macro_specificity:.4f}")
print(f"Overall F1-Score (macro):      {overall_f1_macro:.4f}")

# Create summary dataframe
baseline_summary = {
    'Metric': ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score'],
    'Overall (Macro)': [
        overall_accuracy,
        overall_precision_macro,
        overall_sensitivity_macro,
        macro_specificity,
        overall_f1_macro
    ]
}
baseline_summary_df = pd.DataFrame(baseline_summary)

print("\n" + "=" * 80)
print("✅ BASELINE RANDOM FOREST METRICS COMPUTED")
print("=" * 80)
print(f"   - Model: Random Forest (100 trees)")
print(f"   - Features: All 41 features (NO feature selection)")
print(f"   - Training: TR1 + TR2 = {X_train_combined.shape[0]:,} samples")
print(f"   - Test: {X_test_normalized.shape[0]:,} samples")
print(f"   - Classes in test set: {len(classes_present_in_test)} out of {n_classes} total")
print(f"   - Overall Accuracy: {overall_accuracy:.4f}")
print("=" * 80)
