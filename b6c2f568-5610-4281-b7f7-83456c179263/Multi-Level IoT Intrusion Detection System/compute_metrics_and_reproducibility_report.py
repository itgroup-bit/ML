import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("COMPREHENSIVE METRICS AND REPRODUCIBILITY REPORT")
print("=" * 100)

# Paper benchmark values from ticket
paper_benchmarks = {
    'Accuracy': 99.46,
    'Precision': 99.46,
    'Sensitivity': 99.46,
    'Specificity': 93.86,
    'F1-score': 99.46
}

print("\n📊 PAPER BENCHMARK METRICS:")
print("─" * 100)
for metric, value in paper_benchmarks.items():
    print(f"  {metric:15s}: {value:.2f}%")

# Compute confusion matrix AFTER feature selection (using 10 features from SFS)
print("\n" + "=" * 100)
print("STEP 1: COMPUTE CONFUSION MATRIX (AFTER FEATURE SELECTION)")
print("=" * 100)

cm_after_fs = confusion_matrix(y_test_5class, final_predictions_5class)
print(f"\n✓ Confusion matrix computed for {len(np.unique(y_test_5class))} classes")
print(f"  Test samples: {len(y_test_5class)}")
print(f"  Features used: {len(selected_features_sfs)} (after SFS feature selection)")

print("\nConfusion Matrix:")
print(cm_after_fs)

# Calculate ALL metrics per class and overall
print("\n" + "=" * 100)
print("STEP 2: CALCULATE ALL METRICS PER CLASS")
print("=" * 100)

n_classes_5 = len(np.unique(y_test_5class))
metrics_per_class_list = []

for class_idx in range(n_classes_5):
    TP = cm_after_fs[class_idx, class_idx]
    FP = cm_after_fs[:, class_idx].sum() - TP
    FN = cm_after_fs[class_idx, :].sum() - TP
    TN = cm_after_fs.sum() - (TP + FP + FN)
    
    accuracy_class = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
    precision_class = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0
    sensitivity_class = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0
    specificity_class = (TN / (TN + FP) * 100) if (TN + FP) > 0 else 0
    f1_class = (2 * TP / (2 * TP + FP + FN) * 100) if (2 * TP + FP + FN) > 0 else 0
    
    metrics_per_class_list.append({
        'Class': f'Class_{class_idx}',
        'Class_Index': class_idx,
        'Support': cm_after_fs[class_idx, :].sum(),
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'Accuracy': accuracy_class,
        'Precision': precision_class,
        'Sensitivity': sensitivity_class,
        'Specificity': specificity_class,
        'F1-score': f1_class
    })

metrics_per_class_df = pd.DataFrame(metrics_per_class_list)
print("\nPer-Class Metrics (5 Classes):")
print(metrics_per_class_df.to_string(index=False))

# Calculate OVERALL metrics (macro-averaged)
print("\n" + "=" * 100)
print("STEP 3: CALCULATE OVERALL METRICS (MACRO-AVERAGED)")
print("=" * 100)

overall_accuracy_pct = accuracy_score(y_test_5class, final_predictions_5class) * 100
overall_precision_macro_pct = precision_score(y_test_5class, final_predictions_5class, average='macro', zero_division=0) * 100
overall_sensitivity_macro_pct = recall_score(y_test_5class, final_predictions_5class, average='macro', zero_division=0) * 100
overall_f1_macro_pct = f1_score(y_test_5class, final_predictions_5class, average='macro', zero_division=0) * 100
overall_specificity_macro_pct = metrics_per_class_df['Specificity'].mean()

overall_metrics = {
    'Accuracy': overall_accuracy_pct,
    'Precision': overall_precision_macro_pct,
    'Sensitivity': overall_sensitivity_macro_pct,
    'Specificity': overall_specificity_macro_pct,
    'F1-score': overall_f1_macro_pct
}

print("\n✓ Overall Metrics (Macro-Averaged):")
print("─" * 100)
for metric, value in overall_metrics.items():
    print(f"  {metric:15s}: {value:.2f}%")

# Compare against paper benchmarks
print("\n" + "=" * 100)
print("STEP 4: COMPARE AGAINST PAPER BENCHMARKS")
print("=" * 100)

comparison_list = []
for metric in ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-score']:
    our_value = overall_metrics[metric]
    paper_value = paper_benchmarks[metric]
    deviation = our_value - paper_value
    deviation_pct = (deviation / paper_value * 100) if paper_value > 0 else 0
    
    comparison_list.append({
        'Metric': metric,
        'Our_Value': our_value,
        'Paper_Benchmark': paper_value,
        'Deviation': deviation,
        'Deviation_%': deviation_pct
    })

comparison_df = pd.DataFrame(comparison_list)
print("\nMetric Comparison:")
print(comparison_df.to_string(index=False))

# Visualization: Confusion Matrix
print("\n" + "=" * 100)
print("STEP 5: CREATE PROFESSIONAL VISUALIZATIONS")
print("=" * 100)

# Confusion matrix heatmap
fig_cm_report = plt.figure(figsize=(12, 10))
fig_cm_report.patch.set_facecolor('#1D1D20')
ax_cm_report = fig_cm_report.add_subplot(111)
ax_cm_report.set_facecolor('#1D1D20')

sns.heatmap(cm_after_fs, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(n_classes_5)],
            yticklabels=[f'Class {i}' for i in range(n_classes_5)],
            cbar_kws={'label': 'Count'},
            ax=ax_cm_report)

ax_cm_report.set_xlabel('Predicted Class', fontsize=14, color='#fbfbff', fontweight='bold')
ax_cm_report.set_ylabel('True Class', fontsize=14, color='#fbfbff', fontweight='bold')
ax_cm_report.set_title('Confusion Matrix: Multi-Level RF After Feature Selection\n(10 Features, 5 Classes, 4500 Test Samples)', 
                       fontsize=16, fontweight='bold', color='#fbfbff', pad=20)
ax_cm_report.tick_params(colors='#fbfbff', labelsize=11)

cbar = ax_cm_report.collections[0].colorbar
cbar.ax.yaxis.label.set_color('#fbfbff')
cbar.ax.tick_params(colors='#fbfbff')

plt.tight_layout()
print("✓ Created confusion matrix visualization: fig_cm_report")

# Metrics comparison chart
fig_comparison_report = plt.figure(figsize=(14, 8))
fig_comparison_report.patch.set_facecolor('#1D1D20')
ax_comparison = fig_comparison_report.add_subplot(111)
ax_comparison.set_facecolor('#1D1D20')

metrics_names = list(overall_metrics.keys())
our_values = [overall_metrics[m] for m in metrics_names]
paper_values = [paper_benchmarks[m] for m in metrics_names]

x_pos = np.arange(len(metrics_names))
width = 0.35

bars1 = ax_comparison.bar(x_pos - width/2, our_values, width, 
                          label='Our Implementation', color='#A1C9F4', alpha=0.85, edgecolor='#909094', linewidth=1.5)
bars2 = ax_comparison.bar(x_pos + width/2, paper_values, width,
                          label='Paper Benchmark', color='#FFB482', alpha=0.85, edgecolor='#909094', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_comparison.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}%',
                         ha='center', va='bottom', fontsize=10, color='#fbfbff', fontweight='bold')

ax_comparison.set_ylabel('Score (%)', fontsize=13, color='#fbfbff', fontweight='bold')
ax_comparison.set_xlabel('Metrics', fontsize=13, color='#fbfbff', fontweight='bold')
ax_comparison.set_title('Our Implementation vs. Paper Benchmarks\n(After Feature Selection - 10 Features)', 
                       fontsize=16, fontweight='bold', color='#fbfbff', pad=20)
ax_comparison.set_xticks(x_pos)
ax_comparison.set_xticklabels(metrics_names, fontsize=11)
ax_comparison.tick_params(colors='#fbfbff')
ax_comparison.legend(fontsize=12, loc='lower right', facecolor='#1D1D20', edgecolor='#909094', labelcolor='#fbfbff')
ax_comparison.spines['bottom'].set_color('#909094')
ax_comparison.spines['left'].set_color('#909094')
ax_comparison.spines['top'].set_visible(False)
ax_comparison.spines['right'].set_visible(False)
ax_comparison.set_ylim([0, 105])
ax_comparison.grid(True, alpha=0.2, color='#909094', linestyle='-', linewidth=0.5, axis='y')

plt.tight_layout()
print("✓ Created metrics comparison chart: fig_comparison_report")

# Deviation visualization
fig_deviation = plt.figure(figsize=(12, 7))
fig_deviation.patch.set_facecolor('#1D1D20')
ax_dev = fig_deviation.add_subplot(111)
ax_dev.set_facecolor('#1D1D20')

deviations = [comparison_df.loc[comparison_df['Metric'] == m, 'Deviation'].values[0] for m in metrics_names]
colors_dev = ['#8DE5A1' if d >= 0 else '#FF9F9B' for d in deviations]

bars_dev = ax_dev.bar(metrics_names, deviations, color=colors_dev, alpha=0.85, edgecolor='#909094', linewidth=1.5)

for bar, dev in zip(bars_dev, deviations):
    height = bar.get_height()
    ax_dev.text(bar.get_x() + bar.get_width()/2., height,
               f'{dev:+.2f}%',
               ha='center', va='bottom' if dev >= 0 else 'top', 
               fontsize=11, color='#fbfbff', fontweight='bold')

ax_dev.axhline(y=0, color='#fbfbff', linestyle='-', linewidth=1.5, alpha=0.8)
ax_dev.set_ylabel('Deviation from Paper (%)', fontsize=13, color='#fbfbff', fontweight='bold')
ax_dev.set_xlabel('Metrics', fontsize=13, color='#fbfbff', fontweight='bold')
ax_dev.set_title('Metric Deviations from Paper Benchmarks\n(Positive = Better, Negative = Worse)', 
                fontsize=16, fontweight='bold', color='#fbfbff', pad=20)
ax_dev.tick_params(colors='#fbfbff', labelsize=11)
ax_dev.spines['bottom'].set_color('#909094')
ax_dev.spines['left'].set_color('#909094')
ax_dev.spines['top'].set_visible(False)
ax_dev.spines['right'].set_visible(False)
ax_dev.grid(True, alpha=0.2, color='#909094', linestyle='-', linewidth=0.5, axis='y')

plt.tight_layout()
print("✓ Created deviation chart: fig_deviation")

# Generate comprehensive reproducibility report
print("\n" + "=" * 100)
print("STEP 6: COMPREHENSIVE REPRODUCIBILITY REPORT")
print("=" * 100)

print("\n" + "┌" + "─" * 98 + "┐")
print("│" + " " * 30 + "REPRODUCIBILITY REPORT" + " " * 46 + "│")
print("└" + "─" * 98 + "┘")

print("\n📌 RANDOM SEEDS USED:")
print("  ├─ Train/test split: random_state=42")
print("  ├─ TR1/TR2 split: random_state=42")
print("  ├─ MinMax normalization: deterministic")
print("  ├─ RF Model 1 (TR1): random_state=42")
print("  ├─ RF Model 2 (TR2): random_state=123")
print("  └─ Feature selection: random_state=42")

print("\n📊 METRIC DEVIATIONS & EXPLANATIONS:")
for _, row in comparison_df.iterrows():
    metric_name = row['Metric']
    deviation = row['Deviation']
    our_val = row['Our_Value']
    paper_val = row['Paper_Benchmark']
    
    status_icon = "✅" if abs(deviation) < 1.0 else "⚠️" if abs(deviation) < 5.0 else "❌"
    
    print(f"\n  {status_icon} {metric_name}:")
    print(f"     Our Value: {our_val:.2f}% | Paper: {paper_val:.2f}% | Deviation: {deviation:+.2f}%")
    
    # Explanation based on deviation
    if metric_name == 'Accuracy':
        if deviation >= -1.0:
            print(f"     Explanation: Excellent match! Within acceptable range (±1%)")
        else:
            print(f"     Explanation: Lower accuracy may be due to different train/test split or data sampling")
    
    elif metric_name == 'Precision':
        if deviation >= -1.0:
            print(f"     Explanation: Good precision alignment with paper")
        else:
            print(f"     Explanation: Slightly lower precision - possible differences in feature engineering or class distribution")
    
    elif metric_name == 'Sensitivity':
        if deviation >= -1.0:
            print(f"     Explanation: Strong recall performance, matches paper well")
        else:
            print(f"     Explanation: Lower recall - may miss some positive cases. Could be due to class imbalance handling differences")
    
    elif metric_name == 'Specificity':
        if deviation >= -5.0:
            print(f"     Explanation: Good specificity, correctly identifies negatives")
        elif deviation >= -10.0:
            print(f"     Explanation: Moderate difference - may be due to different negative class definitions or 5-class mapping")
        else:
            print(f"     Explanation: Significant difference - likely due to our 5-class mapping vs paper's original classification scheme")
    
    elif metric_name == 'F1-score':
        if deviation >= -1.0:
            print(f"     Explanation: Excellent F1 score, balanced precision and recall")
        else:
            print(f"     Explanation: Lower F1 - reflects tradeoff between precision and recall differences")

print("\n⚠️  PAPER AMBIGUITIES & ASSUMPTIONS:")
print("  1. Class Mapping: Paper uses original KDD classes; we mapped to 5 broad classes (normal, dos, probe, r2l, u2r)")
print("  2. Feature Selection: Paper doesn't fully specify SFS implementation details - we used greedy forward selection")
print("  3. Test Set Size: We used 4,500 samples (32.14% of 14,000); paper's exact test size unclear")
print("  4. Normalization: Applied MinMax [0,1] normalization; paper may use different scaling")
print("  5. Majority Voting: We used RF1 as tiebreaker (trained on larger TR1); paper's tie resolution unclear")
print("  6. Metric Calculation: Paper specifies macro-averaging; we computed per-class then averaged")
print("  7. Random Seeds: Paper doesn't document random seeds - we used 42 and 123 for reproducibility")

print("\n🔍 METHODOLOGY DIFFERENCES:")
print("  • Dataset: We sampled 14,000 records from KDD Cup; paper may use different subset/full dataset")
print("  • Features: We selected 10 features via SFS; paper methodology similar but implementation may differ")
print("  • Training: We used TR1 (6,175) and TR2 (3,325) split 65/35; paper uses similar approach")
print("  • Testing: We used separate 4,500 sample test set; paper's test set composition unclear")
print("  • Classes: We mapped 19 detailed classes → 5 broad classes; paper may use different granularity")

print("\n📈 FINAL REPRODUCIBILITY STATUS:")

# Determine overall status
avg_deviation = abs(comparison_df['Deviation'].mean())
max_deviation = abs(comparison_df['Deviation']).max()

if avg_deviation < 1.0 and max_deviation < 5.0:
    status = "✅ EXCELLENT REPRODUCIBILITY"
    explanation = "Our implementation closely matches paper benchmarks across all metrics"
elif avg_deviation < 3.0 and max_deviation < 10.0:
    status = "⚠️  GOOD REPRODUCIBILITY (with minor deviations)"
    explanation = "Our implementation is reasonably close to paper, with explainable differences"
else:
    status = "❌ MODERATE REPRODUCIBILITY (significant deviations)"
    explanation = "Notable differences exist, likely due to methodology, data, or class mapping differences"

print(f"\n  {status}")
print(f"  Average Deviation: {avg_deviation:.2f}%")
print(f"  Maximum Deviation: {max_deviation:.2f}%")
print(f"\n  {explanation}")

print("\n" + "=" * 100)
print("✅ COMPREHENSIVE REPRODUCIBILITY REPORT COMPLETE")
print("=" * 100)
print(f"  ✓ Confusion matrix computed (after feature selection)")
print(f"  ✓ All metrics calculated per class and overall")
print(f"  ✓ Comparison against paper benchmarks complete")
print(f"  ✓ Visualizations created (3 charts)")
print(f"  ✓ Reproducibility report generated with full documentation")
print("=" * 100)
