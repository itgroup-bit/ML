import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

print("=" * 80)
print("LEVEL 2 WRAPPER METHOD: SEQUENTIAL FORWARD SELECTION (SFS)")
print("=" * 80)
print("\nImplementation Details:")
print("  - Algorithm: Sequential Forward Selection")
print("  - Starting Set: Y₀ = {} (empty set)")
print("  - Training Set: TR1 (6,175 samples)")
print("  - Validation Set: TR2 (3,325 samples)")
print("  - Classifier: Random Forest (100 trees)")
print("  - Target: ~10 features (per Table 8)")
print("  - Criterion: Maximize accuracy on TR2")

# Get all feature names from TR1
all_features = list(X_TR1_normalized.columns)
n_total_features = len(all_features)

print(f"\n✓ Total features available: {n_total_features}")
print(f"✓ TR1 shape: {X_TR1_normalized.shape}")
print(f"✓ TR2 shape: {X_TR2_normalized.shape}")

# Initialize SFS
sfs_selected_features = []  # Y_k - the selected feature subset
sfs_remaining_features = all_features.copy()  # Remaining candidate features
sfs_accuracy_history = []  # Track accuracy at each iteration
sfs_iteration_history = []  # Track which feature was added

# Track computation time
sfs_start_time = time.time()

print("\n" + "=" * 80)
print("STARTING SEQUENTIAL FORWARD SELECTION")
print("=" * 80)

# Target number of features (around 10 as per Table 8)
TARGET_N_FEATURES = 10

# Iterate until we have TARGET_N_FEATURES
for iteration in range(1, TARGET_N_FEATURES + 1):
    print(f"\n{'─' * 80}")
    print(f"ITERATION {iteration}: Evaluating {len(sfs_remaining_features)} candidate features")
    print(f"{'─' * 80}")
    
    best_accuracy_iter = 0
    best_feature_iter = None
    
    # Try adding each remaining feature
    for candidate_feature in sfs_remaining_features:
        # Create feature subset: current selected + candidate
        candidate_subset = sfs_selected_features + [candidate_feature]
        
        # Extract features from TR1 and TR2
        X_TR1_subset = X_TR1_normalized[candidate_subset]
        X_TR2_subset = X_TR2_normalized[candidate_subset]
        
        # Train Random Forest on TR1
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_TR1_subset, y_TR1)
        
        # Evaluate on TR2
        y_pred_TR2 = rf_model.predict(X_TR2_subset)
        accuracy_TR2 = accuracy_score(y_TR2, y_pred_TR2)
        
        # Check if this is the best feature to add
        if accuracy_TR2 > best_accuracy_iter:
            best_accuracy_iter = accuracy_TR2
            best_feature_iter = candidate_feature
    
    # Add the best feature to selected set
    sfs_selected_features.append(best_feature_iter)
    sfs_remaining_features.remove(best_feature_iter)
    sfs_accuracy_history.append(best_accuracy_iter)
    sfs_iteration_history.append({
        'iteration': iteration,
        'feature_added': best_feature_iter,
        'accuracy': best_accuracy_iter,
        'n_features': len(sfs_selected_features)
    })
    
    print(f"\n✓ Best feature to add: {best_feature_iter}")
    print(f"  - Accuracy on TR2: {best_accuracy_iter:.6f}")
    print(f"  - Current feature subset size: {len(sfs_selected_features)}")
    print(f"  - Features selected so far: {sfs_selected_features}")

# Compute total time
sfs_elapsed_time = time.time() - sfs_start_time

print("\n" + "=" * 80)
print("SEQUENTIAL FORWARD SELECTION COMPLETE")
print("=" * 80)
print(f"\n✓ Total iterations: {TARGET_N_FEATURES}")
print(f"✓ Total features selected: {len(sfs_selected_features)}")
print(f"✓ Computation time: {sfs_elapsed_time:.2f} seconds")
print(f"✓ Final accuracy on TR2: {sfs_accuracy_history[-1]:.6f}")

print("\n" + "─" * 80)
print("FINAL FEATURE SUBSET (Level 2 - SFS):")
print("─" * 80)
for i, feat in enumerate(sfs_selected_features, 1):
    print(f"  {i:2d}. {feat}")

# Create summary DataFrame
sfs_summary_df = pd.DataFrame(sfs_iteration_history)
print("\n" + "─" * 80)
print("SFS ITERATION SUMMARY:")
print("─" * 80)
print(sfs_summary_df.to_string(index=False))

# Visualize accuracy improvement over iterations
print("\n✓ Creating accuracy progression visualization...")
fig_sfs_accuracy = plt.figure(figsize=(12, 6))
fig_sfs_accuracy.patch.set_facecolor('#1D1D20')

ax = fig_sfs_accuracy.add_subplot(111)
ax.set_facecolor('#1D1D20')

# Plot accuracy progression
iterations = list(range(1, len(sfs_accuracy_history) + 1))
ax.plot(iterations, sfs_accuracy_history, 
        marker='o', linewidth=2, markersize=8, 
        color='#A1C9F4', label='Accuracy on TR2')

# Formatting
ax.set_xlabel('Number of Features Selected', fontsize=12, color='#fbfbff')
ax.set_ylabel('Accuracy on TR2', fontsize=12, color='#fbfbff')
ax.set_title('Sequential Forward Selection: Accuracy Progression', 
             fontsize=14, fontweight='bold', color='#fbfbff', pad=20)
ax.tick_params(colors='#fbfbff')
ax.spines['bottom'].set_color('#909094')
ax.spines['left'].set_color('#909094')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, color='#909094', linestyle='-', linewidth=0.5)
ax.legend(loc='lower right', facecolor='#1D1D20', edgecolor='#909094', 
          labelcolor='#fbfbff', fontsize=10)

# Add value labels on points
for i, (x, y) in enumerate(zip(iterations, sfs_accuracy_history)):
    if i % 2 == 0 or i == len(iterations) - 1:  # Label every other point and the last one
        ax.annotate(f'{y:.4f}', (x, y), 
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=8, color='#fbfbff')

plt.tight_layout()
print("✓ Visualization created: fig_sfs_accuracy")

print("\n" + "=" * 80)
print("✅ LEVEL 2 WRAPPER METHOD (SFS) COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"   - Method: Sequential Forward Selection")
print(f"   - Features selected: {len(sfs_selected_features)}")
print(f"   - Final TR2 accuracy: {sfs_accuracy_history[-1]:.6f}")
print(f"   - Computation time: {sfs_elapsed_time:.2f}s")
print("=" * 80)
