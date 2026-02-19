import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import time

print("=" * 80)
print("RANDOM FOREST HYPERPARAMETER TUNING & MODEL TRAINING")
print("=" * 80)

# Use all features from preprocessing for comprehensive model
print(f"\n✓ Training data shape: {X_train_scaled.shape}")
print(f"✓ Test data shape: {X_test_scaled.shape}")
print(f"✓ Number of classes: {len(np.unique(y_train))}")

# Define hyperparameter grid for tuning - smaller grid for faster execution
print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH SPACE")
print("=" * 80)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

print("Parameters to optimize:")
for param, values in param_grid.items():
    print(f"  • {param}: {values}")
    
total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n✓ Total parameter combinations: {total_combinations}")

# Initialize Random Forest
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Grid Search with Cross-Validation
print("\n" + "=" * 80)
print("RUNNING GRID SEARCH WITH 3-FOLD CROSS-VALIDATION")
print("=" * 80)
print("This may take several minutes...")

start_time = time.time()

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train_scaled, y_train)

elapsed_time = time.time() - start_time

print(f"\n✅ Grid search completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

# Display best parameters
print("\n" + "=" * 80)
print("OPTIMAL HYPERPARAMETERS")
print("=" * 80)

best_params = grid_search.best_params_
for param, value in best_params.items():
    print(f"  • {param}: {value}")

print(f"\n✓ Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Get top 5 parameter combinations
cv_results_df = pd.DataFrame(grid_search.cv_results_)
top_5_models = cv_results_df.nlargest(5, 'mean_test_score')[
    ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 
     'mean_test_score', 'std_test_score']
]

print("\n" + "=" * 80)
print("TOP 5 PARAMETER COMBINATIONS")
print("=" * 80)
print(top_5_models.to_string(index=False))

# Train final model with best parameters
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL")
print("=" * 80)

optimized_rf = grid_search.best_estimator_
print(f"✓ Model trained with optimal hyperparameters")

# Evaluate on test set
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

test_predictions = optimized_rf.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\n✓ Test Accuracy: {test_accuracy:.4f}")

# Generate predictions with confidence scores (probabilities)
print("\n" + "=" * 80)
print("GENERATING PREDICTIONS WITH CONFIDENCE SCORES")
print("=" * 80)

test_probabilities = optimized_rf.predict_proba(X_test_scaled)
max_confidence_scores = np.max(test_probabilities, axis=1)

# Create comprehensive predictions dataframe
rf_predictions_df = pd.DataFrame({
    'true_label': y_test.values,
    'predicted_label': test_predictions,
    'confidence_score': max_confidence_scores,
    'correct': (y_test.values == test_predictions)
})

print(f"✓ Generated predictions for {len(rf_predictions_df):,} test samples")
print(f"✓ Average confidence score: {max_confidence_scores.mean():.4f}")
print(f"✓ Min confidence: {max_confidence_scores.min():.4f}")
print(f"✓ Max confidence: {max_confidence_scores.max():.4f}")

# Display sample predictions
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (First 10 samples)")
print("=" * 80)
print(rf_predictions_df.head(10).to_string(index=False))

# Confidence distribution analysis
print("\n" + "=" * 80)
print("CONFIDENCE SCORE DISTRIBUTION")
print("=" * 80)

confidence_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
confidence_dist = pd.cut(max_confidence_scores, bins=confidence_bins).value_counts().sort_index()

print("Confidence Range          | Count      | Percentage")
print("-" * 55)
for interval, count in confidence_dist.items():
    pct = count / len(max_confidence_scores) * 100
    print(f"{str(interval):25s} | {count:8d}   | {pct:6.2f}%")

# Performance by confidence level
high_conf_mask = max_confidence_scores >= 0.9
high_conf_accuracy = accuracy_score(
    y_test.values[high_conf_mask], 
    test_predictions[high_conf_mask]
)

print(f"\n✓ High confidence predictions (≥0.9): {high_conf_mask.sum():,} samples ({high_conf_mask.sum()/len(max_confidence_scores)*100:.1f}%)")
print(f"✓ Accuracy on high confidence predictions: {high_conf_accuracy:.4f}")

# Feature importance from optimized model
feature_importance_rf = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': optimized_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 80)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 80)
print(feature_importance_rf.head(15).to_string(index=False))

# Final summary
print("\n" + "=" * 80)
print("FINAL MODEL SUMMARY")
print("=" * 80)
print(f"✅ Model: Random Forest Classifier")
print(f"✅ Hyperparameters: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")
print(f"✅ Training samples: {X_train_scaled.shape[0]:,}")
print(f"✅ Test samples: {X_test_scaled.shape[0]:,}")
print(f"✅ Features used: {X_train_scaled.shape[1]}")
print(f"✅ Cross-validation accuracy: {grid_search.best_score_:.4f}")
print(f"✅ Test accuracy: {test_accuracy:.4f}")
print(f"✅ Average confidence: {max_confidence_scores.mean():.4f}")
print("=" * 80)
