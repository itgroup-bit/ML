import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

print("="*70)
print("LEVEL 1: CORRELATION-BASED FEATURE SELECTION (CFS)")
print("WITH GENETIC SEARCH ALGORITHM (GSA)")
print("="*70)

# ================================================================
# CORRELATION-BASED FEATURE SELECTION
# ================================================================

print("\n--- Step 1: Compute Feature-Target Correlations ---")

# Compute correlations between all features and target
# Use normalized TR1 data for correlation analysis
correlation_scores = {}
for feature in X_TR1_normalized.columns:
    corr = np.abs(np.corrcoef(X_TR1_normalized[feature], y_TR1)[0, 1])
    # Handle NaN correlations (constant features)
    if np.isnan(corr):
        corr = 0.0
    correlation_scores[feature] = corr

# Sort features by absolute correlation
sorted_features = sorted(correlation_scores.items(), key=lambda x: x[1], reverse=True)

print(f"✓ Computed correlations for {len(sorted_features)} features")
print(f"\nTop 10 features by correlation with target:")
for i, (feat, corr) in enumerate(sorted_features[:10], 1):
    print(f"  {i:2d}. {feat:35s} : {corr:.4f}")

# ================================================================
# GENETIC SEARCH ALGORITHM (GSA) FOR FEATURE SELECTION
# ================================================================

print("\n" + "="*70)
print("GENETIC SEARCH ALGORITHM")
print("="*70)

# Define fitness function from Equation 4 in paper:
# fitness(X) = (3/4) * Accuracy + (1/4) * (1 - (S+F)/2)
# where S = selected features / total features
#       F = feature-target correlation (normalized)

def compute_fitness(individual, all_features):
    """
    Compute fitness using Equation 4 from the paper.
    individual: binary array indicating selected features
    """
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    
    # Handle edge case: no features selected
    if len(selected_indices) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    selected_features = [all_features[i] for i in selected_indices]
    
    # Train Random Forest on TR1 with selected features
    X_train_subset = X_TR1_normalized[selected_features]
    X_val_subset = X_TR2_normalized[selected_features]
    
    rf_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train_subset, y_TR1)
    
    # Evaluate on TR2 (validation set)
    y_pred = rf_model.predict(X_val_subset)
    accuracy = accuracy_score(y_TR2, y_pred)
    
    # Compute S: ratio of selected features to total features
    S = len(selected_indices) / len(all_features)
    
    # Compute F: average correlation of selected features (normalized)
    avg_correlation = np.mean([correlation_scores[feat] for feat in selected_features])
    # Normalize to [0,1] assuming max correlation is 1.0
    F = 1.0 - avg_correlation
    
    # Equation 4: fitness = (3/4)*Accuracy + (1/4)*(1 - (S+F)/2)
    fitness = (3/4) * accuracy + (1/4) * (1 - (S + F) / 2)
    
    return fitness, accuracy, S, F

# ================================================================
# GA PARAMETERS
# ================================================================

all_feature_names = X_TR1_normalized.columns.tolist()
n_features = len(all_feature_names)

# GA hyperparameters
POPULATION_SIZE = 50
N_GENERATIONS = 30
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3

print(f"\nGA Configuration:")
print(f"  Population Size: {POPULATION_SIZE}")
print(f"  Generations: {N_GENERATIONS}")
print(f"  Mutation Rate: {MUTATION_RATE}")
print(f"  Crossover Rate: {CROSSOVER_RATE}")
print(f"  Tournament Size: {TOURNAMENT_SIZE}")
print(f"  Total Features: {n_features}")

# ================================================================
# INITIALIZE POPULATION
# ================================================================

random.seed(42)
np.random.seed(42)

# Initialize population with random binary vectors
population = []
for _ in range(POPULATION_SIZE):
    # Random selection: each feature has 50% chance of being selected
    individual = np.random.randint(0, 2, n_features).tolist()
    population.append(individual)

print(f"\n✓ Initialized population of {POPULATION_SIZE} individuals")

# ================================================================
# GENETIC ALGORITHM EVOLUTION
# ================================================================

print("\n--- Running Genetic Algorithm ---")

best_overall_fitness = 0.0
best_overall_individual = None
best_overall_metrics = None

fitness_history_gen = []
accuracy_history_gen = []

for generation in range(N_GENERATIONS):
    # Evaluate fitness for all individuals
    fitness_results = []
    for individual in population:
        fitness, acc, s, f = compute_fitness(individual, all_feature_names)
        fitness_results.append((fitness, acc, s, f))
    
    # Track best individual in this generation
    gen_best_idx = np.argmax([f[0] for f in fitness_results])
    gen_best_fitness, gen_best_acc, gen_best_s, gen_best_f = fitness_results[gen_best_idx]
    gen_avg_fitness = np.mean([f[0] for f in fitness_results])
    gen_avg_accuracy = np.mean([f[1] for f in fitness_results])
    
    fitness_history_gen.append(gen_best_fitness)
    accuracy_history_gen.append(gen_best_acc)
    
    # Update best overall individual
    if gen_best_fitness > best_overall_fitness:
        best_overall_fitness = gen_best_fitness
        best_overall_individual = population[gen_best_idx].copy()
        best_overall_metrics = (gen_best_acc, gen_best_s, gen_best_f)
    
    # Print progress every 5 generations
    if (generation + 1) % 5 == 0:
        n_selected = sum(population[gen_best_idx])
        print(f"  Gen {generation+1:2d}: Best Fitness={gen_best_fitness:.4f}, "
              f"Accuracy={gen_best_acc:.4f}, Features={n_selected}, "
              f"Avg Fitness={gen_avg_fitness:.4f}")
    
    # ================================================================
    # SELECTION, CROSSOVER, MUTATION
    # ================================================================
    
    new_population = []
    
    while len(new_population) < POPULATION_SIZE:
        # Tournament selection for parent 1
        tournament_indices = random.sample(range(POPULATION_SIZE), TOURNAMENT_SIZE)
        parent1_idx = max(tournament_indices, key=lambda i: fitness_results[i][0])
        
        # Tournament selection for parent 2
        tournament_indices = random.sample(range(POPULATION_SIZE), TOURNAMENT_SIZE)
        parent2_idx = max(tournament_indices, key=lambda i: fitness_results[i][0])
        
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        
        # Crossover
        if random.random() < CROSSOVER_RATE:
            crossover_point = random.randint(1, n_features - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
        else:
            child = parent1.copy()
        
        # Mutation
        for i in range(len(child)):
            if random.random() < MUTATION_RATE:
                child[i] = 1 - child[i]
        
        new_population.append(child)
    
    population = new_population

# ================================================================
# FINAL RESULTS
# ================================================================

print("\n" + "="*70)
print("LEVEL 1 CFS+GSA RESULTS")
print("="*70)

best_acc, best_s, best_f = best_overall_metrics
selected_feature_indices = [i for i, bit in enumerate(best_overall_individual) if bit == 1]
level1_selected_features = [all_feature_names[i] for i in selected_feature_indices]

print(f"\n✅ Genetic Algorithm Completed")
print(f"   Best Fitness Score: {best_overall_fitness:.4f}")
print(f"   Accuracy on TR2: {best_acc:.4f}")
print(f"   Feature Ratio (S): {best_s:.4f} ({len(selected_feature_indices)}/{n_features})")
print(f"   Correlation Penalty (F): {best_f:.4f}")

print(f"\n📊 Selected Features ({len(level1_selected_features)} total):")
for i, feat in enumerate(level1_selected_features, 1):
    corr = correlation_scores[feat]
    print(f"  {i:2d}. {feat:35s} (corr: {corr:.4f})")

print(f"\n📈 Fitness Evolution:")
print(f"   Initial Best Fitness: {fitness_history_gen[0]:.4f}")
print(f"   Final Best Fitness: {fitness_history_gen[-1]:.4f}")
print(f"   Improvement: {(fitness_history_gen[-1] - fitness_history_gen[0]):.4f}")

print("\n" + "="*70)
print(f"✅ LEVEL 1 COMPLETE - {len(level1_selected_features)} features selected")
print("="*70)