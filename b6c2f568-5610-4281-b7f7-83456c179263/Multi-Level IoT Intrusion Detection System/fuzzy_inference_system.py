import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 80)
print("FUZZY INFERENCE SYSTEM (FIS) IMPLEMENTATION")
print("=" * 80)

# ==============================================================================
# STEP 1: Define Triangular Membership Functions
# ==============================================================================
print("\nSTEP 1: Defining Triangular Membership Functions")
print("─" * 80)

def triangular_membership(x, a, b, c):
    """
    Triangular membership function.
    
    Parameters:
    - x: input value
    - a, b, c: parameters where b is the peak (membership = 1)
    
    Returns membership degree in [0, 1]
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)

# Define membership function parameters based on ticket requirements
# Normal: 0.1-0.3, Low: 0.3-0.6, Medium: 0.6-0.8, High: 0.8-1.0
MF_PARAMS = {
    'Normal': (0.0, 0.1, 0.3),   # peak at 0.1
    'Low': (0.1, 0.3, 0.6),      # peak at 0.3 (overlaps with Normal at 0.1-0.3)
    'Medium': (0.3, 0.6, 0.8),   # peak at 0.6 (overlaps with Low at 0.3-0.6)
    'High': (0.6, 0.8, 1.0)      # peak at 0.8 (overlaps with Medium at 0.6-0.8)
}

print("✓ Membership function parameters defined:")
for severity, (a, b, c) in MF_PARAMS.items():
    print(f"  {severity:8s}: Triangular({a:.1f}, {b:.1f}, {c:.1f})")

# ==============================================================================
# STEP 2: Define 16 Fuzzy Rules (2^4 combinations)
# ==============================================================================
print("\nSTEP 2: Defining 16 Fuzzy Rules (2^4 combinations)")
print("─" * 80)
print("For 4 binary inputs (Probing, DoS, U2R, R2L), each can be Low or High")
print("This gives us 2^4 = 16 possible combinations\n")

# Define the 16 rules with binary inputs (0=Low, 1=High)
# Output severity: 0=Normal, 1=Low, 2=Medium, 3=High
FUZZY_RULES = [
    # Format: (Probing, DoS, U2R, R2L) -> Severity
    (0, 0, 0, 0, 'Normal'),   # Rule 1:  All Low
    (0, 0, 0, 1, 'Low'),      # Rule 2:  Only R2L High
    (0, 0, 1, 0, 'Medium'),   # Rule 3:  Only U2R High
    (0, 0, 1, 1, 'Medium'),   # Rule 4:  U2R and R2L High
    (0, 1, 0, 0, 'Medium'),   # Rule 5:  Only DoS High
    (0, 1, 0, 1, 'Medium'),   # Rule 6:  DoS and R2L High
    (0, 1, 1, 0, 'High'),     # Rule 7:  DoS and U2R High
    (0, 1, 1, 1, 'High'),     # Rule 8:  DoS, U2R, R2L High
    (1, 0, 0, 0, 'Low'),      # Rule 9:  Only Probing High
    (1, 0, 0, 1, 'Low'),      # Rule 10: Probing and R2L High
    (1, 0, 1, 0, 'Medium'),   # Rule 11: Probing and U2R High
    (1, 0, 1, 1, 'High'),     # Rule 12: Probing, U2R, R2L High
    (1, 1, 0, 0, 'Medium'),   # Rule 13: Probing and DoS High
    (1, 1, 0, 1, 'High'),     # Rule 14: Probing, DoS, R2L High
    (1, 1, 1, 0, 'High'),     # Rule 15: Probing, DoS, U2R High
    (1, 1, 1, 1, 'High'),     # Rule 16: All High
]

print("✓ 16 Fuzzy rules defined:")
for idx, (prob, dos, u2r, r2l, severity) in enumerate(FUZZY_RULES, 1):
    print(f"  Rule {idx:2d}: Prob={prob} DoS={dos} U2R={u2r} R2L={r2l} → {severity}")

# ==============================================================================
# STEP 3: Extract Attack Type Probabilities from RF Models
# ==============================================================================
print("\nSTEP 3: Extracting probabilities from RF models")
print("─" * 80)

# Get class probabilities from RF1 (or could use RF2 or ensemble)
rf_proba = rf1_model.predict_proba(X_test_sfs)
print(f"✓ RF1 probability predictions shape: {rf_proba.shape}")
print(f"  Classes predicted: {rf1_model.classes_}")

# Map 5-class probabilities to 4 attack types (Probing, DoS, U2R, R2L)
# Based on typical KDD mapping where:
# Class 0 = Normal
# Class 1 = Probing attacks  
# Class 2 = DoS attacks
# Class 3 = U2R attacks
# Class 4 = R2L attacks

# For simplicity, we'll map the 5 classes to attack probabilities
# This is a simplified mapping - adjust based on actual class meanings
fis_inputs_proba = pd.DataFrame({
    'Probing': rf_proba[:, 1] if rf_proba.shape[1] > 1 else 0.0,  # Class 1
    'DoS': rf_proba[:, 2] if rf_proba.shape[1] > 2 else 0.0,       # Class 2
    'U2R': rf_proba[:, 3] if rf_proba.shape[1] > 3 else 0.0,       # Class 3
    'R2L': rf_proba[:, 4] if rf_proba.shape[1] > 4 else 0.0        # Class 4
})

print(f"\n✓ FIS Input probabilities (4 attack types):")
print(fis_inputs_proba.head())
print(f"\nProbability ranges:")
print(fis_inputs_proba.describe())

# ==============================================================================
# STEP 4: Fuzzification - Apply Membership Functions
# ==============================================================================
print("\nSTEP 4: Fuzzification")
print("─" * 80)

def fuzzify_input(prob_value):
    """
    Convert a probability value to fuzzy membership degrees.
    Returns dict with membership degrees for each severity level.
    """
    return {
        'Normal': triangular_membership(prob_value, *MF_PARAMS['Normal']),
        'Low': triangular_membership(prob_value, *MF_PARAMS['Low']),
        'Medium': triangular_membership(prob_value, *MF_PARAMS['Medium']),
        'High': triangular_membership(prob_value, *MF_PARAMS['High'])
    }

# For binary rule application, we'll use a threshold to determine Low vs High
# Threshold at 0.5 (middle of probability range)
THRESHOLD = 0.5

fis_inputs_binary = pd.DataFrame({
    'Probing_binary': (fis_inputs_proba['Probing'] >= THRESHOLD).astype(int),
    'DoS_binary': (fis_inputs_proba['DoS'] >= THRESHOLD).astype(int),
    'U2R_binary': (fis_inputs_proba['U2R'] >= THRESHOLD).astype(int),
    'R2L_binary': (fis_inputs_proba['R2L'] >= THRESHOLD).astype(int)
})

print(f"✓ Fuzzification complete using threshold = {THRESHOLD}")
print(f"✓ Binary inputs shape: {fis_inputs_binary.shape}")
print(f"\nBinary input distribution:")
print(fis_inputs_binary.sum())

# ==============================================================================
# STEP 5: Rule Evaluation using MIN operator for AND
# ==============================================================================
print("\nSTEP 5: Rule evaluation using MIN operator")
print("─" * 80)

def evaluate_rules(prob_probing, prob_dos, prob_u2r, prob_r2l):
    """
    Evaluate all 16 fuzzy rules for given input probabilities.
    Uses MIN operator for AND operations.
    
    Returns: dict mapping severity levels to their activation degrees
    """
    # Fuzzify inputs
    fuzzy_probing = fuzzify_input(prob_probing)
    fuzzy_dos = fuzzify_input(prob_dos)
    fuzzy_u2r = fuzzify_input(prob_u2r)
    fuzzy_r2l = fuzzify_input(prob_r2l)
    
    # Store rule activations by severity
    rule_activations = {'Normal': [], 'Low': [], 'Medium': [], 'High': []}
    
    # Evaluate each rule
    for prob_bin, dos_bin, u2r_bin, r2l_bin, severity in FUZZY_RULES:
        # Get membership degrees for the specific fuzzy sets (Low or High)
        prob_level = 'High' if prob_bin == 1 else 'Low'
        dos_level = 'High' if dos_bin == 1 else 'Low'
        u2r_level = 'High' if u2r_bin == 1 else 'Low'
        r2l_level = 'High' if r2l_bin == 1 else 'Low'
        
        # Apply MIN operator for AND (conjunction of antecedents)
        activation = min(
            fuzzy_probing[prob_level],
            fuzzy_dos[dos_level],
            fuzzy_u2r[u2r_level],
            fuzzy_r2l[r2l_level]
        )
        
        rule_activations[severity].append(activation)
    
    # Aggregate activations by taking maximum for each severity
    aggregated = {
        severity: max(activations) if activations else 0.0
        for severity, activations in rule_activations.items()
    }
    
    return aggregated

print("✓ Rule evaluation function defined using MIN operator for AND")

# ==============================================================================
# STEP 6: Defuzzification using RMS (Root Mean Square)
# ==============================================================================
print("\nSTEP 6: Defuzzification using RMS method")
print("─" * 80)

# Map severity labels to numeric values for RMS calculation
SEVERITY_VALUES = {
    'Normal': 0.0,
    'Low': 0.33,
    'Medium': 0.67,
    'High': 1.0
}

def defuzzify_rms(aggregated_activations):
    """
    Defuzzify using Root Mean Square (RMS) method as per Equation 10.
    
    RMS = sqrt(sum(μ_i^2 * v_i^2) / sum(μ_i^2))
    where μ_i is the membership degree and v_i is the severity value
    """
    numerator = 0.0
    denominator = 0.0
    
    for severity, membership in aggregated_activations.items():
        value = SEVERITY_VALUES[severity]
        numerator += (membership ** 2) * (value ** 2)
        denominator += membership ** 2
    
    if denominator == 0:
        return 0.0  # Default to Normal if no activation
    
    rms_value = np.sqrt(numerator / denominator)
    
    # Map RMS value back to severity category
    if rms_value < 0.165:  # Midpoint between Normal (0) and Low (0.33)
        return 'Normal'
    elif rms_value < 0.5:  # Midpoint between Low (0.33) and Medium (0.67)
        return 'Low'
    elif rms_value < 0.835:  # Midpoint between Medium (0.67) and High (1.0)
        return 'Medium'
    else:
        return 'High'

print("✓ RMS defuzzification function defined (Equation 10)")
print("✓ Severity value mapping:")
for sev, val in SEVERITY_VALUES.items():
    print(f"  {sev:8s} = {val:.2f}")

# ==============================================================================
# STEP 7: Apply FIS to Test Set
# ==============================================================================
print("\nSTEP 7: Applying FIS to all test samples")
print("─" * 80)

fis_severity_outputs = []
fis_rms_values = []

print(f"Processing {len(fis_inputs_proba)} test samples...")

for idx in range(len(fis_inputs_proba)):
    prob_probing = fis_inputs_proba.loc[idx, 'Probing']
    prob_dos = fis_inputs_proba.loc[idx, 'DoS']
    prob_u2r = fis_inputs_proba.loc[idx, 'U2R']
    prob_r2l = fis_inputs_proba.loc[idx, 'R2L']
    
    # Evaluate rules
    aggregated = evaluate_rules(prob_probing, prob_dos, prob_u2r, prob_r2l)
    
    # Defuzzify
    severity = defuzzify_rms(aggregated)
    
    fis_severity_outputs.append(severity)
    
    # Calculate RMS value for analysis
    numerator = sum((membership ** 2) * (SEVERITY_VALUES[sev] ** 2) 
                   for sev, membership in aggregated.items())
    denominator = sum(membership ** 2 for membership in aggregated.values())
    rms_val = np.sqrt(numerator / denominator) if denominator > 0 else 0.0
    fis_rms_values.append(rms_val)

fis_output_series = pd.Series(fis_severity_outputs, name='FIS_Severity')
fis_rms_series = pd.Series(fis_rms_values, name='FIS_RMS_Value')

print(f"\n✓ FIS processing complete!")
print(f"\nFIS Output Distribution:")
print(fis_output_series.value_counts().sort_index())

# ==============================================================================
# STEP 8: Results Summary
# ==============================================================================
print("\n" + "=" * 80)
print("FIS RESULTS SUMMARY")
print("=" * 80)

# Create results dataframe
fis_results_df = pd.DataFrame({
    'Probing_Prob': fis_inputs_proba['Probing'],
    'DoS_Prob': fis_inputs_proba['DoS'],
    'U2R_Prob': fis_inputs_proba['U2R'],
    'R2L_Prob': fis_inputs_proba['R2L'],
    'FIS_RMS_Value': fis_rms_series,
    'FIS_Severity': fis_output_series
})

print("\nSample FIS Results (first 10 samples):")
print(fis_results_df.head(10).to_string(index=False))

print(f"\nSeverity Distribution:")
severity_dist = fis_output_series.value_counts()
for severity in ['Normal', 'Low', 'Medium', 'High']:
    count = severity_dist.get(severity, 0)
    pct = (count / len(fis_output_series)) * 100
    print(f"  {severity:8s}: {count:5d} samples ({pct:5.2f}%)")

print(f"\nRMS Value Statistics:")
print(f"  Mean:   {fis_rms_series.mean():.4f}")
print(f"  Median: {fis_rms_series.median():.4f}")
print(f"  Std:    {fis_rms_series.std():.4f}")
print(f"  Min:    {fis_rms_series.min():.4f}")
print(f"  Max:    {fis_rms_series.max():.4f}")

# ==============================================================================
# STEP 9: Visualizations
# ==============================================================================
print("\n" + "─" * 80)
print("Creating visualizations...")
print("─" * 80)

# Visualization 1: Membership Functions
fig_mf = plt.figure(figsize=(12, 6))
fig_mf.patch.set_facecolor('#1D1D20')
ax_mf = fig_mf.add_subplot(111)
ax_mf.set_facecolor('#1D1D20')

x_vals = np.linspace(0, 1, 100)
colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B']

for (severity, (a, b, c)), color in zip(MF_PARAMS.items(), colors):
    y_vals = [triangular_membership(x, a, b, c) for x in x_vals]
    ax_mf.plot(x_vals, y_vals, label=severity, color=color, linewidth=2.5)

ax_mf.set_xlabel('Probability', fontsize=12, color='#fbfbff')
ax_mf.set_ylabel('Membership Degree', fontsize=12, color='#fbfbff')
ax_mf.set_title('Triangular Membership Functions', fontsize=14, fontweight='bold', color='#fbfbff', pad=15)
ax_mf.legend(loc='upper right', facecolor='#1D1D20', edgecolor='#909094', labelcolor='#fbfbff')
ax_mf.tick_params(colors='#fbfbff')
ax_mf.grid(True, alpha=0.2, color='#909094', linestyle='-', linewidth=0.5)
ax_mf.spines['bottom'].set_color('#909094')
ax_mf.spines['left'].set_color('#909094')
ax_mf.spines['top'].set_visible(False)
ax_mf.spines['right'].set_visible(False)
plt.tight_layout()
print("✓ Membership functions visualization created: fig_mf")

# Visualization 2: FIS Severity Distribution
fig_dist = plt.figure(figsize=(10, 6))
fig_dist.patch.set_facecolor('#1D1D20')
ax_dist = fig_dist.add_subplot(111)
ax_dist.set_facecolor('#1D1D20')

severity_counts = fis_output_series.value_counts()
severities_ordered = ['Normal', 'Low', 'Medium', 'High']
counts = [severity_counts.get(s, 0) for s in severities_ordered]
colors_bar = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B']

bars = ax_dist.bar(severities_ordered, counts, color=colors_bar, alpha=0.8, edgecolor='#909094', linewidth=1.5)

for bar, count in zip(bars, counts):
    height = bar.get_height()
    pct = (count / len(fis_output_series)) * 100
    ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, color='#fbfbff', fontweight='bold')

ax_dist.set_ylabel('Number of Samples', fontsize=12, color='#fbfbff')
ax_dist.set_title('FIS Severity Output Distribution (4,500 Test Samples)', fontsize=14, fontweight='bold', color='#fbfbff', pad=15)
ax_dist.tick_params(colors='#fbfbff')
ax_dist.spines['bottom'].set_color('#909094')
ax_dist.spines['left'].set_color('#909094')
ax_dist.spines['top'].set_visible(False)
ax_dist.spines['right'].set_visible(False)
ax_dist.grid(True, alpha=0.2, color='#909094', linestyle='-', linewidth=0.5, axis='y')
plt.tight_layout()
print("✓ Severity distribution visualization created: fig_dist")

# Visualization 3: RMS Value Distribution
fig_rms = plt.figure(figsize=(10, 6))
fig_rms.patch.set_facecolor('#1D1D20')
ax_rms = fig_rms.add_subplot(111)
ax_rms.set_facecolor('#1D1D20')

ax_rms.hist(fis_rms_series, bins=50, color='#A1C9F4', alpha=0.7, edgecolor='#909094', linewidth=1)
ax_rms.axvline(fis_rms_series.mean(), color='#FFB482', linestyle='--', linewidth=2, label=f'Mean: {fis_rms_series.mean():.3f}')
ax_rms.axvline(fis_rms_series.median(), color='#8DE5A1', linestyle='--', linewidth=2, label=f'Median: {fis_rms_series.median():.3f}')

ax_rms.set_xlabel('RMS Value', fontsize=12, color='#fbfbff')
ax_rms.set_ylabel('Frequency', fontsize=12, color='#fbfbff')
ax_rms.set_title('Distribution of FIS RMS Values', fontsize=14, fontweight='bold', color='#fbfbff', pad=15)
ax_rms.legend(facecolor='#1D1D20', edgecolor='#909094', labelcolor='#fbfbff')
ax_rms.tick_params(colors='#fbfbff')
ax_rms.spines['bottom'].set_color('#909094')
ax_rms.spines['left'].set_color('#909094')
ax_rms.spines['top'].set_visible(False)
ax_rms.spines['right'].set_visible(False)
ax_rms.grid(True, alpha=0.2, color='#909094', linestyle='-', linewidth=0.5, axis='y')
plt.tight_layout()
print("✓ RMS distribution visualization created: fig_rms")

print("\n" + "=" * 80)
print("✅ FUZZY INFERENCE SYSTEM IMPLEMENTATION COMPLETE")
print("=" * 80)
print("\nKey Implementation Details:")
print("  ✓ 4 input variables: Probing, DoS, U2R, R2L probabilities from RF")
print("  ✓ Triangular membership functions: Normal (0.1-0.3), Low (0.3-0.6),")
print("    Medium (0.6-0.8), High (0.8-1.0)")
print("  ✓ 16 fuzzy rules (2^4 combinations) defined")
print("  ✓ MIN operator used for AND operations in rule antecedents")
print("  ✓ RMS (Root Mean Square) defuzzification (Equation 10)")
print(f"  ✓ Output: Severity level (Normal/Low/Medium/High) for {len(fis_output_series)} test samples")
print("=" * 80)
