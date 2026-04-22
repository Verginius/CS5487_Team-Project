"""
Analysis script for experiment results
"""

import json
import glob
import os
import numpy as np

results = {}

# Read all CV results from trial directories
for trial_dir in sorted(glob.glob('src/results/trial_*')):
    trial_name = os.path.basename(trial_dir)
    for f in sorted(glob.glob(os.path.join(trial_dir, 'cv_results_*.json'))):
        config_name = os.path.basename(f).replace('cv_results_', '').replace('.json', '')
        key = f"{trial_name}/{config_name}"
        with open(f) as fp:
            data = json.load(fp)
            max_score = max(data['mean_test_score'])
            best_idx = data['mean_test_score'].index(max_score)

            results[key] = {
                'trial': trial_name,
                'config': config_name,
                'best_score': max_score,
                'best_params': data['params'][best_idx]
            }

# Summary table
print("="*80)
print("EXPERIMENT RESULTS SUMMARY (per trial)")
print("="*80)
print(f"{'Trial':<10} {'Config':<10} {'Best Score':<12} {'Best Params'}")
print("-"*80)

config_names = ['A', 'B', 'B_linear', 'B_poly', 'B_rbf', 'B_sigmoid',
                'C', 'D', 'D_linear', 'D_poly', 'D_rbf', 'D_sigmoid', 'E', 'F']
dim_reduction_map = {'A': 'PCA', 'B': 'Kernel PCA', 'C': 'PCA', 'D': 'Kernel PCA', 'E': 'None', 'F': 'None'}
classifier_map = {'A': 'SVM', 'B': 'SVM', 'C': 'Random Forest', 'D': 'Random Forest', 'E': 'SVM', 'F': 'Random Forest'}

for config in config_names:
    for trial_dir in sorted(glob.glob('src/results/trial_*')):
        trial_name = os.path.basename(trial_dir)
        key = f"{trial_name}/{config}"
        if key in results:
            print(f"{trial_name:<10} {config:<10} {results[key]['best_score']:<12.4f} {results[key]['best_params']}")

# Aggregated comparison across trials
print("\n" + "="*80)
print("AGGREGATED RESULTS (mean across trials)")
print("="*80)

# Collect scores per config across trials
config_scores = {}
for key, val in results.items():
    c = val['config']
    if c not in config_scores:
        config_scores[c] = []
    config_scores[c].append(val['best_score'])

# 1-NN baseline
baseline = [0.9135, 0.9185]
print(f"{'Config':<15} {'Trial 1':<12} {'Trial 2':<12} {'Mean':<10} {'Std':<10}")
print("-"*60)
print(f"{'1-NN base':<15} {baseline[0]:<12.4f} {baseline[1]:<12.4f} {np.mean(baseline):<10.4f} {np.std(baseline):<10.4f}")
print("-"*60)

for config in sorted(config_scores.keys()):
    scores = config_scores[config]
    t1 = scores[0] if len(scores) > 0 else 0
    t2 = scores[1] if len(scores) > 1 else 0
    print(f"{config:<15} {t1:<12.4f} {t2:<12.4f} {np.mean(scores):<10.4f} {np.std(scores):<10.4f}")

print("\n" + "="*80)
print("PCA THRESHOLD COMPARISON")
print("="*80)

for thresh in ['80', '90', '95', '99']:
    for prefix in ['A', 'C']:
        key = f'{prefix}_{thresh}'
        if key in config_scores:
            scores = config_scores[key]
            clf = 'SVM' if prefix == 'A' else 'Random Forest'
            print(f"{key}: threshold={thresh}%, PCA+{clf}, mean={np.mean(scores):.4f}, scores={scores}")