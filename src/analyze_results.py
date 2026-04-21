"""
Analysis script for experiment results
"""

import json
import glob
import os

results = {}

# Read all CV results
for f in sorted(glob.glob('src/results/cv_results_*.json')):
    config_name = os.path.basename(f).replace('cv_results_', '').replace('.json', '')
    with open(f) as fp:
        data = json.load(fp)
        max_score = max(data['mean_test_score'])
        best_idx = data['mean_test_score'].index(max_score)
        
        results[config_name] = {
            'best_score': max_score,
            'best_params': data['params'][best_idx]
        }

# Summary table
print("="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"{'Config':<10} {'Dim Red':<15} {'Classifier':<15} {'Best Score':<12} {'Best Params'}")
print("-"*80)

config_names = ['A', 'B', 'C', 'D', 'E', 'F']
dim_reduction_map = {'A': 'PCA', 'B': 'Kernel PCA', 'C': 'PCA', 'D': 'Kernel PCA', 'E': 'None', 'F': 'None'}
classifier_map = {'A': 'SVM', 'B': 'SVM', 'C': 'Random Forest', 'D': 'Random Forest', 'E': 'SVM', 'F': 'Random Forest'}

for config in config_names:
    if config in results:
        print(f"{config:<10} {dim_reduction_map[config]:<15} {classifier_map[config]:<15} {results[config]['best_score']:<12.4f} {results[config]['best_params']}")

print("\n" + "="*80)
print("PCA THRESHOLD COMPARISON")
print("="*80)

for thresh in ['80', '90', '95', '99']:
    for prefix in ['A', 'C']:
        key = f'{prefix}_{thresh}'
        if key in results:
            dim = 'PCA' if prefix == 'A' else 'PCA'
            clf = 'SVM' if prefix == 'A' else 'Random Forest'
            print(f"{key}: threshold={thresh}%, {dim}+{clf}, score={results[key]['best_score']:.4f}")