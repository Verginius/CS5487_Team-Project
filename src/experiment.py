"""
Main experiment logic for MNIST classification experiments.
"""

import os
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from data_loader import load_digits4000, split_data, preprocess
from dimensionality_reduction import apply_pca, apply_kernel_pca, apply_no_reduction
from classifiers import (
    create_svm_classifier,
    create_random_forest_classifier,
    get_svm_param_grid,
    get_rf_param_grid,
    grid_search_cv
)
from evaluation import calculate_metrics, print_metrics, plot_confusion_matrix, save_cv_results, plot_cv_training_curves, plot_per_fold_curves, plot_learning_curve


class ExperimentPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline that combines dimensionality reduction and classification.
    """
    
    def __init__(self, dim_reduction='pca', classifier='svm', kernel='rbf', 
                 pca_variance=0.95, kpca_kernel='rbf'):
        self.dim_reduction = dim_reduction
        self.classifier_type = classifier
        self.kernel = kernel
        self.pca_variance = pca_variance
        self.kpca_kernel = kpca_kernel
        self.dim_reducer = None
        self.classifier = None
        self.n_components = None
        
    def fit(self, X_train, y_train):
        """Fit the pipeline."""
        # Apply dimensionality reduction
        if self.dim_reduction == 'pca':
            X_train_reduced, _, dim_reducer, n_components = apply_pca(
                X_train, X_train, variance_threshold=self.pca_variance
            )
            self.n_components = n_components
            self.dim_reducer = dim_reducer
        else:  # kernel_pca
            X_train_reduced, _, dim_reducer, n_components = apply_kernel_pca(
                X_train, X_train, kernel=self.kpca_kernel
            )
            self.n_components = n_components
            self.dim_reducer = dim_reducer
            
        # Create and fit classifier
        if self.classifier_type == 'svm':
            param_grid = get_svm_param_grid()
            self.classifier, best_params = grid_search_cv(
                create_svm_classifier(),
                X_train_reduced, y_train, param_grid
            )
        else:
            param_grid = get_rf_param_grid()
            self.classifier, best_params = grid_search_cv(
                create_random_forest_classifier(),
                X_train_reduced, y_train, param_grid
            )
            
        return self
    
    def predict(self, X_test):
        """Predict on test data."""
        X_test_reduced = self.dim_reducer.transform(X_test)
        return self.classifier.predict(X_test_reduced)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        X_test_reduced = self.dim_reducer.transform(X_test)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_test_reduced)
        return None


def run_trial(X_train, X_test, y_train, y_test, config_name, dim_reduction,
              classifier, output_dir='results', pca_threshold=0.95, kernel=None,
              trial=0):
    """
    Run experiment for a given configuration.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        config_name: Configuration name (A, B, C, D, E, F)
        dim_reduction: 'pca' or 'kernel_pca' or 'none'
        classifier: 'svm' or 'random_forest'
        output_dir: Directory to save results
        pca_threshold: Variance threshold for PCA
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"[Trial {trial+1}] Running Configuration {config_name}: {dim_reduction} + {classifier}")
    print(f"{'='*60}")
    
    # Time the dimensionality reduction
    start_dr = time.time()
    if dim_reduction == 'pca':
        X_train_red, X_test_red, reducer, n_components = apply_pca(
            X_train, X_test, variance_threshold=pca_threshold
        )
    elif dim_reduction == 'kernel_pca':
        if kernel is None:
            kernel = 'rbf'
        X_train_red, X_test_red, reducer, n_components = apply_kernel_pca(
            X_train, X_test, kernel=kernel
        )
    else:  # 'none' - no dimensionality reduction
        X_train_red, X_test_red, reducer, n_components = apply_no_reduction(
            X_train, X_test
        )
    time_dr = time.time() - start_dr
    print(f"Dimensionality reduction time: {time_dr:.2f}s")
    
    # Time the classifier training
    start_clf = time.time()
    
    if classifier == 'svm':
        param_grid = get_svm_param_grid()
        base_clf = create_svm_classifier()
        best_clf, best_params, cv_results = grid_search_cv(
            base_clf, X_train_red, y_train, param_grid, cv=5
        )
    else:
        param_grid = get_rf_param_grid()
        base_clf = create_random_forest_classifier()
        best_clf, best_params, cv_results = grid_search_cv(
            base_clf, X_train_red, y_train, param_grid, cv=5
        )
    
    time_clf = time.time() - start_clf
    print(f"Classifier training time: {time_clf:.2f}s")
    
    # Evaluate on test set
    y_pred = best_clf.predict(X_test_red)
    metrics = calculate_metrics(y_test, y_pred)
    
    print_metrics(metrics, f"Config {config_name}")
    
    # Save confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=f"{output_dir}/config_{config_name}_confusion.png"
    )
    
    # Save CV results and training curves
    cv_results_file = f"{output_dir}/cv_results_{config_name}.json"
    save_cv_results(cv_results, cv_results_file)
    
    cv_curve_file = f"{output_dir}/cv_curves_{config_name}.png"
    plot_cv_training_curves(cv_results, save_path=cv_curve_file)
    
    # Save per-fold accuracy curves
    per_fold_file = f"{output_dir}/per_fold_{config_name}.png"
    plot_per_fold_curves(cv_results, save_path=per_fold_file)
    
    return {
        'config': config_name,
        'dim_reduction': dim_reduction,
        'classifier': classifier,
        'pca_threshold': pca_threshold,
        'n_components': n_components,
        'metrics': metrics,
        'best_params': best_params,
        'cv_results': cv_results,
        'time_dr': time_dr,
        'time_clf': time_clf,
        'y_pred': y_pred
    }


def run_experiment(output_dir='results'):
    """
    Run the full experiment with all configurations over 2 trials.

    Args:
        output_dir: Directory to save results

    Returns:
        List of results for all configurations and trials
    """
    all_results = []

    from config import KERNEL_PCA_KERNELS

    configs = [
        ('A', 'pca', 'svm'),
        ('C', 'pca', 'random_forest'),
        ('E', 'none', 'svm'),
        ('F', 'none', 'random_forest'),
    ]
    # Add kernel_pca configs for each kernel
    for kernel in KERNEL_PCA_KERNELS:
        configs.append((f'B_{kernel}', 'kernel_pca', 'svm', kernel))
        configs.append((f'D_{kernel}', 'kernel_pca', 'random_forest', kernel))

    # Add PCA threshold variants for comparison
    pca_thresholds = [0.80, 0.90, 0.95, 0.99]
    for threshold in pca_thresholds:
        configs.append((f'A_{int(threshold*100)}', 'pca', 'svm', threshold))
        configs.append((f'C_{int(threshold*100)}', 'pca', 'random_forest', threshold))

    # Run 2 experiment trials as required by the assignment
    for trial in range(2):
        print(f"\n{'#'*60}")
        print(f"TRIAL {trial + 1}")
        print(f"{'#'*60}")

        # Load data for this trial (pre-defined train/test split)
        X_train, X_test, y_train, y_test = load_digits4000(
            data_dir='data/digits4000_txt', trial=trial
        )
        X_train, X_test, _ = preprocess(X_train, X_test, normalize=True, method='minmax')

        # Create trial-specific output directory
        trial_dir = os.path.join(output_dir, f'trial_{trial+1}')
        os.makedirs(trial_dir, exist_ok=True)

        for config_item in configs:
            if len(config_item) == 4:
                config_name, dim_reduction, classifier, extra = config_item
                if dim_reduction == 'pca':
                    result = run_trial(
                        X_train, X_test, y_train, y_test,
                        config_name, dim_reduction, classifier, trial_dir,
                        pca_threshold=extra, trial=trial
                    )
                elif dim_reduction == 'kernel_pca':
                    result = run_trial(
                        X_train, X_test, y_train, y_test,
                        config_name, dim_reduction, classifier, trial_dir,
                        kernel=extra, trial=trial
                    )
            else:
                config_name, dim_reduction, classifier = config_item
                result = run_trial(
                    X_train, X_test, y_train, y_test,
                    config_name, dim_reduction, classifier, trial_dir,
                    trial=trial
                )
            result['trial'] = trial + 1
            all_results.append(result)

    return all_results


def summarize_results(all_results, output_dir='results'):
    """
    Summarize experiment results across trials.

    Args:
        all_results: List of result dictionaries (with 'trial' field)
        output_dir: Directory to save plots
    """
    import numpy as np

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    # Group results by config
    configs = {}
    for result in all_results:
        key = result['config']
        if key not in configs:
            configs[key] = []
        configs[key].append(result)

    # Print per-trial results
    print(f"\n{'Config':<12} {'Dim Red':<15} {'Classifier':<15} {'Trial 1 Acc':<12} {'Trial 2 Acc':<12} {'Mean':<10} {'Std':<10}")
    print("-"*90)

    # 1-NN baseline from the assignment
    baseline = [0.9135, 0.9185]
    print(f"{'1-NN base':<12} {'None':<15} {'1-NN':<15} {baseline[0]:<12.4f} {baseline[1]:<12.4f} {np.mean(baseline):<10.4f} {np.std(baseline):<10.4f}")
    print("-"*90)

    for config_name, results_list in configs.items():
        accuracies = [r['metrics']['accuracy'] for r in results_list]
        trial1_acc = accuracies[0] if len(accuracies) > 0 else 0
        trial2_acc = accuracies[1] if len(accuracies) > 1 else 0
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        r0 = results_list[0]

        print(f"{config_name:<12} {r0['dim_reduction']:<15} {r0['classifier']:<15} "
              f"{trial1_acc:<12.4f} {trial2_acc:<12.4f} {mean_acc:<10.4f} {std_acc:<10.4f}")

    # Plot learning curves for all configurations
    os.makedirs(output_dir, exist_ok=True)
    learning_curve_path = os.path.join(output_dir, 'learning_curves.png')
    plot_learning_curve('all', all_results, save_path=learning_curve_path)
    print(f"\nLearning curves saved to: {learning_curve_path}")

        # ...existing code...
