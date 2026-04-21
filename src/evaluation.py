"""
Evaluation metrics and visualization module.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multiclass metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average=average),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
    }
    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_scores, n_classes=10, save_path=None):
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        n_classes: Number of classes
        save_path: Path to save the plot
    """
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_classification_report(y_true, y_pred):
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])


def plot_cv_training_curves(cv_results, save_path=None):
    """
    Plot CV training curves (precision/accuracy across hyperparameter combinations).
    
    Args:
        cv_results: Dictionary containing cv_results from GridSearchCV
        save_path: Path to save the plot
    """
    if not cv_results or cv_results.get('mean_test_score') is None or len(cv_results.get('mean_test_score', [])) == 0:
        return
    
    mean_scores = cv_results['mean_test_score']
    std_scores = cv_results['std_test_score']
    params = cv_results['params']
    
    plt.figure(figsize=(12, 6))
    
    # Plot with error bars
    x = range(len(mean_scores))
    plt.errorbar(x, mean_scores, yerr=std_scores, fmt='o-', capsize=5, capthick=2)
    
    # Create param labels
    labels = [str(p) for p in params]
    plt.xticks(x, labels, rotation=45, ha='right')
    
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('CV Score (Accuracy)')
    plt.title('Cross-Validation Training Curves')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_curves(cv_results, save_path=None):
    """
    Plot per-fold accuracy changes for each hyperparameter combination.
    Shows how accuracy varies across CV folds.
    
    Args:
        cv_results: Dictionary containing cv_results from GridSearchCV
        save_path: Path to save the plot
    """
    if not cv_results or cv_results.get('mean_test_score') is None:
        return
    
    split_scores = cv_results.get('split_test_scores', [])
    if not split_scores or len(split_scores) == 0:
        return
    
    n_folds = len(split_scores)
    n_params = len(split_scores[0]) if split_scores else 0
    
    if n_params == 0:
        return
    
    params = cv_results.get('params', [])
    
    # Create subplot for each fold
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4), squeeze=False)
    
    for fold_idx in range(n_folds):
        ax = axes[0, fold_idx]
        fold_scores = split_scores[fold_idx]
        x = range(1, len(fold_scores) + 1)
        
        ax.plot(x, fold_scores, 'o-', linewidth=2, markersize=6, label=f'Fold {fold_idx+1}')
        ax.plot(x, cv_results['mean_test_score'], 's--', linewidth=2, label='Mean', alpha=0.7)
        
        ax.set_xlabel('Parameter Combination')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Fold {fold_idx+1} Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a combined plot showing all folds
    plt.figure(figsize=(12, 6))
    
    for fold_idx in range(n_folds):
        fold_scores = split_scores[fold_idx]
        x = range(1, len(fold_scores) + 1)
        plt.plot(x, fold_scores, 'o-', linewidth=2, markersize=6, label=f'Fold {fold_idx+1}')
    
    # Add mean line
    plt.plot(x, cv_results['mean_test_score'], 's--', linewidth=3, label='Mean', color='black')
    
    plt.xlabel('Parameter Combination')
    plt.ylabel('Accuracy')
    plt.title('Per-Fold Accuracy Changes')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    combined_save_path = save_path.replace('.png', '_combined.png') if save_path else None
    if combined_save_path:
        plt.savefig(combined_save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_cv_results(cv_results, save_path):
    """
    Save CV results to JSON file.
    
    Args:
        cv_results: Dictionary containing cv_results
        save_path: Path to save the JSON file
    """
    # Convert numpy arrays and nested structures to lists for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(cv_results)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def plot_learning_curve(config_name, all_results, save_path=None):
    """
    Plot learning curves showing training progress through CV folds.
    Shows accuracy and loss (1-accuracy) for each hyperparameter combination.
    
    Args:
        config_name: Name of the configuration (or 'all' for all configs)
        all_results: List of all experiment results
        save_path: Path to save the plot
    """
    if config_name == 'all':
        # Create separate plot for each config
        for result in all_results:
            config = result['config']
            cv_results = result.get('cv_results', {})
            mean_scores = cv_results.get('mean_test_score', [])
            
            if len(mean_scores) == 0:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # X-axis: hyperparameter combinations (simulating training steps)
            steps = range(1, len(mean_scores) + 1)
            accuracies = mean_scores
            losses = [1 - acc for acc in accuracies]
            
            plt.plot(steps, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=8)
            plt.plot(steps, losses, 's--', label='Loss (1-Accuracy)', linewidth=2, markersize=8)
            
            plt.xlabel('Hyperparameter Combination (Training Progress)')
            plt.ylabel('Score')
            plt.title(f'Learning Curve - Config {config}\n{result["dim_reduction"]} + {result["classifier"]}')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            
            # Save each config's plot
            config_save_path = save_path.replace('.png', f'_config_{config}.png') if save_path else None
            if config_save_path:
                plt.savefig(config_save_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        # Also create a combined accuracy plot
        plt.figure(figsize=(12, 8))
        for result in all_results:
            config = result['config']
            cv_results = result.get('cv_results', {})
            mean_scores = cv_results.get('mean_test_score', [])
            
            if len(mean_scores) == 0:
                continue
            
            steps = range(1, len(mean_scores) + 1)
            plt.plot(steps, mean_scores, 'o-', label=f'{config}', linewidth=2)
        
        plt.xlabel('Hyperparameter Combination (Training Progress)')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves - All Configurations')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # Single config plot
        for result in all_results:
            if result['config'] != config_name:
                continue
            
            cv_results = result.get('cv_results', {})
            mean_scores = cv_results.get('mean_test_score', [])
            
            if len(mean_scores) == 0:
                return
            
            plt.figure(figsize=(10, 6))
            steps = range(1, len(mean_scores) + 1)
            accuracies = mean_scores
            losses = [1 - acc for acc in accuracies]
            
            plt.plot(steps, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=8)
            plt.plot(steps, losses, 's--', label='Loss (1-Accuracy)', linewidth=2, markersize=8)
            
            plt.xlabel('Hyperparameter Combination (Training Progress)')
            plt.ylabel('Score')
            plt.title(f'Learning Curve - Config {config_name}')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            break
