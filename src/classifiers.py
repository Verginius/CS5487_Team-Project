"""
Classification module using SVM, Random Forest, and Gradient Boosting.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def create_svm_classifier(C=1.0, kernel='rbf', gamma='scale', degree=3):
    """
    Create an SVM classifier.
    
    Args:
        C: Regularization parameter
        kernel: Kernel type
        gamma: Kernel coefficient
        degree: Degree for polynomial kernel
        
    Returns:
        SVM classifier
    """
    return SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=42)


def create_random_forest_classifier(n_estimators=100, max_depth=None, random_state=42):
    """
    Create a Random Forest classifier.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Random Forest classifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )


def create_gradient_boosting_classifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
    """
    Create an XGBoost classifier.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum depth of individual trees
        learning_rate: Shrinkage rate
        random_state: Random seed

    Returns:
        XGBClassifier
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='mlogloss',
        verbosity=0,
        n_jobs=-1,
    )


def get_svm_param_grid():
    """
    Get SVM hyperparameter grid for GridSearchCV.
    
    Returns:
        Parameter grid dictionary
    """
    return {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }


def get_rf_param_grid():
    """
    Get Random Forest hyperparameter grid for GridSearchCV.
    
    Returns:
        Parameter grid dictionary
    """
    return {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }


def get_gb_param_grid():
    """
    Get Gradient Boosting hyperparameter grid for GridSearchCV.

    Returns:
        Parameter grid dictionary
    """
    return {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.1, 0.5],
    }


def grid_search_cv(classifier, X_train, y_train, param_grid, cv=5, scoring='accuracy'):
    """
    Perform grid search with cross-validation.
    
    Args:
        classifier: Base classifier
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid
        cv: Number of folds
        scoring: Scoring metric
        
    Returns:
        Best classifier and parameters
    """
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Extract CV results including per-fold scores
    cv_results = {
        'mean_test_score': grid_search.cv_results_.get('mean_test_score', []),
        'std_test_score': grid_search.cv_results_.get('std_test_score', []),
        'params': grid_search.cv_results_.get('params', []),
        'best_index': grid_search.best_index_,
        # Per-fold scores for each parameter combination
        'split_test_scores': [
            grid_search.cv_results_.get(f'split{i}_test_score', [])
            for i in range(cv)
        ],
    }
    
    return grid_search.best_estimator_, grid_search.best_params_, cv_results


def get_classifier(classifier_type):
    """
    Get classifier factory based on type.
    
    Args:
        classifier_type: 'svm' or 'random_forest'
        
    Returns:
        Classifier factory function
    """
    classifiers = {
        'svm': create_svm_classifier,
        'random_forest': create_random_forest_classifier,
        'gradient_boosting': create_gradient_boosting_classifier,
    }
    return classifiers.get(classifier_type, create_svm_classifier)
