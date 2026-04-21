"""
Dimensionality reduction module using PCA and Kernel PCA.
"""

import numpy as np
from sklearn.decomposition import PCA, KernelPCA


def apply_pca(X_train, X_test, variance_threshold=0.95):
    """
    Apply PCA for linear dimensionality reduction.
    
    Args:
        X_train: Training features
        X_test: Test features
        variance_threshold: Minimum variance to preserve
        
    Returns:
        X_train_pca, X_test_pca, pca_model, n_components
    """
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA: Reduced to {pca.n_components_} components (preserving {variance_threshold*100}% variance)")
    return X_train_pca, X_test_pca, pca, pca.n_components_


def apply_kernel_pca(X_train, X_test, kernel='rbf', **kwargs):
    """
    Apply Kernel PCA for non-linear dimensionality reduction.
    
    Args:
        X_train: Training features
        X_test: Test features
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        **kwargs: Additional kernel parameters
        
    Returns:
        X_train_kpca, X_test_kpca, kpca_model, n_components
    """
    n_samples = X_train.shape[0]
    n_components = min(n_samples, 500)  # Limit components for efficiency
    
    kpca = KernelPCA(n_components=n_components, kernel=kernel, **kwargs)
    X_train_kpca = kpca.fit_transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    print(f"Kernel PCA ({kernel}): Reduced to {n_components} components")
    return X_train_kpca, X_test_kpca, kpca, n_components


def get_reduction_method(method):
    """
    Get dimensionality reduction function based on method name.
    
    Args:
        method: 'pca', 'kernel_pca', or 'none'
        
    Returns:
        Reduction function
    """
    methods = {
        'pca': apply_pca,
        'kernel_pca_rbf': lambda X_train, X_test, **kwargs: apply_kernel_pca(X_train, X_test, kernel='rbf', **kwargs),
        'kernel_pca_poly': lambda X_train, X_test, **kwargs: apply_kernel_pca(X_train, X_test, kernel='poly', **kwargs),
        'kernel_pca_sigmoid': lambda X_train, X_test, **kwargs: apply_kernel_pca(X_train, X_test, kernel='sigmoid', **kwargs),
        'none': apply_no_reduction,
    }
    return methods.get(method, apply_pca)


def apply_no_reduction(X_train, X_test):
    """
    No dimensionality reduction - return original data.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        X_train, X_test, None, original feature count
    """
    n_components = X_train.shape[1]
    print(f"No reduction: Using {n_components} original features")
    return X_train, X_test, None, n_components
