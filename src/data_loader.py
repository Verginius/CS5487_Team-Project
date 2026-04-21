"""
Data loading and preprocessing module for digits4000 dataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def load_digits4000(data_dir='data/digits4000_txt'):
    """
    Load digits4000 dataset from local files.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading digits4000 dataset...")
    
    # Load feature vectors (all digit images)
    vec_path = os.path.join(data_dir, 'digits4000_digits_vec.txt')
    labels_path = os.path.join(data_dir, 'digits4000_digits_labels.txt')
    trainset_path = os.path.join(data_dir, 'digits4000_trainset.txt')
    testset_path = os.path.join(data_dir, 'digits4000_testset.txt')
    
    # Load labels (4000 samples, one per line)
    labels = np.loadtxt(labels_path, dtype=int)
    print(f"Loaded {len(labels)} labels")
    
    # Load feature vectors (tab-delimited)
    # Data is stored as (4000 samples x 784 features) based on actual shape
    X = np.loadtxt(vec_path, dtype=float, delimiter='\t')
    print(f"Loaded feature vectors: {X.shape}")
    
    # Load train/test indices
    trainset = np.loadtxt(trainset_path, dtype=int, delimiter='\t')
    testset = np.loadtxt(testset_path, dtype=int, delimiter='\t')
    
    # Get indices (second column contains sample indices)
    # Convert to 0-based indexing for Python arrays
    train_indices = trainset[:, 1] - 1  # 1-indexed to 0-indexed
    test_indices = testset[:, 1] - 1
    
    # Check bounds
    max_idx = X.shape[0] - 1
    print(f"Index range: 0 to {max_idx}")
    print(f"Train indices range: {train_indices.min()} to {train_indices.max()}")
    print(f"Test indices range: {test_indices.min()} to {test_indices.max()}")
    
    # Extract train/test sets (X is already (n_samples, n_features))
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Classes: {np.unique(y_train)}")
    
    return X_train, X_test, y_train, y_test


def load_mnist():
    """
    Load MNIST dataset from OpenML (fallback option).
    
    Returns:
        X: Feature matrix
        y: Target labels
    """
    from sklearn.datasets import fetch_openml
    print("Loading MNIST dataset from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y


def normalize_data(X, method='minmax'):
    """
    Normalize pixel values to [0, 1] or standard scale.
    
    Args:
        X: Feature matrix
        method: 'minmax' for [0,1] normalization, 'standard' for z-score
        
    Returns:
        Normalized feature matrix
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler


def split_data(X, y, train_size=0.5, test_size=0.5, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target labels
        train_size: Proportion for training
        test_size: Proportion for testing
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_size, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test, normalize=True, method='minmax'):
    """
    Preprocess training and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        normalize: Whether to normalize
        method: Normalization method
        
    Returns:
        Preprocessed X_train, X_test, and fitted scaler
    """
    if normalize:
        X_train, scaler = normalize_data(X_train, method)
        X_test = scaler.transform(X_test)
        return X_train, X_test, scaler
    return X_train, X_test, None
