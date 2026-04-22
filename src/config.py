# Configuration for MNIST Dimensionality Reduction and Classification Project

# Dataset
MNIST_PATH = None  # None will fetch from sklearn datasets

# Data Preprocessing
NORMALIZE = True
TRAIN_SIZE = 0.5
TEST_SIZE = 0.5
RANDOM_STATE = 42

# PCA Configuration
PCA_VARIANCE_THRESHOLD = [0.80, 0.90, 0.95, 0.99]  # Multiple thresholds for comparison

# Kernel PCA Configuration
KERNEL_PCA_KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']

# SVM Configuration
SVM_C_VALUES = [0.1, 1, 10]
SVM_GAMMA_VALUES = ['scale', 'auto']
SVM_KERNEL = 'rbf'

# Random Forest Configuration
RF_N_ESTIMATORS = [100, 200]
RF_MAX_DEPTH = [10, 20, None]

# Cross-Validation
CV_FOLDS = 5

# 2 experiment trials with pre-defined train/test splits

# Classification Tasks
DIGIT_CLASSES = list(range(10))  # 0-9

# Experiment Configurations
EXPERIMENT_CONFIGS = {
    'A': {'dim_reduction': 'pca', 'classifier': 'svm'},
    'B': {'dim_reduction': 'kernel_pca', 'classifier': 'svm'},
    'C': {'dim_reduction': 'pca', 'classifier': 'random_forest'},
    'D': {'dim_reduction': 'kernel_pca', 'classifier': 'random_forest'},
    # Baseline: No dimensionality reduction
    'E': {'dim_reduction': 'none', 'classifier': 'svm'},
    'F': {'dim_reduction': 'none', 'classifier': 'random_forest'},
}
