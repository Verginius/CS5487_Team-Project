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

# Gradient Boosting Configuration
GB_N_ESTIMATORS = [100, 200]
GB_MAX_DEPTH = [3, 5, 10]
GB_LEARNING_RATE = [0.1, 0.5]

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
    # Gradient Boosting configs
    'G': {'dim_reduction': 'pca', 'classifier': 'gradient_boosting'},
    'H': {'dim_reduction': 'kernel_pca', 'classifier': 'gradient_boosting'},
    'I': {'dim_reduction': 'none', 'classifier': 'gradient_boosting'},
}

# Display names for dim_reduction and classifier types
DIM_REDUCTION_DISPLAY = {
    'pca': 'PCA',
    'kernel_pca': 'Kernel PCA',
    'none': 'None',
}

CLASSIFIER_DISPLAY = {
    'svm': 'SVM',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
}

# Which base configs support which variant expansions
_PCA_CONFIGS = ['A', 'C', 'G']
_KERNEL_PCA_CONFIGS = ['B', 'D', 'H']


def generate_configs(config_filter=None):
    """
    Generate the full list of experiment configurations.

    Returns a list of tuples: (config_name, dim_reduction, classifier, extra_params)
    where extra_params is a dict with optional 'kernel' or 'pca_threshold' keys.

    Args:
        config_filter: Optional list of config names to include (e.g. ['A', 'B_rbf', 'G_90']).
                       If None, generates all configs.
    """
    configs = []

    for letter, cfg in EXPERIMENT_CONFIGS.items():
        if cfg['dim_reduction'] == 'none':
            # Base configs without dimensionality reduction
            configs.append((letter, cfg['dim_reduction'], cfg['classifier'], {}))
        elif cfg['dim_reduction'] == 'pca':
            # PCA base config (default threshold 0.95) + threshold variants
            configs.append((letter, cfg['dim_reduction'], cfg['classifier'], {'pca_threshold': 0.95}))
            for threshold in PCA_VARIANCE_THRESHOLD:
                name = f'{letter}_{int(threshold * 100)}'
                configs.append((name, cfg['dim_reduction'], cfg['classifier'], {'pca_threshold': threshold}))
        elif cfg['dim_reduction'] == 'kernel_pca':
            # Kernel PCA variants for each kernel type
            for kernel in KERNEL_PCA_KERNELS:
                name = f'{letter}_{kernel}'
                configs.append((name, cfg['dim_reduction'], cfg['classifier'], {'kernel': kernel}))

    # Apply filter
    if config_filter is not None:
        config_filter_set = set(config_filter)
        configs = [c for c in configs if c[0] in config_filter_set]

    return configs


def get_all_config_names():
    """Return all possible config names (useful for analysis scripts)."""
    return [c[0] for c in generate_configs()]


def get_base_config(config_name):
    """Extract the base letter from a config name (e.g. 'B_rbf' -> 'B')."""
    return config_name.split('_')[0]
