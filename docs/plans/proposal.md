# MNIST Dimensionality Reduction and SVM Classification Project Proposal

## 1. Project Title
Comparative Analysis of Linear (PCA) and Non-Linear (Kernel PCA) Dimensionality Reduction for Handwritten Digit Classification using Support Vector Machines and Random Forests.

## 2. Objective
The primary objective of this project is to evaluate and compare the effectiveness of linear (PCA) and non-linear (Kernel PCA) dimensionality reduction techniques when paired with Support Vector Machine (SVM) and Random Forest classifiers on the MNIST dataset. We aim to determine if capturing non-linear manifolds in the data leads to significantly better classification performance and how these methods scale in terms of both accuracy and computational overhead for a multi-class problem.

## 3. Dataset
- **Dataset Name:** MNIST (Modified National Institute of Standards and Technology)
- **Description:** A collection of 70,000 grayscale images of handwritten digits (0-9).
- **Format:** Each image is $28 \times 28$ pixels, flattened into a 784-dimensional vector.
- **Classes:** 10 (multi-class classification of digits 0-9).

## 4. Methodology

### 4.1 Data Preprocessing
- **No Class Filtering:** We will utilize the full dataset containing all 10 digit classes (0-9) to evaluate the models on a comprehensive multi-class classification task.
- **Normalization:** Scale pixel values from [0, 255] to [0, 1] using standard scaling or min-max normalization.
- **Dimensionality Reduction Methods:**
    - **Linear PCA:** Standard PCA to extract primary orthogonal components (e.g., preserving 95% variance).
    - **Kernel PCA (kPCA):** Non-linear dimensionality reduction using kernels. We will evaluate:
        - **Linear Kernel** (Baseline)
        - **Polynomial Kernel**
        - **Radial Basis Function (RBF) Kernel**
        - **Sigmoid Kernel**

### 4.2 Classification Models
- **Algorithm 1:** Support Vector Machine (SVM).
- **Kernels:** We will evaluate the following kernels for the SVM classifier:
    - **Linear Kernel**
    - **Polynomial Kernel**
    - **Radial Basis Function (RBF) Kernel**
    - **Sigmoid Kernel**
- **Algorithm 2:** Random Forest (RF).
    - An ensemble learning method for classification, providing a robust baseline and alternative strategy to SVM methods.

### 4.3 Experimental Configurations (A, B, C, D)
The project will evaluate four primary system architectures applied to the full 10-class dataset (digits 0-9):
- **Configuration A (PCA + SVM):** Linear PCA reduction paired with a Linear/Kernel SVM.
- **Configuration B (Kernel PCA + SVM):** Non-linear reduction (evaluated across all kPCA kernels) paired with a Linear/Kernel SVM.
- **Configuration C (PCA + Random Forest):** Linear PCA reduction paired with a Random Forest classifier.
- **Configuration D (Kernel PCA + Random Forest):** Both dimensionality reduction and classification utilize non-linear models.

- **Hyperparameters:** 
    - **SVM:** $C$ (Regularization), $\gamma$ (Kernel coefficient for RBF/Poly/Sigmoid), and $d$ (Degree for Polynomial).
    - **Random Forest:** Number of estimators, maximum depth.
    - **PCA:** Number of components.

## 5. Experimental Design and Evaluation

### 5.1 Trial Structure
The experiment will consist of **2 independent trials**. For each trial:
1.  **Data Split:** The full dataset (containing all digits 0-9) will be randomly shuffled and split:
    - **50% Training Set:** Used for model fitting and hyperparameter optimization.
    - **50% Test Set:** Held out strictly for final performance evaluation.
2.  **Cross-Validation:** Within the 50% training set, **5-fold Cross-Validation** will be used to find the optimal values for SVM parameters ($C, \gamma$) and potentially kPCA parameters.
3.  **Testing:** The best model from cross-validation will be evaluated on the 50% test set.

### 5.2 Performance Metrics
The following metrics will be calculated and averaged across the two trials:
- **Classification Accuracy:** Overall percentage of correctly identified digits.
- **F1-Score:** Weighted harmonic mean of precision and recall across all 10 classes.
- **Precision and Recall:** To analyze the model's performance on a per-class basis (0-9).
- **Confusion Matrix:** To visualize misclassifications across all classes.
- **Receiver Operating Characteristic (ROC) Curve & Area Under Curve (AUC):** To evaluate the model's diagnostic ability across varying thresholds using a Multi-class strategy (One-vs-Rest or One-vs-One).
- **Computational Efficiency:** Training and inference time for PCA-based models vs. kPCA-based models.

### 5.3 Baseline Performance
Based on established benchmarks for the multi-class classification problem in MNIST:
- **Linear Classifier (Baseline):** Typically achieves a baseline performance with moderate error rate compared to complex models.
- **Support Vector Machine and Random Forest (without dimensionality reduction):** Will serve as the primary baselines to quantify the performance gain provided by dimensionality reduction in configurations A, B, C, and D.

## 6. Tools and Libraries
- **Language:** Python 3.x
- **Libraries:** `scikit-learn` (PCA, kPCA, SVM, RandomForestClassifier, GridSearchCV, metrics), `numpy`, `matplotlib` (for visualization), `seaborn`.

## 7. Expected Outcome
We expect to observe a trade-off between computational cost and accuracy. While Kernel PCA is likely to capture more complex patterns in the MNIST data, standard PCA may provide a more efficient baseline that is sufficient for high-accuracy classification when paired with a non-linear SVM kernel.
