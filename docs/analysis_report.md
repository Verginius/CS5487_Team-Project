# MNIST Dimensionality Reduction and Classification - 实验分析报告

## 1. 实验结果总结

### 1.1 主要配置对比

| 配置 | 降维方法 | 分类器 | 最佳CV准确率 | 最佳参数 |
|------|----------|--------|---------------|----------|
| A | PCA | SVM | **94.95%** | C=10, gamma='scale', kernel='rbf' |
| B | Kernel PCA (RBF) | SVM | **94.90%** | C=10, gamma='scale', kernel='poly' |
| C | PCA | Random Forest | 89.70% | max_depth=10, n_estimators=200 |
| D | Kernel PCA | Random Forest | 86.95% | max_depth=10, n_estimators=200 |
| E | None | SVM | 94.30% | C=1, gamma='scale', kernel='rbf' |
| F | None | Random Forest | 92.85% | max_depth=20, n_estimators=200 |

### 1.2 PCA 阈值对比

| 配置 | 阈值 | 准确率 | 变化 |
|------|------|--------|------|
| A_80 | 80% | **95.30%** |
| A_90 | 90% | **95.40%** |
| A_95 | 95% | 94.95% |
| A_99 | 99% | 94.80% |
| C_80 | 80% | 91.05% |
| C_90 | 90% | 90.60% |
| C_95 | 95% | 89.70% |
| C_99 | 99% | 87.45% |

---

## 2. 数学分析

### 2.1 SVM 分类器性能分析

**核心发现：**
- SVM 在所有配置中表现最佳 (94-95% 准确率)
- RBF kernel 表现最稳定
- PCA 略微提升 SVM 性能 (A vs E: 94.95% vs 94.30%)

**数学解释：**
SVM 的决策函数为：
$$f(x) = \text{sign}\left(\sum_{i=1}^{N} \alpha_i y_i K(x_i, x) + b\right)$$

RBF kernel 定义为：
$$K(x_i, x) = \exp\left(-\gamma \|x_i - x\|^2\right)$$

关键点：
1. **Gamma 参数**：控制 RBF 核的宽度
   - gamma='scale' → $\gamma = \frac{1}{n \cdot \text{var}(X)}$
   - 较小的 gamma 导致更宽松的决策边界，减少过拟合风险

2. **PCA 的作用**：降维后特征相关性降低，使 RBF 核更有效计算相似度

### 2.2 Random Forest 性能分析

**核心发现：**
- RF 在无降维时表现更好 (F: 92.85%)
- PCA 降低 RF 性能
- Kernel PCA 进一步降低性能

**数学解释：**
Random Forest 是集成学习方法：
$$\hat{y} = \text{mode}\{h_k(x)\}_{k=1}^{K}$$

每棵树的构建过程：
- 分裂准则：$Gini = 1 - \sum_{c} p_c^2$
- 最大深度限制树的复杂度

PCA 为何降低 RF 性能：
1. PCA 最大化方差，但可能丢失类别区分信息
2. 线性组合可能破坏原始特征的物理意义
3. RF 本身能处理高维特征

### 2.3 PCA 阈值分析

**趋势：**
- SVM: 80%→90% 上升，之后下降
- RF: 随阈值提高持续下降

**数学解释：**

PCA 方差解释：
$$\text{Var explained} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{n} \lambda_i}$$

- **80-90% 阈值**：保留关键主成分，去除噪声
- **95-99% 阈值**：保留过多成分，可能引入冗余/噪声

最佳阈值选择需要平衡：
- 信息保留 vs 噪声去除

---

## 3. 结论与建议

### 3.1 主要发现

1. **SVM + PCA (RBF kernel)** 是最佳组合（94.95%）
2. **PCA 对 SVM 有帮助**，但对 RF 无帮助
3. **80-90% 方差阈值** 优于 95-99%

### 3.2 数学启示

1. 非线性核（RBF）能捕捉复杂模式
2. 降维应移除噪声而非重要特征
3. 集成方法（RF）本身具有降维能力

### 3.3 改进建议

1. 尝试更细粒度的 PCA 阈值（如 85%）
2. 考虑 LDA 进行监督降维
3. 深度学习方法可能进一步提升性能