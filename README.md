# MNIST Dimensionality Reduction and Classification

## 项目概述

本项目对比了多种降维方法（PCA、Kernel PCA）和分类器（SVM、Random Forest、Gradient Boosting）在 MNIST 手写数字识别任务上的表现。

## 项目结构

```
├── data/
│   └── digits4000_txt/       # 预处理后的MNIST子集（4000张图片）
├── docs/                      # 分析报告和计划文档
├── src/
│   ├── analyze_results.py     # 结果分析脚本
│   ├── classifiers.py         # 分类器定义（SVM、Random Forest、Gradient Boosting）
│   ├── config.py              # 实验配置（所有配置的唯一定义来源）
│   ├── data_loader.py         # 数据加载模块
│   ├── dimensionality_reduction.py  # 降维模块（PCA、Kernel PCA）
│   ├── evaluation.py          # 评估指标和可视化
│   ├── experiment.py          # 实验流程
│   └── main.py                # 主入口（支持指定配置运行）
├── results/                   # 实验结果（自动生成）
├── requirements.txt           # 依赖包
└── README.md
```

## 环境配置

### 使用 uv（推荐）

```bash
uv pip install -r requirements.txt
```

### 使用 Conda

```bash
conda env create -f environment.yml  # 或手动安装
conda activate <env_name>
```

### 使用 venv

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

## 运行实验

```bash
# 运行全部配置（30个）
uv run python src/main.py

# 运行指定配置
uv run python src/main.py A C E        # 只跑 A、C、E
uv run python src/main.py B_rbf G_90   # 只跑特定变体
```

### 实验配置

所有配置定义在 `src/config.py` 中的 `EXPERIMENT_CONFIGS`，命名规则如下：

| 配置 | 降维方法 | 分类器 | 变体 |
|------|----------|--------|------|
| A | PCA | SVM | `_80`, `_90`, `_95`, `_99`（方差阈值） |
| B | Kernel PCA | SVM | `_linear`, `_poly`, `_rbf`, `_sigmoid` |
| C | PCA | Random Forest | `_80`, `_90`, `_95`, `_99` |
| D | Kernel PCA | Random Forest | `_linear`, `_poly`, `_rbf`, `_sigmoid` |
| E | 无降维 | SVM | — |
| F | 无降维 | Random Forest | — |
| G | PCA | Gradient Boosting | `_80`, `_90`, `_95`, `_99` |
| H | Kernel PCA | Gradient Boosting | `_linear`, `_poly`, `_rbf`, `_sigmoid` |
| I | 无降维 | Gradient Boosting | — |

不带后缀的基础配置（如 `A`）使用默认 PCA 阈值 95%。

### 分析结果

```bash
uv run python src/analyze_results.py
```

## 实验结果

详细结果见 `src/results/` 目录。

## 依赖包

- numpy
- scikit-learn
- matplotlib
- pandas
- seaborn
