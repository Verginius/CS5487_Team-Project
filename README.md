# MNIST Dimensionality Reduction and Classification

## 项目概述

本项目对比了多种降维方法（PCA、Kernel PCA）和分类器（SVM、Random Forest）在 MNIST 手写数字识别任务上的表现。

## 项目结构

```
├── data/
│   └── digits4000_txt/       # 预处理后的MNIST子集（4000张图片）
├── docs/                      # 分析报告和计划文档
├── src/
│   ├── classifiers.py         # 分类器定义（SVM、Random Forest）
│   ├── config.py              # 配置文件
│   ├── data_loader.py         # 数据加载模块
│   ├── dimensionality_reduction.py  # 降维模块（PCA、Kernel PCA）
│   ├── evaluation.py          # 评估指标和可视化
│   ├── experiment.py           # 实验流程
│   └── main.py                # 主入口
├── results/                   # 实验结果（自动生成）
├── requirements.txt           # 依赖包
└── README.md
```

## 环境配置

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
python src/main.py
```

实验会自动遍历以下配置：

### 降维方法
- **PCA**：方差阈值 80%、90%、95%、99%
- **Kernel PCA**：linear、poly、rbf、sigmoid 核
- **无降维**：基准对比

### 分类器
- **SVM**：多参数网格搜索（C、gamma、kernel）
- **Random Forest**：多参数网格搜索（n_estimators、max_depth）

## 实验结果

详细结果见 `src/results/` 目录。

## 依赖包

- numpy
- scikit-learn
- matplotlib
- pandas
- seaborn
