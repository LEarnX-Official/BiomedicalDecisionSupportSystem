# 🌳 README — Trees and Random Forests for Time Series Classification

## 🧠 Project Summary

This project delivers an advanced, modular, and research-grade implementation of **PromptTree algorithms** and **Random Forest models** specifically adapted for **univariate and multivariate time series classification**. Designed for both **supervised and unsupervised learning**, the implementation supports a broad range of evaluation techniques, model configurations, and analysis workflows—all fully integrated into a single, self-contained Jupyter notebook.

---

## 🚀 Key Features

### 📌 PromptTree Algorithm
- **Reference Slice Tests (RST)** for univariate time series
- **Channel Reference Slice Tests (CRST)** for multivariate time series
- **Customizable components**:
  - Promptness function `fp` (interval selection logic)
  - Sampling function `fs` (candidate slice generation)
  - Classification function `fc` (label assignment logic)
  - Distance function `∆` (multiple built-in options)
  - Optimization function `fo` (test selection)
  - Stopping criterion `fe` (leaf creation control)
- **Post-pruning validation** for optimized tree depth and generalization
- **Interpretability** via path-based distance and feature importance tracking

### 🌲 Random Forest Classifier
- Implements:
  - **Majority Voting**: Uniform voting across decision trees
  - **Weighted Voting**: Accuracy-weighted decision aggregation
  - **Historical Voting**: Track-record based influence per tree
- **Bootstrap sampling** for data variance across trees
- **Feature bagging** for dimensionality randomness and decorrelation
- **Isolation Forest Mode**: Tree-based unsupervised anomaly detection

### 📏 Distance Metrics
- **Breiman Distance**: Based on traditional tree traversal paths
- **Zhu Distance**: Improved discriminative power via enhanced splits
- **RatioRF Distance**: Focuses on relative class purity and distribution

All metrics support **parameter tuning** for domain-specific optimization.

### 🔍 Clustering and Unsupervised Evaluation
- **Hierarchical clustering** using learned tree-based distances
- **Internal validation**:
  - Intra/inter-cluster distance comparisons
  - Silhouette coefficients
- **External validation**:
  - **Purity**: Class consistency within clusters
  - **Entropy**: Disorder-based evaluation
  - **ARI (Adjusted Rand Index)**: Agreement with ground truth labels

### 📊 Supervised and Conformal Evaluation
- **Supervised Metrics**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrices
  - Cross-validation and robustness testing
- **Conformal Prediction**:
  - Miscalibration detection
  - Coverage and efficiency analysis

### 🎨 Visualizations
- Over **13 high-quality visualizations**, including:
  - Accuracy and error plots
  - Heatmaps and dendrograms for clustering
  - Prediction confidence and interval widths
  - Conformal prediction calibration plots
- All plots are **publication-ready**, using libraries like `matplotlib`, `seaborn`, and `plotly`

---




```

## Usage

Open the Jupyter notebook and run all cells:

```bash
jupyter notebook a3.ipynb
```


## Author

**Ritwick Haldar**




This project exemplifies rigorous development and evaluation of decision tree-based methods for time series classification using modern algorithmic strategies.
