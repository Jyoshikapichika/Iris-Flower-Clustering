# Week 3: Iris Flower Clustering Project
## Unsupervised Learning & Model Evaluation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Verified%20%26%20Complete-brightgreen.svg)]()

---

## 1. Project Title
**Week 3 Machine Learning & AI Mini Project 3: Iris Flower Clustering**

## 2. Project Objective
The primary objective of this mini-project is to apply **Unsupervised Machine Learning** to partition the classic Iris flower dataset into natural clusters using **K-Means Clustering** ($k=3$). Key learning goals include:
- Exploring and understanding multidimensional botanical measurements.
- Standardizing features using `StandardScaler` to ensure unbiased distance calculations.
- Implementing the **Elbow Method** to justify $k=3$ using Within-Cluster Sum of Squares (Inertia).
- Training K-Means without any exposure to ground-truth species labels.
- Visualizing clusters both in original 2D feature space and in reduced 2D **Principal Component Analysis (PCA)** space.
- Objectively evaluating cluster purity against actual Iris species using cross-tabulation without making false assumptions about arbitrary cluster IDs.
- Persisting and reloading the trained model with `joblib`.

---

## 3. Dataset
The project uses the authentic **Iris Dataset** originally introduced by British biologist and statistician Ronald Fisher (1936), available from the UCI Machine Learning Repository and Kaggle.

- **Storage Location:** `data/iris.csv`
- **Total Samples:** 150 observations
- **Target Classes:** 3 balanced species (50 instances each):
  - *Iris-setosa*
  - *Iris-versicolor*
  - *Iris-virginica*
- **Continuous Numerical Features (in cm):**
  1. `sepal_length`: Sepal length in centimeters ($4.3 - 7.9$ cm)
  2. `sepal_width`: Sepal width in centimeters ($2.0 - 4.4$ cm)
  3. `petal_length`: Petal length in centimeters ($1.0 - 6.9$ cm)
  4. `petal_width`: Petal width in centimeters ($0.1 - 2.5$ cm)
- **Data Quality:** Zero missing or null values.

---

## 4. Problem Statement
Given four morphological measurements of Iris flowers, can an unsupervised clustering algorithm discover the three underlying biological species purely from geometric relationships in feature space without access to species labels?

---

## 5. Technologies Used
- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `KMeans`, `PCA`, `silhouette_score`, `adjusted_rand_score`)
- **Visual Analytics:** `matplotlib`, `seaborn`
- **Model Serialization:** `joblib`
- **Interactive Computing:** Jupyter Notebook

---

## 6. Unsupervised Learning Explanation
Unlike **Supervised Learning**—where an algorithm is provided input features alongside labeled target outputs ($X \rightarrow y$)—**Unsupervised Learning** discovers latent patterns, groupings, and underlying distributions within unlabeled data ($X$). 

In this project, true species labels (`species`) are strictly separated and completely withheld from the clustering algorithm. The algorithm relies solely on spatial proximity and density in continuous 4-dimensional Euclidean space.

---

## 7. Preprocessing & Feature Isolation
1. **Feature Separation:** The four numerical attributes (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) are extracted into feature matrix $X$.
2. **Label Isolation:** The ground-truth species column is stored in $y_{\text{true}}$ purely for post-hoc validation.
3. **Data Integrity:** The absence of missing values is verified; non-numeric values are guarded against.

---

## 8. Feature Scaling (`StandardScaler`)
### Why Scaling is Mandatory for Distance-Based Clustering:
K-Means groups data points by minimizing the **Euclidean Distance** between points $\mathbf{p}$ and cluster centroids $\boldsymbol{\mu}$:
$$d(\mathbf{p}, \boldsymbol{\mu}) = \sqrt{\sum_{i=1}^{n} (p_i - \mu_i)^2}$$

In the unscaled Iris data:
- `petal_length` ranges from $1.0$ to $6.9$ cm ($\Delta = 5.9$ cm, variance $\sigma^2 = 3.11$)
- `sepal_width` ranges from $2.0$ to $4.4$ cm ($\Delta = 2.4$ cm, variance $\sigma^2 = 0.19$)

Without scaling, distance calculations would be overwhelmingly dominated by `petal_length`, virtually ignoring `sepal_width`.

`StandardScaler` normalizes each feature to have:
$$\mu = 0, \quad \sigma = 1 \quad \left(z = \frac{x - \mu}{\sigma}\right)$$
This gives every feature equal influence during cluster centroid updates.

---

## 9. K-Means Methodology
**K-Means** partitions $N$ observations into $K$ non-overlapping clusters by minimizing the Within-Cluster Sum of Squares (**WCSS / Inertia**):
$$\text{Inertia} = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} ||\mathbf{x} - \boldsymbol{\mu}_k||^2$$

### Algorithm Steps:
1. **Initialization:** Randomly initialize $K$ centroids using `k-means++` (with `random_state=42`, `n_init=10`).
2. **Assignment:** Assign each data point to its closest centroid using Euclidean distance.
3. **Update:** Recalculate centroids as the mean of all points assigned to that cluster.
4. **Convergence:** Repeat steps 2 and 3 until centroids stabilize (convergence reached within 6 iterations).

---

## 10. The Elbow Method
To validate the choice of $K$, we evaluated $K \in [1, 10]$ and plotted the resulting Inertia curve (`outputs/elbow_method.png`):

| Number of Clusters ($K$) | Inertia (WCSS) | Marginal Reduction |
| :---: | :---: | :---: |
| 1 | 600.000 | - |
| 2 | 222.362 | -377.638 |
| **3** | **139.820** | **-82.542** (Elbow Point) |
| 4 | 114.093 | -25.727 |
| 5 | 90.928 | -23.165 |
| 6 | 81.544 | -9.384 |
| 7 | 72.631 | -8.913 |
| 8 | 62.541 | -10.090 |
| 9 | 55.119 | -7.422 |
| 10 | 47.391 | -7.728 |

### Elbow Discussion:
The steep drop from $K=1 \rightarrow K=2$ and $K=2 \rightarrow K=3$ represents substantial grouping signal. Beyond $K=3$, the curve flattens noticeably with diminishing returns per additional cluster. Coupled with the biological domain context of 3 species, $k=3$ is mathematically sound and directly satisfies the course requirement.

---

## 11. PCA (Principal Component Analysis)
### Purpose & Mathematical Rationale:
Human vision is limited to 2D/3D displays, whereas the Iris dataset occupies a 4-dimensional space. While plotting pairs of original features (e.g., Petal Length vs. Petal Width) shows meaningful slices, it discards two features entirely.

**PCA** computes orthogonal linear combinations of features that maximize variance:
- **Principal Component 1 (PC1):** Captures **72.96%** of total variance.
- **Principal Component 2 (PC2):** Captures **22.85%** of total variance.
- **Total Variance Retained:** **95.81%** in just 2 dimensions!

This demonstrates that plotting clusters on PC1 vs. PC2 (`outputs/pca_clusters.png`) preserves almost all geometric relationships of the 4D space with negligible distortion.

---

## 12. Cluster Visualizations
All visual outputs are saved in high resolution (300 DPI) under `outputs/`:

1. **`outputs/elbow_method.png`**: Inertia vs. $K$ ($1 \dots 10$) with the $k=3$ elbow annotated.
2. **`outputs/iris_clusters.png`**: Scatter plot of `petal_length` vs. `petal_width` with distinct cluster colors, markers, sample counts, and inverse-transformed centroids.
3. **`outputs/pca_clusters.png`**: Scatter plot of PC1 vs. PC2 displaying cluster separation and projected centroids with explained variance percentages on axes.

---

## 13. Comparison with True Labels
### Critical Rule: Arbitrary Cluster IDs
K-Means cluster assignments ($0, 1, 2$) are arbitrary index labels. K-Means does **not** know that Setosa exists, nor does it guarantee that `Cluster 0 == Setosa`.

### Contingency Cross-Tabulation (`outputs/cluster_comparison.csv`):
```
actual_species   cluster_0   cluster_1   cluster_2
setosa                   0          50           0
versicolor              39           0          11
virginica               14           0          36
```

### Analysis & Insights:
- **Cluster 1 $\rightarrow$ *Setosa*:** 50/50 samples ($100\%$ purity). *Setosa* is completely linearly separable from the other two species.
- **Cluster 0 $\rightarrow$ *Versicolor* (Dominant):** 39 Versicolor, 14 Virginica ($73.6\%$ purity).
- **Cluster 2 $\rightarrow$ *Virginica* (Dominant):** 36 Virginica, 11 Versicolor ($76.6\%$ purity).
- **Notice:** Cluster 1 is *Setosa*, while Cluster 0 is *Versicolor*. Assuming `Cluster 0 = Setosa` would have been completely incorrect!
- **Adjusted Rand Index (ARI):** **$0.6201$** — demonstrates high statistical agreement between unsupervised clusters and ground-truth classes.

---

## 14. Model Saving & Loading
- **Saved Model:** `models/kmeans_iris.joblib`
- **Serialization Tool:** `joblib.dump(kmeans, 'models/kmeans_iris.joblib')`
- **Loading & Verification:**
  ```python
  import joblib
  loaded_model = joblib.load('models/kmeans_iris.joblib')
  predictions = loaded_model.predict(X_scaled)
  ```
- **Round-Trip Test:** Reloaded model predictions verified $100\%$ identical to in-memory model predictions.

---

## 15. Project Structure
```
Week3_Iris_Flower_Clustering/
│
├── data/
│   └── iris.csv                 # Authentic 150-sample Iris dataset
│
├── notebooks/
│   └── iris_clustering.ipynb    # Comprehensive 18-section interactive notebook
│
├── outputs/
│   ├── elbow_method.png         # Elbow Method inertia curve (k=1..10)
│   ├── iris_clusters.png        # Original feature space cluster scatter plot
│   ├── pca_clusters.png         # PCA 2D subspace cluster scatter plot
│   └── cluster_comparison.csv   # Contingency table: Predicted vs. True labels
│
├── models/
│   └── kmeans_iris.joblib       # Serialized K-Means model artifact
│
├── src/
│   └── iris_clustering.py       # Reusable, production-grade ML pipeline
│
├── tools/
│   ├── verify_project.py        # Automated 17-point requirement verification suite
│   └── generate_notebook.py     # Script to regenerate notebook structure
│
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
```

---

## 16. How to Run

### Option A: Run the End-to-End Pipeline
To run the complete pipeline and regenerate all outputs:
```bash
python src/iris_clustering.py
```

### Option B: Run Automated Verification Suite
To verify all 17 project requirements and file integrity:
```bash
python tools/verify_project.py
```

### Option C: Launch Jupyter Notebook
To run the interactive step-by-step notebook:
```bash
jupyter notebook notebooks/iris_clustering.ipynb
```

---

## 17. Quantitative Results Summary

| Metric / Attribute | Value |
| :--- | :--- |
| **Number of Observations** | 150 |
| **Number of Features** | 4 continuous measurements |
| **Preprocessing** | `StandardScaler` (zero mean, unit variance) |
| **Chosen Clusters ($k$)** | 3 (verified via Elbow Method) |
| **Final K-Means Inertia** | 139.82 |
| **Silhouette Score** | 0.460 |
| **PCA Dimensionality** | 4D $\rightarrow$ 2D |
| **PCA Total Retained Variance** | **95.81%** (PC1: 72.96%, PC2: 22.85%) |
| **Adjusted Rand Index (ARI)** | **0.6201** |
| **Cluster 1 Composition** | 50 Setosa (100% pure) |
| **Cluster 0 Composition** | 39 Versicolor, 14 Virginica |
| **Cluster 2 Composition** | 36 Virginica, 11 Versicolor |

---

## 18. Conclusion
1. **Unsupervised Discoverability:** Without observing species labels, K-Means successfully separated *Iris-setosa* cleanly into its own cluster and separated *versicolor* and *virginica* with high accuracy.
2. **Standardization:** `StandardScaler` eliminated feature-magnitude bias, allowing all 4 geometric measurements to inform distance calculations.
3. **Dimensionality Reduction:** 2D PCA effectively captured $95.81\%$ of information, validating 2D visualization of high-dimensional groupings.
4. **Label Permutation Understanding:** Proven that unsupervised cluster IDs are arbitrary indices that must be compared via cross-tabulation rather than direct number matching.
