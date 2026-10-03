"""
Enhanced Notebook Generator
===========================
Generates a pre-executed, fully rendered Jupyter Notebook matching all
18 required sections with code, captured outputs, base64 rendered figures,
and markdown analysis cells.
"""

import sys
import io
import json
import base64
import contextlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "iris_clustering.ipynb"


def build_and_execute_notebook():
    cells = []
    exec_env = {}
    exec_count = 1

    def add_markdown(title_or_text: str):
        lines = [line + "\n" for line in title_or_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def run_and_add_code(code_str: str):
        nonlocal exec_count
        source_lines = [line + "\n" for line in code_str.strip().split("\n")]
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip("\n")

        # Execute code in exec_env and capture stdout and matplotlib figures
        import matplotlib.pyplot as plt
        plt.close("all")

        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            try:
                exec(code_str, exec_env)
            except Exception as e:
                print(f"Error executing cell: {e}", file=sys.stderr)
                raise e

        captured_stdout = stdout_buf.getvalue()
        outputs = []

        if captured_stdout:
            stdout_lines = [line + "\n" for line in captured_stdout.split("\n")]
            if stdout_lines and stdout_lines[-1] == "\n":
                stdout_lines.pop()
            if stdout_lines:
                stdout_lines[-1] = stdout_lines[-1].rstrip("\n")
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": stdout_lines
            })

        # Check for open figures
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=120)
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": img_b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {}
            })
            plt.close(fig)

        cells.append({
            "cell_type": "code",
            "execution_count": exec_count,
            "metadata": {},
            "outputs": outputs,
            "source": source_lines
        })
        exec_count += 1

    # ==========================================
    # 1. Project Introduction
    # ==========================================
    add_markdown("""# 1. Project Introduction

## Week 3 Mini Project: Iris Flower Clustering
**Course:** Machine Learning & AI  
**Module:** Unsupervised Learning & Model Evaluation  
**Dataset:** Iris Dataset (UCI / Kaggle)

---

### Project Overview
In supervised learning, algorithms rely on labelled ground-truth targets to guide model training. In **Unsupervised Learning**, we discover underlying patterns, geometric structures, and natural cluster groupings directly from unlabeled feature distributions.

This mini project demonstrates an end-to-end unsupervised learning workflow on the classic **Iris Flower Dataset**:
- **Dataset:** 150 botanical observations with 4 morphological measurements across 3 Iris species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*).
- **Core Algorithm:** **K-Means Clustering** ($k=3$).
- **Key Methodologies:**
  1. Feature separation to ensure target species labels are **never** used during clustering.
  2. Feature standardization using `StandardScaler` to prevent feature scale bias.
  3. The **Elbow Method** to empirically evaluate cluster cohesion (Inertia/WCSS) across $K = 1 \\dots 10$.
  4. Dimensionality reduction using **Principal Component Analysis (PCA)** ($n=2$) to visualize multidimensional clusters in 2D space.
  5. Post-clustering contingency comparison against ground-truth species without making false assumptions about arbitrary cluster IDs.
  6. Model persistence using `joblib` for serialization and inference.""")

    # ==========================================
    # 2. Import Libraries
    # ==========================================
    add_markdown("""---
# 2. Import Libraries
We import essential libraries for data manipulation, visualization, unsupervised learning, and model persistence:
- `pandas` & `numpy`: Tabular data management and numerical linear algebra.
- `matplotlib.pyplot` & `seaborn`: High-resolution visual analytics.
- `sklearn.preprocessing.StandardScaler`: Z-score feature standardization.
- `sklearn.cluster.KMeans`: Lloyd's algorithm for K-Means partitioning.
- `sklearn.decomposition.PCA`: Unsupervised dimensionality reduction.
- `sklearn.metrics`: Cluster quality (Silhouette Score) and agreement evaluation (Adjusted Rand Index).
- `joblib`: Model persistence.""")

    run_and_add_code("""import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import joblib

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

print("All required libraries imported successfully.")""")

    add_markdown("""**Analysis:** The computational environment is initialized with all required dependencies and consistent aesthetic styling for downstream visualizations.""")

    # ==========================================
    # 3. Load Iris Dataset
    # ==========================================
    add_markdown("""---
# 3. Load Iris Dataset
We load the authentic Iris dataset from `data/iris.csv`. Ronald Fisher published this benchmark dataset in 1936, and it remains the cornerstone benchmark for classification and clustering.""")

    run_and_add_code("""# Resolve path dynamically
data_path = Path("data/iris.csv")
if not data_path.exists():
    data_path = Path("../data/iris.csv")

df = pd.read_csv(data_path)
print(f"Dataset successfully loaded from: {data_path.resolve()}")
print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print("\\nFirst 5 observations:")
print(df.head())""")

    add_markdown("""**Analysis:** The dataset loads accurately with 150 rows and 5 columns (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`). All measurements represent flower dimensions in centimeters.""")

    # ==========================================
    # 4. Dataset Exploration
    # ==========================================
    add_markdown("""---
# 4. Dataset Exploration
We conduct comprehensive Exploratory Data Analysis (EDA) using Pandas to inspect data types, feature distributions, missing values, and class balance.""")

    run_and_add_code("""print("--- 1. DATASET SCHEMA & TYPES ---")
df.info()

print("\\n--- 2. DESCRIPTIVE STATISTICS ---")
print(df.describe().round(3))

print("\\n--- 3. SPECIES CLASS DISTRIBUTION ---")
print(df['species'].value_counts())

print("\\n--- 4. MISSING VALUES CHECK ---")
print(df.isnull().sum())""")

    add_markdown("""**Exploration Findings:**
1. **Dimensions:** 150 instances and 5 columns (4 continuous float features, 1 categorical string label).
2. **Missing Values:** Exactly 0 missing or NaN values across all columns.
3. **Class Balance:** Perfectly balanced: 50 *setosa*, 50 *versicolor*, 50 *virginica*.
4. **Variance & Range Differences:**
   - `sepal_length`: range $[4.3, 7.9]$ cm, mean $5.84$
   - `sepal_width`: range $[2.0, 4.4]$ cm, mean $3.06$
   - `petal_length`: range $[1.0, 6.9]$ cm, mean $3.76$ (large variance: $\\sigma = 1.77$)
   - `petal_width`: range $[0.1, 2.5]$ cm, mean $1.20$
   Because `petal_length` has a much larger numeric range than `sepal_width`, unscaled distance metrics would be biased toward petal length.""")

    # ==========================================
    # 5. Feature and Label Separation
    # ==========================================
    add_markdown("""---
# 5. Feature and Label Separation
In unsupervised learning, **target labels must never be used during model training**. 
We extract the 4 numerical feature columns into feature matrix `X` and isolate the true `species` column in `y_true`. 
`y_true` is kept strictly untouched until the post-clustering evaluation phase.""")

    run_and_add_code("""feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Isolate feature matrix X
X = df[feature_cols].copy()

# Isolate target species labels y_true (NOT passed to clustering algorithm)
y_true = df['species'].copy()

print(f"Feature matrix X shape: {X.shape}")
print(f"Features: {feature_cols}")
print(f"True labels y_true shape: {y_true.shape}")
print(f"Unique ground-truth species: {list(y_true.unique())}")""")

    add_markdown("""**Analysis:** Feature matrix `X` contains only the 4 continuous geometric measurements. Target labels are cleanly partitioned away to preserve strict unsupervised integrity.""")

    # ==========================================
    # 6. Data Preprocessing
    # ==========================================
    add_markdown("""---
# 6. Data Preprocessing
We confirm data completeness, check for potential formatting inconsistencies, and ensure all features are valid continuous numerical inputs.""")

    run_and_add_code("""# Defensive check: ensure no NaNs or inf values
assert X.isnull().sum().sum() == 0, "Missing values found!"
assert np.isfinite(X.values).all(), "Non-finite values found!"

print("Data Preprocessing Check:")
print("  * Missing values: 0")
print("  * Infinite values: 0")
print("  * All features: Valid continuous float64")
print("  * Status: READY FOR FEATURE SCALING")""")

    add_markdown("""**Analysis:** Data preprocessing confirms the dataset is clean, complete, and properly formatted.""")

    # ==========================================
    # 7. Feature Scaling
    # ==========================================
    add_markdown("""---
# 7. Feature Scaling
### Why Scaling is Essential for Distance-Based Clustering:
K-Means clustering relies on calculating **Euclidean distance** between observations and centroids:
$$d(\\mathbf{p}, \\mathbf{q}) = \\sqrt{\\sum_{i=1}^{n} (p_i - q_i)^2}$$

If features have different scales or variances, features with larger magnitudes dominate the distance calculations, skewing cluster geometry.

We apply `StandardScaler`, which transforms each feature so that it has zero mean ($\\mu = 0$) and unit variance ($\\sigma = 1$):
$$z = \\frac{x - \\mu}{\\sigma}$$""")

    run_and_add_code("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame to verify properties
df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print("Scaled Feature Means (target = 0.0):")
print(df_scaled.mean().round(4).to_dict())

print("\\nScaled Feature Standard Deviations (target = 1.0):")
print(df_scaled.std(ddof=0).round(4).to_dict())

print("\\nFirst 5 rows of standardized features:")
print(df_scaled.head())""")

    add_markdown("""**Analysis:** All 4 features are now standardized with mean $\\approx 0.00$ and standard deviation $1.00$. Each feature now contributes equally and proportionately to Euclidean distance calculations.""")

    # ==========================================
    # 8. Elbow Method
    # ==========================================
    add_markdown("""---
# 8. Elbow Method
### Determining the Number of Clusters ($K$)
The **Elbow Method** tests multiple values of $K$ ($K=1$ through $10$). For each $K$, we fit K-Means and measure the **Inertia** (Within-Cluster Sum of Squares, WCSS):
$$\\text{Inertia} = \\sum_{k=1}^{K} \\sum_{\\mathbf{x} \\in C_k} ||\\mathbf{x} - \\boldsymbol{\\mu}_k||^2$$

The optimal $K$ corresponds to the "elbow"—the point where the rate of inertia reduction begins to diminish significantly.""")

    run_and_add_code("""inertias = []
k_range = list(range(1, 11))

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"k = {k:2d} | Inertia (WCSS) = {km.inertia_:.3f}")

# Plot Elbow Curve
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(k_range, inertias, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='Inertia (WCSS)')

# Highlight k=3 (course requirement and elbow point)
k_opt = 3
inertia_opt = inertias[k_range.index(k_opt)]
ax.scatter([k_opt], [inertia_opt], color='#d62728', s=180, zorder=5, label=f'Chosen k={k_opt} (Elbow / Course Target)')
ax.annotate(
    f"Elbow Point (k={k_opt})\\nInertia = {inertia_opt:.2f}",
    xy=(k_opt, inertia_opt),
    xytext=(k_opt + 0.8, inertia_opt + 55),
    arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=8),
    fontsize=11,
    fontweight='bold',
    color='#8b0000',
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec="#d62728", lw=1.2)
)

ax.set_title("Elbow Method for Choosing Optimal K (Iris Dataset)", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Number of Clusters (K)", fontsize=12)
ax.set_ylabel("Inertia (Within-Cluster Sum of Squares - WCSS)", fontsize=12)
ax.set_xticks(k_range)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=11, loc="upper right")
plt.tight_layout()

# Save figure to outputs
outputs_dir = Path("outputs") if Path("outputs").exists() else Path("../outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(outputs_dir / "elbow_method.png", dpi=300)
plt.show()""")

    add_markdown("""**Elbow Method Analysis:**
- Inertia drops dramatically from $K=1$ ($600.00$) to $K=2$ ($222.36$), reflecting the sharp separation of the *setosa* species from the rest.
- A second notable decrease occurs from $K=2$ to $K=3$ ($139.82$).
- Beyond $K=3$, the curve flattens into a steady gradual decline ($114.09$ at $K=4$, $90.93$ at $K=5$).
- The elbow occurs around $K=2$ and $K=3$. Because domain knowledge indicates 3 biological species, and $K=3$ captures the core structural groupings before diminishing returns set in, **$k=3$ is mathematically sound and directly satisfies the course requirement**.""")

    # ==========================================
    # 9. K-Means Clustering
    # ==========================================
    add_markdown("""---
# 9. K-Means Clustering
We instantiate and train the K-Means model using:
```python
KMeans(n_clusters=3, random_state=42, n_init=10)
```
The model is fitted strictly on the scaled numerical feature matrix `X_scaled`.""")

    run_and_add_code("""kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

print("K-Means Training Summary:")
print(f"  * Clusters (k): {kmeans.n_clusters}")
print(f"  * Converged in: {kmeans.n_iter_} iterations")
print(f"  * Final Inertia: {kmeans.inertia_:.3f}")
print(f"  * Cluster Centers Shape: {kmeans.cluster_centers_.shape}")""")

    add_markdown("""**Analysis:** The K-Means algorithm converged successfully, computing 3 cluster centroid coordinates in the standardized 4-dimensional feature space.""")

    # ==========================================
    # 10. Cluster Predictions
    # ==========================================
    add_markdown("""---
# 10. Cluster Predictions
We generate cluster assignments for all 150 instances and attach them to a results DataFrame. We also evaluate cluster cohesion via the **Silhouette Score**.""")

    run_and_add_code("""cluster_labels = kmeans.predict(X_scaled)

df_results = df.copy()
df_results['cluster'] = cluster_labels

print("Predicted Cluster Distribution:")
print(df_results['cluster'].value_counts().sort_index())

sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"\\nOverall Silhouette Score: {sil_score:.3f}")

print("\\nSample of Clustered DataFrame:")
print(df_results[['sepal_length', 'petal_length', 'petal_width', 'species', 'cluster']].head(10))""")

    add_markdown("""**Analysis:**
- **Cluster 0:** 53 observations
- **Cluster 1:** 50 observations
- **Cluster 2:** 47 observations
- **Silhouette Score:** $0.460$, indicating substantial within-cluster cohesion and separation from neighboring clusters.""")

    # ==========================================
    # 11. Cluster Visualization
    # ==========================================
    add_markdown("""---
# 11. Cluster Visualization
We visualize the predicted K-Means clusters using the original features **Petal Length vs Petal Width**. 
In botanical literature, petal dimensions provide the sharpest phenotypic boundaries between species.
We map the 4D cluster centroids back to original feature scale using `scaler.inverse_transform()` so they align with the natural centimeter units.""")

    run_and_add_code("""centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)

x_col, y_col = "petal_length", "petal_width"
x_idx, y_idx = feature_cols.index(x_col), feature_cols.index(y_col)

colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
markers = ["o", "s", "^"]
cluster_names = [f"Cluster {i}" for i in range(3)]

fig, ax = plt.subplots(figsize=(9.5, 6))

for c in range(3):
    mask = df_results['cluster'] == c
    ax.scatter(
        df_results.loc[mask, x_col],
        df_results.loc[mask, y_col],
        s=75,
        c=colors[c],
        marker=markers[c],
        alpha=0.85,
        edgecolors="k",
        linewidth=0.6,
        label=f"{cluster_names[c]} (n={np.sum(mask)})"
    )

# Plot Centroids
ax.scatter(
    centroids_orig[:, x_idx],
    centroids_orig[:, y_idx],
    s=260,
    c="#d62728",
    marker="X",
    edgecolors="black",
    linewidth=1.5,
    label="Cluster Centroids",
    zorder=10
)

ax.set_title("K-Means Clustering on Iris Dataset (k=3)\\nFeature Space: Petal Length vs Petal Width", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
plt.tight_layout()

plt.savefig(outputs_dir / "iris_clusters.png", dpi=300)
plt.show()""")

    add_markdown("""**Cluster Visualization Insights:**
1. **Cluster 1 (Green Squares, 50 samples):** Forms a completely isolated cluster characterized by small petals (lengths $1.0 - 1.9$ cm, widths $0.1 - 0.6$ cm).
2. **Cluster 0 (Blue Circles, 53 samples):** Occupies the intermediate region (lengths $3.0 - 5.1$ cm, widths $1.0 - 1.7$ cm).
3. **Cluster 2 (Orange Triangles, 47 samples):** Represents the largest flowers (lengths $4.5 - 6.9$ cm, widths $1.4 - 2.5$ cm).
4. The red `X` markers show the exact geometric centers (centroids) of each cluster.""")

    # ==========================================
    # 12. PCA Dimensionality Reduction
    # ==========================================
    add_markdown("""---
# 12. PCA Dimensionality Reduction
### Purpose of PCA in Clustering:
The Iris dataset has **4 dimensions**. While 2D scatter plots can only display two features at a time (omitting the other two), **Principal Component Analysis (PCA)** finds orthogonal axes of maximum variance and projects all 4 dimensions into a 2D space.

We fit `PCA(n_components=2, random_state=42)` on the scaled feature matrix `X_scaled`.""")

    run_and_add_code("""pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_ratio = pca.explained_variance_ratio_
total_variance = np.sum(var_ratio) * 100

print("PCA Explained Variance Summary:")
print(f"  * Principal Component 1 (PC1): {var_ratio[0]*100:.2f}%")
print(f"  * Principal Component 2 (PC2): {var_ratio[1]*100:.2f}%")
print(f"  * Total Variance Retained:    {total_variance:.2f}%")""")

    add_markdown("""**PCA Analysis:**
- **PC1** captures **72.96%** of the total variance.
- **PC2** captures **22.85%** of the total variance.
- Together, just 2 principal components preserve **95.81%** of all information in the original 4-dimensional space! This demonstrates that our 2D PCA projection provides an accurate, low-loss representation of the multi-dimensional feature space.""")

    # ==========================================
    # 13. PCA Cluster Visualization
    # ==========================================
    add_markdown("""---
# 13. PCA Cluster Visualization
We visualize the K-Means clusters and their centroids projected into the 2D Principal Component subspace.""")

    run_and_add_code("""# Project cluster centroids into PCA subspace
centroids_pca = pca.transform(kmeans.cluster_centers_)

fig, ax = plt.subplots(figsize=(9.5, 6))

for c in range(3):
    mask = cluster_labels == c
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=75,
        c=colors[c],
        marker=markers[c],
        alpha=0.85,
        edgecolors="k",
        linewidth=0.6,
        label=f"{cluster_names[c]} (n={np.sum(mask)})"
    )

# Plot Centroids in PCA space
ax.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    s=260,
    c="#d62728",
    marker="X",
    edgecolors="black",
    linewidth=1.5,
    label="Cluster Centroids (PCA Space)",
    zorder=10
)

ax.set_title(f"PCA Dimensionality Reduction - Iris Clusters (k=3)\\nTotal Retained Variance: {total_variance:.1f}%", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=11)
ax.set_ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
plt.tight_layout()

plt.savefig(outputs_dir / "pca_clusters.png", dpi=300)
plt.show()""")

    add_markdown("""**PCA Visualization Insights:**
- In PCA space, **Cluster 1** is cleanly separated on the negative PC1 axis ($< -1.0$).
- **Cluster 0** and **Cluster 2** are separated along the positive PC1 and PC2 axes.
- Centroids are placed centrally within their respective clusters in the projected 2D space.""")

    # ==========================================
    # 14. Comparison with True Labels
    # ==========================================
    add_markdown("""---
# 14. Comparison with True Labels
### Critical Concept: Arbitrary Cluster IDs
K-Means is an unsupervised algorithm with random centroid initialization. Therefore:
- **Cluster 0 is NOT automatically Setosa.**
- **Cluster 1 is NOT automatically Versicolor.**
- **Cluster 2 is NOT automatically Virginica.**

We create a contingency table (cross-tabulation) between the true botanical species and predicted cluster IDs, and compute the **Adjusted Rand Index (ARI)** to evaluate cluster-to-label agreement without label ordering assumptions.""")

    run_and_add_code("""# Compute cross-tabulation
ct = pd.crosstab(
    df['species'], 
    df_results['cluster'], 
    rownames=['actual_species'], 
    colnames=['predicted_cluster']
)
ct.columns = [f"cluster_{c}" for c in ct.columns]
print("Contingency Table (Cross-Tabulation):")
print(ct)

# Save comparison table
ct.to_csv(outputs_dir / "cluster_comparison.csv")
print(f"\\nSaved cross-tabulation table to: {outputs_dir / 'cluster_comparison.csv'}")

# Calculate Adjusted Rand Index
ari = adjusted_rand_score(y_true, cluster_labels)
print(f"\\nAdjusted Rand Index (ARI): {ari:.4f}")""")

    add_markdown("""**Contingency Matrix Insights:**
1. **Cluster 1 maps 100% to *Setosa*:** Exactly 50/50 samples with zero contamination.
2. **Cluster 0 predominantly maps to *Versicolor*:** 39 Versicolor samples and 14 Virginica samples ($39/53 = 73.6\\%$ purity).
3. **Cluster 2 predominantly maps to *Virginica*:** 36 Virginica samples and 11 Versicolor samples ($36/47 = 76.6\\%$ purity).

**Discussion:**
- As demonstrated, **Cluster 1** represents *Setosa*, not Cluster 0! If we had assumed `Cluster 0 = Setosa`, our conclusions would be false.
- The boundary between *Versicolor* and *Virginica* has slight natural overlap in morphological feature space, which accounts for the 14 and 11 boundary points.
- The **Adjusted Rand Index (ARI)** of **$0.6201$** confirms strong objective agreement between the unsupervised geometric clusters and true species classes.""")

    # ==========================================
    # 15. Model Saving
    # ==========================================
    add_markdown("""---
# 15. Model Saving
We serialize the trained `KMeans` model using `joblib` to `models/kmeans_iris.joblib` so it can be deployed or reloaded in production pipelines.""")

    run_and_add_code("""models_dir = Path("models") if Path("models").exists() else Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "kmeans_iris.joblib"

joblib.dump(kmeans, model_path)
print(f"Trained K-Means model successfully saved to: {model_path.resolve()}")
print(f"Saved file size: {os.path.getsize(model_path)} bytes")""")

    add_markdown("""**Analysis:** Model artifact `kmeans_iris.joblib` is saved and ready for deployment.""")

    # ==========================================
    # 16. Model Loading
    # ==========================================
    add_markdown("""---
# 16. Model Loading & Verification
We load the serialized model back from disk and verify that it produces predictions identical to the in-memory model.""")

    run_and_add_code("""loaded_kmeans = joblib.load(model_path)
print(f"Model successfully loaded from: {model_path.resolve()}")

# Verify predictions match
original_preds = kmeans.predict(X_scaled)
reloaded_preds = loaded_kmeans.predict(X_scaled)

assert np.array_equal(original_preds, reloaded_preds), "Model predictions mismatch!"
print("Verification SUCCESS: Reloaded model reproduces 100% of original predictions.")

# Demonstrate inference on a new unseen sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Typical setosa proportions
new_sample_scaled = scaler.transform(new_sample)
new_pred = loaded_kmeans.predict(new_sample_scaled)[0]

print(f"\\nInference on new observation {new_sample[0]}:")
print(f"  * Predicted Cluster: {new_pred}")
print(f"  * Associated Morphological Group: Setosa-like cluster (Cluster {new_pred})")""")

    add_markdown("""**Analysis:** The model persistence round-trip test passed with 100% prediction parity, and inference on a new sample executed seamlessly.""")

    # ==========================================
    # 17. Final Results
    # ==========================================
    add_markdown("""---
# 17. Final Results Summary
We compile all quantitative metrics from this study into a consolidated summary table.""")

    run_and_add_code("""results_summary = pd.DataFrame({
    "Component / Metric": [
        "Dataset Source",
        "Number of Samples",
        "Number of Features",
        "Feature Scaling Method",
        "Optimal K (Elbow Method)",
        "Chosen K (Course Target)",
        "Final Inertia (WCSS)",
        "Silhouette Score (k=3)",
        "PCA Dimensions",
        "PCA Total Retained Variance",
        "Adjusted Rand Index (ARI)",
        "Cluster 0 Size",
        "Cluster 1 Size",
        "Cluster 2 Size",
        "Model Serialization Format"
    ],
    "Value": [
        "Iris Dataset (Fisher / UCI / Kaggle)",
        "150",
        "4 continuous features",
        "StandardScaler (mean=0, std=1)",
        "k = 2 to 3 (clear elbow transition)",
        "k = 3",
        f"{kmeans.inertia_:.2f}",
        f"{sil_score:.3f}",
        "2 Components (PC1, PC2)",
        f"{total_variance:.2f}%",
        f"{ari:.4f}",
        f"{np.sum(cluster_labels == 0)} (Versicolor dominant)",
        f"{np.sum(cluster_labels == 1)} (Setosa exclusive: 50/50)",
        f"{np.sum(cluster_labels == 2)} (Virginica dominant)",
        "joblib (kmeans_iris.joblib)"
    ]
})

print(results_summary.to_string(index=False))""")

    add_markdown("""**Analysis:** The consolidated summary confirms that the K-Means clustering pipeline successfully satisfied all quantitative and qualitative targets.""")

    # ==========================================
    # 18. Conclusion
    # ==========================================
    add_markdown("""---
# 18. Conclusion & Takeaways

### Key Findings & Technical Insights:
1. **Unsupervised Clustering Integrity:**
   - K-Means clustered the Iris dataset strictly using numerical geometry, without accessing ground-truth species labels during training.
2. **Feature Scaling is Essential:**
   - Without `StandardScaler`, features with larger variance (petal length) would distort Euclidean distance calculations. Scaling normalized all features to unit variance.
3. **Elbow Method:**
   - Inertia dropped from $600.00$ ($k=1$) to $139.82$ ($k=3$), confirming $k=3$ as the optimal balance between cluster compactness and model parsimony.
4. **PCA Dimensionality Reduction:**
   - Projecting 4 dimensions to 2 principal components retained **$95.81\\%$** of total variance, providing an accurate 2D visual representation of the multidimensional cluster space.
5. **Arbitrary Cluster Labels:**
   - The contingency cross-tabulation confirmed that cluster IDs ($0, 1, 2$) are arbitrary index labels. Cluster 1 isolated *Setosa* with $100\\%$ purity, while Cluster 0 and 2 captured *Versicolor* and *Virginica* with an **Adjusted Rand Index of $0.6201$**.
6. **Model Persistence:**
   - The model was saved and loaded via `joblib`, verifying end-to-end reproducibility and inference readiness.

---
**Week 3 Mini Project Complete & Ready for Submission.**""")

    # Assemble notebook dictionary
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"\nSuccessfully generated and pre-executed notebook at: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    build_and_execute_notebook()
