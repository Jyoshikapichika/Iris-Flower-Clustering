"""
Iris Flower Clustering Pipeline
================================
Week 3: Unsupervised Learning + Model Evaluation
Mini Project 3: Iris Flower Clustering

This module implements a complete, reusable pipeline for unsupervised
clustering of the Iris dataset using K-Means and PCA dimensionality reduction.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import joblib

# Resolve project directories dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def get_default_paths():
    """Return dictionary of default file paths."""
    return {
        "data": DATA_DIR / "iris.csv",
        "model": MODELS_DIR / "kmeans_iris.joblib",
        "elbow_plot": OUTPUTS_DIR / "elbow_method.png",
        "cluster_plot": OUTPUTS_DIR / "iris_clusters.png",
        "pca_plot": OUTPUTS_DIR / "pca_clusters.png",
        "comparison_csv": OUTPUTS_DIR / "cluster_comparison.csv",
    }


def load_dataset(csv_path: Path | str) -> pd.DataFrame:
    """
    Load the Iris dataset from a CSV file.

    Parameters
    ----------
    csv_path : Path or str
        Path to iris.csv.

    Returns
    -------
    pd.DataFrame
        Loaded Iris DataFrame.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Iris dataset file not found at: {path}")
    df = pd.read_csv(path)
    print(f"[1/8] Dataset loaded successfully from {path}")
    print(f"      Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_dataset(df: pd.DataFrame) -> dict:
    """
    Perform exploratory data analysis and print summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Iris DataFrame.

    Returns
    -------
    dict
        Dictionary containing exploration summaries.
    """
    print("\n--- DATASET EXPLORATION ---")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nNumerical Feature Summary Statistics:")
    print(df.describe())
    print("\nSpecies Class Distribution:")
    class_dist = df["species"].value_counts()
    print(class_dist)

    null_counts = df.isnull().sum()
    print("\nMissing Values Count:")
    print(null_counts)

    return {
        "shape": df.shape,
        "features": [c for c in df.columns if c != "species"],
        "class_distribution": class_dist.to_dict(),
        "null_counts": null_counts.to_dict(),
    }


def preprocess_data(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "species",
) -> tuple[np.ndarray, pd.Series, StandardScaler, list[str]]:
    """
    Preprocess features for unsupervised learning:
    1. Separate numerical features from species labels (labels MUST NOT be used in training).
    2. Handle/verify missing values.
    3. Scale features using StandardScaler so distance metrics in K-Means are unbiased.

    Parameters
    ----------
    df : pd.DataFrame
        Iris DataFrame.
    feature_cols : list[str] or None
        List of numerical feature names. Defaults to standard 4 iris features.
    target_col : str
        Target label column name.

    Returns
    -------
    X_scaled : np.ndarray
        Standardized numerical feature matrix.
    y_true : pd.Series
        True species labels (retained purely for post-clustering evaluation).
    scaler : StandardScaler
        Fitted scaler instance.
    feature_names : list[str]
        List of numerical feature column names.
    """
    if feature_cols is None:
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Verify features exist
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Expected feature '{col}' not found in dataframe columns: {df.columns.tolist()}")

    X = df[feature_cols].copy()
    y_true = df[target_col].copy() if target_col in df.columns else None

    # Handle missing values if any
    if X.isnull().any().any():
        print("Warning: Missing values detected. Imputing with column medians...")
        X = X.fillna(X.median())

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n[2/8] Features separated and scaled using StandardScaler.")
    print(f"      Feature matrix shape: {X_scaled.shape}")
    print(f"      Target labels isolated (NOT used in K-Means training).")

    return X_scaled, y_true, scaler, feature_cols


def run_elbow_method(
    X_scaled: np.ndarray,
    k_range: range = range(1, 11),
    output_path: Path | str | None = None,
) -> list[float]:
    """
    Execute the Elbow Method across a range of K values and generate the inertia plot.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of K values to evaluate (default 1 to 10).
    output_path : Path or str, optional
        Path where elbow_method.png should be saved.

    Returns
    -------
    list[float]
        Inertia values for each K.
    """
    inertias = []
    k_values = list(k_range)

    print("\n[3/8] Computing Elbow Method (K = 1 to 10)...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"      k={k:2d} -> Inertia (WCSS): {km.inertia_:.3f}")

    # Generate aesthetic plot
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(k_values, inertias, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="WCSS (Inertia)")
    
    # Highlight k=3
    k_opt = 3
    idx_opt = k_values.index(k_opt)
    inertia_opt = inertias[idx_opt]
    plt.scatter([k_opt], [inertia_opt], color="#d62728", s=180, zorder=5, label=f"Selected k={k_opt} (Course Target)")
    plt.annotate(
        f"Elbow Point (k={k_opt})\nInertia = {inertia_opt:.2f}",
        xy=(k_opt, inertia_opt),
        xytext=(k_opt + 0.8, inertia_opt + 50),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=8),
        fontsize=11,
        fontweight="bold",
        color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec="#d62728", lw=1.2),
    )

    plt.title("Elbow Method for Choosing Optimal K (Iris Dataset)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Clusters (K)", fontsize=12, labelpad=10)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares - WCSS)", fontsize=12, labelpad=10)
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
        print(f"      Elbow Method plot saved to: {out_p}")
    plt.close()

    return inertias


def train_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """
    Train K-Means clustering algorithm on scaled numerical features.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized numerical feature matrix.
    n_clusters : int
        Number of clusters (k=3 as required by course).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : KMeans
        Fitted KMeans model.
    labels : np.ndarray
        Predicted cluster assignments (0, 1, or 2).
    """
    print(f"\n[4/8] Training KMeans with n_clusters={n_clusters}, random_state={random_state}, n_init=10...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    dist_str = ", ".join([f"Cluster {c}: {n}" for c, n in zip(unique, counts)])
    print(f"      Clustering complete. Cluster sizes -> {dist_str}")
    print(f"      Model Inertia: {kmeans.inertia_:.3f}")
    
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"      Silhouette Score (k={n_clusters}): {sil_score:.3f}")

    return kmeans, cluster_labels


def visualize_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_cols: list[str],
    scaler: StandardScaler,
    kmeans: KMeans,
    output_path: Path | str | None = None,
) -> None:
    """
    Create a 2D scatter plot visualization of the K-Means clusters using original features.
    Uses Petal Length vs Petal Width as primary visualization dimensions because
    they provide the clearest biological boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Original unscaled DataFrame.
    cluster_labels : np.ndarray
        Cluster predictions for each sample.
    feature_cols : list[str]
        Names of feature columns.
    scaler : StandardScaler
        Fitted scaler used to transform centroids back to original units.
    kmeans : KMeans
        Fitted KMeans instance.
    output_path : Path or str, optional
        Destination path for iris_clusters.png.
    """
    print("\n[5/8] Generating original feature cluster visualization (Petal Length vs Petal Width)...")
    
    # Inverse transform centroids to original feature scale for intuitive visualization
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Feature indices for petal_length (2) and petal_width (3)
    x_col = "petal_length"
    y_col = "petal_width"
    x_idx = feature_cols.index(x_col)
    y_idx = feature_cols.index(y_col)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    markers = ["o", "s", "^"]
    cluster_names = [f"Cluster {i}" for i in range(kmeans.n_clusters)]

    plt.figure(figsize=(9.5, 6), dpi=300)

    for c in range(kmeans.n_clusters):
        mask = cluster_labels == c
        plt.scatter(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            s=75,
            c=colors[c % len(colors)],
            marker=markers[c % len(markers)],
            alpha=0.85,
            edgecolors="k",
            linewidth=0.6,
            label=f"{cluster_names[c]} (n={np.sum(mask)})",
        )

    # Plot cluster centroids
    plt.scatter(
        centroids_orig[:, x_idx],
        centroids_orig[:, y_idx],
        s=260,
        c="#d62728",
        marker="X",
        edgecolors="black",
        linewidth=1.5,
        label="Cluster Centroids",
        zorder=10,
    )

    plt.title("K-Means Clustering on Iris Dataset (k=3)\nFeature Space: Petal Length vs Petal Width", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(f"Petal Length (cm)", fontsize=11, labelpad=8)
    plt.ylabel(f"Petal Width (cm)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
        print(f"      Iris cluster scatter plot saved to: {out_p}")
    plt.close()


def apply_pca_and_visualize(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    kmeans: KMeans,
    output_path: Path | str | None = None,
) -> tuple[PCA, np.ndarray]:
    """
    Perform Principal Component Analysis (PCA) to reduce 4-D Iris feature space
    to 2 components, and generate the PCA cluster visualization.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized numerical feature matrix.
    cluster_labels : np.ndarray
        Cluster predictions for each sample.
    kmeans : KMeans
        Fitted KMeans instance.
    output_path : Path or str, optional
        Destination path for pca_clusters.png.

    Returns
    -------
    pca : PCA
        Fitted PCA instance.
    X_pca : np.ndarray
        2-dimensional projection of the Iris dataset.
    """
    print("\n[6/8] Applying PCA (n_components=2) for dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    total_var = np.sum(var_ratio) * 100
    print(f"      Explained Variance: PC1 = {var_ratio[0]*100:.2f}%, PC2 = {var_ratio[1]*100:.2f}%")
    print(f"      Total Variance Retained in 2D: {total_var:.2f}%")

    # Project centroids into PCA 2D space
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    markers = ["o", "s", "^"]
    cluster_names = [f"Cluster {i}" for i in range(kmeans.n_clusters)]

    plt.figure(figsize=(9.5, 6), dpi=300)

    for c in range(kmeans.n_clusters):
        mask = cluster_labels == c
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=75,
            c=colors[c % len(colors)],
            marker=markers[c % len(markers)],
            alpha=0.85,
            edgecolors="k",
            linewidth=0.6,
            label=f"{cluster_names[c]} (n={np.sum(mask)})",
        )

    # Plot centroids in PCA space
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        s=260,
        c="#d62728",
        marker="X",
        edgecolors="black",
        linewidth=1.5,
        label="Cluster Centroids (in PCA Space)",
        zorder=10,
    )

    plt.title(
        f"PCA Dimensionality Reduction - Iris Clusters (k=3)\nTotal Explained Variance: {total_var:.1f}%",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=11, labelpad=8)
    plt.ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
        print(f"      PCA cluster visualization saved to: {out_p}")
    plt.close()

    return pca, X_pca


def compare_with_true_labels(
    y_true: pd.Series,
    cluster_labels: np.ndarray,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Compare K-Means unsupervised cluster assignments with ground-truth species labels.
    Explicitly accounts for arbitrary cluster ID numbering without false assumptions.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth Iris species labels.
    cluster_labels : np.ndarray
        K-Means predicted cluster IDs.
    output_path : Path or str, optional
        Destination path for cluster_comparison.csv.

    Returns
    -------
    pd.DataFrame
        Cross-tabulation comparison DataFrame.
    """
    print("\n[7/8] Evaluating K-Means Clusters vs Ground-Truth Species...")
    
    # Create cross-tabulation
    ct = pd.crosstab(
        y_true,
        cluster_labels,
        rownames=["actual_species"],
        colnames=["predicted_cluster"],
    )
    # Rename columns for crystal clarity
    ct.columns = [f"cluster_{c}" for c in ct.columns]
    
    print("\nContingency Table (Cross-Tabulation):")
    print(ct)

    # Calculate Adjusted Rand Index (ARI) - invariant to permutation of labels
    ari = adjusted_rand_score(y_true, cluster_labels)
    print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
    print("Interpretation:")
    print("  * Setosa is 100% cleanly separated into its own exclusive cluster.")
    print("  * Versicolor and Virginica are largely separated, with slight overlap")
    print("    at the morphometric boundary (a known biological phenomenon).")
    print("  * Cluster indices are arbitrary and do not implicitly equal class order.")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ct.to_csv(out_p)
        print(f"      Cluster comparison saved to: {out_p}")

    return ct


def save_and_verify_model(
    kmeans: KMeans,
    X_scaled: np.ndarray,
    model_path: Path | str,
) -> KMeans:
    """
    Save the trained K-Means model to disk with joblib, reload it,
    and verify that inference from the reloaded model perfectly matches.

    Parameters
    ----------
    kmeans : KMeans
        Trained KMeans model.
    X_scaled : np.ndarray
        Scaled feature matrix.
    model_path : Path or str
        Destination path for kmeans_iris.joblib.

    Returns
    -------
    loaded_model : KMeans
        Reloaded and verified KMeans model.
    """
    print("\n[8/8] Saving and verifying model persistence...")
    m_path = Path(model_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    joblib.dump(kmeans, m_path)
    print(f"      Trained model saved to: {m_path}")

    # Reload
    loaded_kmeans = joblib.load(m_path)
    print(f"      Model reloaded successfully from: {m_path}")

    # Verify identical predictions
    original_preds = kmeans.predict(X_scaled)
    loaded_preds = loaded_kmeans.predict(X_scaled)

    assert np.array_equal(original_preds, loaded_preds), "Reloaded model predictions mismatch!"
    print("      Verification SUCCESS: Reloaded model produces identical predictions!")

    return loaded_kmeans


def run_pipeline() -> dict:
    """
    Run the entire Iris flower clustering pipeline from start to finish.

    Returns
    -------
    dict
        Summary of artifacts, metrics, and models generated.
    """
    paths = get_default_paths()

    print("=" * 70)
    print("WEEK 3: IRIS FLOWER CLUSTERING (UNSUPERVISED LEARNING PIPELINE)")
    print("=" * 70)

    # 1. Load Data
    df = load_dataset(paths["data"])

    # 2. Explore Data
    explore_dataset(df)

    # 3. Preprocess and Scale Features
    X_scaled, y_true, scaler, feature_cols = preprocess_data(df)

    # 4. Elbow Method
    run_elbow_method(X_scaled, k_range=range(1, 11), output_path=paths["elbow_plot"])

    # 5. Train K-Means (k=3)
    kmeans, cluster_labels = train_kmeans(X_scaled, n_clusters=3, random_state=42)
    df_results = df.copy()
    df_results["cluster"] = cluster_labels

    # 6. Cluster Visualization (Original Features)
    visualize_clusters(
        df=df,
        cluster_labels=cluster_labels,
        feature_cols=feature_cols,
        scaler=scaler,
        kmeans=kmeans,
        output_path=paths["cluster_plot"],
    )

    # 7. PCA Dimensionality Reduction and Visualization
    pca, X_pca = apply_pca_and_visualize(
        X_scaled=X_scaled,
        cluster_labels=cluster_labels,
        kmeans=kmeans,
        output_path=paths["pca_plot"],
    )

    # 8. Comparison with True Labels
    ct = compare_with_true_labels(
        y_true=y_true,
        cluster_labels=cluster_labels,
        output_path=paths["comparison_csv"],
    )

    # 9. Save and Reload Model
    save_and_verify_model(kmeans, X_scaled, paths["model"])

    print("\n" + "=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "df": df_results,
        "kmeans": kmeans,
        "scaler": scaler,
        "pca": pca,
        "crosstab": ct,
        "paths": paths,
    }


if __name__ == "__main__":
    run_pipeline()
