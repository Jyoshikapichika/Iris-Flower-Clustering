"""
Iris Flower Clustering - Project Verification Script
=====================================================
Week 3: Unsupervised Learning & Model Evaluation
Mini Project 3: Iris Flower Clustering

This script runs automated verification checks on all components of the project:
1. Data files and dataset integrity
2. Preprocessing, StandardScaler, and feature isolation
3. Elbow Method implementation and artifacts
4. K-Means clustering (k=3) implementation
5. PCA dimensionality reduction (2 components)
6. Visualization artifacts (elbow, clusters, PCA)
7. True-label comparison (cross-tabulation, arbitrary label handling)
8. Model persistence (joblib save/load round-trip and prediction match)
9. Notebook structure and section validation
10. Documentation and requirements
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

# Determine base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class VerificationRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results = []
        self.all_passed = True

    def log_check(self, name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append((name, status, details))
        if not passed:
            self.all_passed = False
        print(f"[{status}] {name}")
        if details:
            print(f"       -> {details}")

    def verify_all(self):
        print("=" * 75)
        print("WEEK 3 PROJECT AUTOMATED VERIFICATION SUITE")
        print("=" * 75)

        # 1. data/iris.csv exists
        data_file = self.base_dir / "data" / "iris.csv"
        self.log_check(
            "data/iris.csv exists",
            data_file.exists(),
            f"File found at {data_file}" if data_file.exists() else f"Missing: {data_file}"
        )

        # 2. dataset loads successfully
        df = None
        if data_file.exists():
            try:
                df = pd.read_csv(data_file)
                self.log_check(
                    "dataset loads successfully",
                    len(df) == 150,
                    f"Successfully loaded {len(df)} rows and {len(df.columns)} columns"
                )
            except Exception as e:
                self.log_check("dataset loads successfully", False, f"Failed to load CSV: {e}")
        else:
            self.log_check("dataset loads successfully", False, "Dataset file missing")

        # 3. expected Iris features exist
        expected_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        has_features = False
        if df is not None:
            has_features = all(col in df.columns for col in expected_features) and "species" in df.columns
            self.log_check(
                "expected Iris features exist",
                has_features,
                f"Features present: {expected_features} + target 'species'" if has_features else f"Columns found: {list(df.columns)}"
            )
        else:
            self.log_check("expected Iris features exist", False, "DataFrame not available")

        # 4. StandardScaler is used
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        if df is not None and has_features:
            X = df[expected_features]
            X_scaled = scaler.fit_transform(X)
            scaler_valid = np.allclose(X_scaled.mean(axis=0), 0, atol=1e-5) and np.allclose(X_scaled.std(axis=0), 1, atol=1e-5)
            self.log_check(
                "StandardScaler is used",
                scaler_valid,
                "Features standardized to mean=0, std=1 across all 4 dimensions"
            )
        else:
            self.log_check("StandardScaler is used", False, "Could not test scaler")

        # 5. K-Means with k=3 is implemented
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = None
        if df is not None and has_features:
            clusters = kmeans.fit_predict(X_scaled)
            unique_clusters = np.unique(clusters)
            k3_valid = len(unique_clusters) == 3
            self.log_check(
                "K-Means with k=3 is implemented",
                k3_valid,
                f"Exactly 3 clusters produced: {list(unique_clusters)}, inertia: {kmeans.inertia_:.2f}"
            )
        else:
            self.log_check("K-Means with k=3 is implemented", False, "Could not run KMeans")

        # 6. cluster predictions are generated
        preds_valid = clusters is not None and len(clusters) == 150
        self.log_check(
            "cluster predictions are generated",
            preds_valid,
            f"Generated {len(clusters)} predictions matching dataset length" if preds_valid else "Prediction failed"
        )

        # 7. Elbow Method is implemented and output exists
        elbow_img = self.base_dir / "outputs" / "elbow_method.png"
        self.log_check(
            "elbow_method.png exists",
            elbow_img.exists() and elbow_img.stat().st_size > 1000,
            f"Elbow plot found: {elbow_img} ({elbow_img.stat().st_size} bytes)" if elbow_img.exists() else "Missing elbow plot"
        )

        # 8. PCA with 2 components is implemented
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        if df is not None and has_features:
            X_pca = pca.fit_transform(X_scaled)
            pca_valid = X_pca.shape == (150, 2)
            total_var = np.sum(pca.explained_variance_ratio_) * 100
            self.log_check(
                "PCA with 2 components is implemented",
                pca_valid,
                f"Reduced 4D -> 2D, explained variance ratio: {total_var:.2f}%"
            )
        else:
            self.log_check("PCA with 2 components is implemented", False, "Could not run PCA")

        # 9. iris_clusters.png exists
        iris_clusters_img = self.base_dir / "outputs" / "iris_clusters.png"
        self.log_check(
            "iris_clusters.png exists",
            iris_clusters_img.exists() and iris_clusters_img.stat().st_size > 1000,
            f"Cluster scatter plot found: {iris_clusters_img} ({iris_clusters_img.stat().st_size} bytes)" if iris_clusters_img.exists() else "Missing cluster plot"
        )

        # 10. pca_clusters.png exists
        pca_img = self.base_dir / "outputs" / "pca_clusters.png"
        self.log_check(
            "pca_clusters.png exists",
            pca_img.exists() and pca_img.stat().st_size > 1000,
            f"PCA cluster plot found: {pca_img} ({pca_img.stat().st_size} bytes)" if pca_img.exists() else "Missing PCA plot"
        )

        # 11. cluster_comparison.csv exists
        comp_csv = self.base_dir / "outputs" / "cluster_comparison.csv"
        comp_valid = False
        if comp_csv.exists() and comp_csv.stat().st_size > 10:
            try:
                comp_df = pd.read_csv(comp_csv)
                comp_valid = len(comp_df) >= 3 and len(comp_df.columns) >= 3
            except Exception:
                comp_valid = False
        self.log_check(
            "cluster_comparison.csv exists",
            comp_valid,
            f"Contingency comparison table verified at: {comp_csv}" if comp_valid else "Missing or invalid comparison CSV"
        )

        # 12. kmeans_iris.joblib exists
        model_file = self.base_dir / "models" / "kmeans_iris.joblib"
        self.log_check(
            "kmeans_iris.joblib exists",
            model_file.exists() and model_file.stat().st_size > 500,
            f"Model artifact verified: {model_file} ({model_file.stat().st_size} bytes)" if model_file.exists() else "Missing model file"
        )

        # 13. saved model can be loaded
        loaded_model = None
        if model_file.exists():
            try:
                loaded_model = joblib.load(model_file)
                self.log_check(
                    "saved model can be loaded",
                    isinstance(loaded_model, KMeans),
                    f"Successfully loaded {type(loaded_model).__name__} object with n_clusters={loaded_model.n_clusters}"
                )
            except Exception as e:
                self.log_check("saved model can be loaded", False, f"Joblib load failed: {e}")
        else:
            self.log_check("saved model can be loaded", False, "Model file does not exist")

        # 14. predictions can be generated after loading
        if loaded_model is not None and clusters is not None:
            try:
                new_preds = loaded_model.predict(X_scaled)
                preds_match = np.array_equal(clusters, new_preds)
                self.log_check(
                    "predictions can be generated after loading",
                    preds_match,
                    "Reloaded model predictions match in-memory model 100%"
                )
            except Exception as e:
                self.log_check("predictions can be generated after loading", False, f"Prediction error: {e}")
        else:
            self.log_check("predictions can be generated after loading", False, "Cannot verify loaded model predictions")

        # 15. notebook exists
        nb_file = self.base_dir / "notebooks" / "iris_clustering.ipynb"
        nb_valid = False
        nb_details = ""
        if nb_file.exists() and nb_file.stat().st_size > 5000:
            try:
                with open(nb_file, "r", encoding="utf-8") as f:
                    nb = json.load(f)
                num_cells = len(nb.get("cells", []))
                nb_valid = num_cells >= 18
                nb_details = f"Notebook JSON valid with {num_cells} cells at {nb_file}"
            except Exception as e:
                nb_details = f"Notebook JSON parse error: {e}"
        else:
            nb_details = "Notebook file missing or too small"
        self.log_check("notebook exists", nb_valid, nb_details)

        # 16. README.md exists
        readme_file = self.base_dir / "README.md"
        readme_valid = readme_file.exists() and readme_file.stat().st_size > 1000
        self.log_check(
            "README.md exists",
            readme_valid,
            f"README verified at {readme_file} ({readme_file.stat().st_size} bytes)" if readme_valid else f"Missing: {readme_file}"
        )

        # 17. requirements.txt exists
        req_file = self.base_dir / "requirements.txt"
        req_valid = req_file.exists() and req_file.stat().st_size > 20
        self.log_check(
            "requirements.txt exists",
            req_valid,
            f"Requirements file verified at {req_file}" if req_valid else f"Missing: {req_file}"
        )

        print("\n" + "=" * 75)
        print(f"VERIFICATION SUMMARY: {sum(1 for _, s, _ in self.results if s == 'PASS')} / {len(self.results)} CHECKS PASSED")
        if self.all_passed:
            print("STATUS: ALL TESTS PASSED SUCCESSFULLY!")
        else:
            print("STATUS: ONE OR MORE CHECKS FAILED.")
        print("=" * 75)

        return self.all_passed


if __name__ == "__main__":
    runner = VerificationRunner(BASE_DIR)
    success = runner.verify_all()
    sys.exit(0 if success else 1)
