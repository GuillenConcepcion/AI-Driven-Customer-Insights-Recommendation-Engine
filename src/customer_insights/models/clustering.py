"""
Customer Segmentation using Unsupervised Learning:
- K-Means with Elbow & Silhouette Score analysis
- PCA 2D/3D Dimensionality Reduction
- DBSCAN Outlier / Atypical Pattern Detection
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from ..config import clustering_config
from ..features.preprocessing import RFMFeaturePreprocessor


class CustomerClusterModel:
    """
    Unsupervised clustering model orchestrating K-Means, PCA, and DBSCAN.
    """

    def __init__(
        self,
        n_clusters: int = clustering_config.default_n_clusters,
        random_state: int = clustering_config.random_state
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.preprocessor = RFMFeaturePreprocessor()
        self.kmeans: Optional[KMeans] = None
        self.pca_2d: Optional[PCA] = None
        self.pca_3d: Optional[PCA] = None
        self.dbscan: Optional[DBSCAN] = None
        self.cluster_profiles: Optional[pd.DataFrame] = None
        self.evaluation_results: Dict[str, Any] = {}

    def evaluate_k_range(
        self,
        rfm_df: pd.DataFrame,
        k_min: int = 2,
        k_max: int = 8
    ) -> Dict[str, Any]:
        """
        Evaluates a range of k values using Inertia (Elbow) and Silhouette Score.
        """
        X_scaled = self.preprocessor.fit_transform(rfm_df)
        k_values = list(range(k_min, k_max + 1))
        inertias = []
        silhouettes = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(float(km.inertia_))
            sil = float(silhouette_score(X_scaled, labels))
            silhouettes.append(sil)

        best_k = k_values[int(np.argmax(silhouettes))]
        self.evaluation_results = {
            "k_values": k_values,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "best_k_by_silhouette": best_k
        }
        return self.evaluation_results

    def fit(self, rfm_df: pd.DataFrame, n_clusters: Optional[int] = None) -> "CustomerClusterModel":
        """
        Fits the preprocessor, K-Means, PCA (2D & 3D), and DBSCAN.

        Args:
            rfm_df (pd.DataFrame): Customer RFM dataframe with ['recency', 'frequency', 'monetary', 'avg_rating'].
            n_clusters (Optional[int], optional): Override for number of clusters k. Defaults to None.

        Returns:
            CustomerClusterModel: Fitted instance with calculated cluster profiles and 3D centroids.
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        X_scaled = self.preprocessor.fit_transform(rfm_df)

        # 1. K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = self.kmeans.fit_predict(X_scaled)

        # 2. PCA Projections (2D and 3D)
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        
        self.pca_2d.fit(X_scaled)
        self.pca_3d.fit(X_scaled)

        # Compute 3D cluster centroids in PCA space
        centroids_scaled = self.kmeans.cluster_centers_
        centroids_3d = self.pca_3d.transform(centroids_scaled)
        self.cluster_centroids_3d = {
            int(k): {
                "x": round(float(centroids_3d[k, 0]), 4),
                "y": round(float(centroids_3d[k, 1]), 4),
                "z": round(float(centroids_3d[k, 2]), 4)
            }
            for k in range(self.n_clusters)
        }

        # 3. DBSCAN for density clustering and anomaly detection
        self.dbscan = DBSCAN(
            eps=clustering_config.dbscan_eps,
            min_samples=clustering_config.dbscan_min_samples
        )
        dbscan_labels = self.dbscan.fit_predict(X_scaled)

        # Calculate Cluster Summary Profiles
        df_copy = rfm_df.copy()
        df_copy["cluster"] = labels
        df_copy["is_outlier"] = (dbscan_labels == -1)

        self.cluster_profiles = df_copy.groupby("cluster").agg(
            customer_count=("reviewerID", "count"),
            mean_recency=("recency", "mean"),
            median_recency=("recency", "median"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
            mean_rating=("avg_rating", "mean")
        ).round(2).reset_index()

        # Add percentage of total customers
        total = len(df_copy)
        self.cluster_profiles["percentage"] = ((self.cluster_profiles["customer_count"] / total) * 100).round(1)

        # Add 3D centroid coordinates to profiles
        self.cluster_profiles["centroid_x"] = [self.cluster_centroids_3d[c]["x"] for c in self.cluster_profiles["cluster"]]
        self.cluster_profiles["centroid_y"] = [self.cluster_centroids_3d[c]["y"] for c in self.cluster_profiles["cluster"]]
        self.cluster_profiles["centroid_z"] = [self.cluster_centroids_3d[c]["z"] for c in self.cluster_profiles["cluster"]]

        return self

    def transform(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns cluster labels, PCA coordinates (2D and 3D), distance to centroid, and anomaly flags.

        Args:
            rfm_df (pd.DataFrame): Customer RFM dataframe to project.

        Returns:
            pd.DataFrame: Dataframe with added ['cluster', 'pca_x', 'pca_y', 'pca_z', 'centroid_distance', 'is_outlier'].

        Raises:
            RuntimeError: If called before model is fitted.
        """
        if self.kmeans is None or self.pca_2d is None or self.pca_3d is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_scaled = self.preprocessor.transform(rfm_df)
        labels = self.kmeans.predict(X_scaled)
        pca_2d_coords = self.pca_2d.transform(X_scaled)
        pca_3d_coords = self.pca_3d.transform(X_scaled)

        result_df = rfm_df.copy()
        result_df["cluster"] = labels
        result_df["pca_x"] = pca_3d_coords[:, 0].round(4)
        result_df["pca_y"] = pca_3d_coords[:, 1].round(4)
        result_df["pca_z"] = pca_3d_coords[:, 2].round(4)

        if self.dbscan is not None:
            # Predict outlier via DBSCAN fit on full dataset or nearest core
            result_df["is_outlier"] = (self.dbscan.fit_predict(X_scaled) == -1)

        # Calculate distance to cluster centroid in 3D PCA space
        distances = []
        for i, c in enumerate(labels):
            c_dict = self.cluster_centroids_3d.get(int(c), {"x": 0.0, "y": 0.0, "z": 0.0})
            c_vec = np.array([c_dict["x"], c_dict["y"], c_dict["z"]])
            pt_vec = pca_3d_coords[i, :]
            distances.append(float(np.linalg.norm(pt_vec - c_vec)))
        result_df["centroid_distance"] = np.round(distances, 3)

        return result_df

    def predict_single(self, recency: float, frequency: float, monetary: float, avg_rating: float) -> Dict[str, Any]:
        """
        Inference for a single customer profile (used by FastAPI endpoint).
        Returns cluster, 3D coordinates, and distance to cluster centroid.

        Args:
            recency (float): Days since last observed interaction.
            frequency (float): Number of purchases in observation window.
            monetary (float): Total historical transaction spend in USD.
            avg_rating (float): Mean product review rating [1.0, 5.0].

        Returns:
            Dict[str, Any]: Inference dictionary containing:
                - cluster (int): Cluster archetype assignment.
                - pca_coordinates (List[float]): 3D orthogonal position [PC1, PC2, PC3].
                - centroid_coordinates (Dict[str, float]): Centroid position {'x', 'y', 'z'}.
                - centroid_distance (float): Euclidean distance to assigned centroid.
        """
        sample_df = pd.DataFrame([{
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_rating": avg_rating
        }])
        X_scaled = self.preprocessor.transform(sample_df)
        cluster = int(self.kmeans.predict(X_scaled)[0])
        pca_3d_coords = self.pca_3d.transform(X_scaled)[0]

        centroid = self.cluster_centroids_3d.get(cluster, {"x": 0.0, "y": 0.0, "z": 0.0})
        c_vec = np.array([centroid["x"], centroid["y"], centroid["z"]])
        dist = float(np.linalg.norm(pca_3d_coords - c_vec))

        return {
            "cluster": cluster,
            "pca_coordinates": [float(round(pca_3d_coords[0], 4)), float(round(pca_3d_coords[1], 4)), float(round(pca_3d_coords[2], 4))],
            "centroid_coordinates": centroid,
            "centroid_distance": round(dist, 4)
        }

    def compare_kmeans_dbscan(self, rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Rigorous Senior Evaluation: Compares K-Means partitions against DBSCAN dense structures
        and identifies anomalies/outliers that K-Means forcibly assigned into spherical clusters.

        Args:
            rfm_df (pd.DataFrame): Customer RFM dataframe for density comparison.

        Returns:
            Dict[str, Any]: Detailed comparison dictionary containing:
                - total_customers (int)
                - total_outliers (int)
                - outlier_percentage (float)
                - unique_dbscan_clusters (int)
                - cluster_breakdown (List[Dict]): Per-cluster outlier proportions and noise rates.

        Raises:
            RuntimeError: If called before model is fitted.
        """
        if self.kmeans is None or self.dbscan is None:
            raise RuntimeError("Models are not fitted. Call fit() first.")

        X_scaled = self.preprocessor.transform(rfm_df)
        km_labels = self.kmeans.predict(X_scaled)
        db_labels = self.dbscan.fit_predict(X_scaled)

        eval_df = pd.DataFrame({
            "kmeans_cluster": km_labels,
            "dbscan_cluster": db_labels,
            "is_dbscan_outlier": (db_labels == -1)
        })

        total = len(eval_df)
        total_outliers = int((db_labels == -1).sum())
        outlier_pct = round((total_outliers / total) * 100, 2)
        unique_db_clusters = len(set(db_labels) - {-1})

        # Breakdown per K-Means cluster
        cluster_breakdown = []
        for c in range(self.n_clusters):
            c_mask = (eval_df["kmeans_cluster"] == c)
            c_total = int(c_mask.sum())
            c_outliers = int((c_mask & eval_df["is_dbscan_outlier"]).sum())
            c_outlier_pct = round((c_outliers / c_total * 100) if c_total > 0 else 0.0, 2)
            c_core_pct = round(100.0 - c_outlier_pct, 2)
            
            verdict = "Dense Homogeneous Core" if c_outlier_pct < 5.0 else (
                "Moderate Dispersion" if c_outlier_pct < 15.0 else "High Noise / Anomalous Outliers Flagged"
            )

            cluster_breakdown.append({
                "kmeans_cluster": c,
                "total_customers": c_total,
                "dbscan_outliers": c_outliers,
                "outlier_percentage": c_outlier_pct,
                "core_density_percentage": c_core_pct,
                "evaluation_verdict": verdict
            })

        return {
            "total_customers": total,
            "kmeans_n_clusters": self.n_clusters,
            "dbscan_natural_clusters": unique_db_clusters,
            "total_dbscan_outliers": total_outliers,
            "global_outlier_percentage": outlier_pct,
            "cluster_breakdown": cluster_breakdown
        }
