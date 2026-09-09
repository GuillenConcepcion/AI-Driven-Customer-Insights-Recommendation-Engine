"""
End-to-End Pipeline Orchestrator for Customer Insights, Segmentation & Recommenders.
Manages data processing, model training, evaluation, and artifact serialization.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd

from ..config import MODELS_DIR, PROCESSED_DATA_DIR
from ..data.amazon_loader import AmazonDataLoader
from ..data.validation import validate_interaction_data
from ..features.rfm import calculate_rfm_metrics
from .clustering import CustomerClusterModel
from .clv import CustomerLifetimeValueModel
from .recommender import HybridRecommender
from .persona import PersonaEngine


class CustomerInsightsPipeline:
    """
    Orchestrates the entire ML lifecycle and serves trained components.
    """

    def __init__(self, models_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.data_dir = data_dir or PROCESSED_DATA_DIR
        
        self.cluster_model: Optional[CustomerClusterModel] = None
        self.clv_model: Optional[CustomerLifetimeValueModel] = None
        self.recommender: Optional[HybridRecommender] = None
        
        self.customer_profiles_: Optional[pd.DataFrame] = None
        self.products_df_: Optional[pd.DataFrame] = None

    def run_training_pipeline(
        self,
        category: Optional[str] = None,
        k_clusters: int = 4,
        save_artifacts: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete training and evaluation pipeline.
        """
        print("=== Step 1: Ingesting Data ===")
        loader = AmazonDataLoader()
        interactions_df, products_df = loader.load_or_fallback(category=category)
        self.products_df_ = products_df

        print(f"Loaded {len(interactions_df)} interactions and {len(products_df)} products.")
        
        # Validation
        is_valid, report = validate_interaction_data(interactions_df)
        if not is_valid:
            print(f"Data validation warnings: {report['warnings']}")

        print("=== Step 2: Computing RFM Features ===")
        rfm_df = calculate_rfm_metrics(interactions_df, products_df)
        print(f"Generated RFM metrics for {len(rfm_df)} customers.")

        print("=== Step 3: Training Clustering & Dimensionality Reduction ===")
        self.cluster_model = CustomerClusterModel(n_clusters=k_clusters)
        eval_metrics = self.cluster_model.evaluate_k_range(rfm_df, k_min=2, k_max=7)
        self.cluster_model.fit(rfm_df, n_clusters=k_clusters)
        clustered_df = self.cluster_model.transform(rfm_df)

        print("=== Step 4: Training Customer Lifetime Value Model ===")
        self.clv_model = CustomerLifetimeValueModel()
        self.clv_model.fit(clustered_df)
        clv_df = self.clv_model.predict_retention_and_clv(clustered_df)

        print("=== Step 5: Assigning Personas and Strategic Playbooks ===")
        self.customer_profiles_ = PersonaEngine.enrich_dataframe(clv_df)

        print("=== Step 6: Training Hybrid Recommender Engine ===")
        self.recommender = HybridRecommender()
        self.recommender.fit(interactions_df, products_df)

        if save_artifacts:
            print("=== Step 7: Persisting Artifacts ===")
            self.save_models_and_data(eval_metrics)

        print("Training pipeline completed successfully.")
        return {
            "num_customers": len(self.customer_profiles_),
            "num_products": len(self.products_df_),
            "clusters": self.cluster_model.cluster_profiles.to_dict(orient="records"),
            "eval_metrics": eval_metrics
        }

    def save_models_and_data(self, eval_metrics: Optional[Dict[str, Any]] = None):
        """
        Saves models and processed tables to disk.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.cluster_model, self.models_dir / "cluster_model.joblib")
        joblib.dump(self.clv_model, self.models_dir / "clv_model.joblib")
        joblib.dump(self.recommender, self.models_dir / "recommender.joblib")

        if self.customer_profiles_ is not None:
            self.customer_profiles_.to_parquet(self.data_dir / "customer_profiles.parquet", index=False)
            self.customer_profiles_.head(500).to_csv(self.data_dir / "customer_profiles_sample.csv", index=False)

        if self.products_df_ is not None:
            self.products_df_.to_parquet(self.data_dir / "products.parquet", index=False)

        if eval_metrics:
            with open(self.models_dir / "clustering_eval.json", "w") as f:
                json.dump(eval_metrics, f, indent=2)

    def load_artifacts(self) -> bool:
        """
        Loads pre-trained models and processed data into memory.
        """
        cluster_path = self.models_dir / "cluster_model.joblib"
        clv_path = self.models_dir / "clv_model.joblib"
        rec_path = self.models_dir / "recommender.joblib"
        profiles_path = self.data_dir / "customer_profiles.parquet"
        products_path = self.data_dir / "products.parquet"

        if not (cluster_path.exists() and clv_path.exists() and rec_path.exists() and profiles_path.exists()):
            return False

        self.cluster_model = joblib.load(cluster_path)
        self.clv_model = joblib.load(clv_path)
        self.recommender = joblib.load(rec_path)
        self.customer_profiles_ = pd.read_parquet(profiles_path)
        if products_path.exists():
            self.products_df_ = pd.read_parquet(products_path)

        return True


if __name__ == "__main__":
    pipe = CustomerInsightsPipeline()
    pipe.run_training_pipeline()
