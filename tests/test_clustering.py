"""
Unit tests for Clustering, CLV, and Persona models.
"""

import pytest
import pandas as pd
from customer_insights.features.rfm import calculate_rfm_metrics
from customer_insights.models.clustering import CustomerClusterModel
from customer_insights.models.clv import CustomerLifetimeValueModel
from customer_insights.models.persona import PersonaEngine


@pytest.fixture
def sample_rfm_dataset():
    # Synthetic RFM data for 20 customers
    data = []
    for i in range(1, 21):
        data.append({
            "reviewerID": f"User_{i}",
            "recency": (i * 15) % 300 + 5,
            "frequency": (i % 8) + 1,
            "monetary": ((i % 8) + 1) * 35.0,
            "avg_rating": 3.0 + (i % 3) * 0.8,
            "customer_age_T": 365
        })
    return pd.DataFrame(data)


def test_customer_clustering_fit_transform(sample_rfm_dataset):
    model = CustomerClusterModel(n_clusters=3)
    model.fit(sample_rfm_dataset)
    
    transformed = model.transform(sample_rfm_dataset)
    assert "cluster" in transformed.columns
    assert "pca_x" in transformed.columns
    assert "pca_y" in transformed.columns
    assert "pca_z" in transformed.columns
    assert transformed["cluster"].nunique() <= 3
    assert len(transformed) == len(sample_rfm_dataset)


def test_clustering_predict_single(sample_rfm_dataset):
    model = CustomerClusterModel(n_clusters=3)
    model.fit(sample_rfm_dataset)
    
    res = model.predict_single(recency=25.0, frequency=5.0, monetary=180.0, avg_rating=4.5)
    assert "cluster" in res
    assert "pca_coordinates" in res
    assert len(res["pca_coordinates"]) == 3
    assert "centroid_distance" in res
    assert res["centroid_distance"] >= 0.0


def test_clustering_compare_kmeans_dbscan(sample_rfm_dataset):
    model = CustomerClusterModel(n_clusters=3)
    model.fit(sample_rfm_dataset)
    
    comp = model.compare_kmeans_dbscan(sample_rfm_dataset)
    assert "kmeans_n_clusters" in comp
    assert "total_dbscan_outliers" in comp
    assert "cluster_breakdown" in comp
    assert len(comp["cluster_breakdown"]) == 3


def test_clv_model(sample_rfm_dataset):
    clv_model = CustomerLifetimeValueModel()
    clv_model.fit(sample_rfm_dataset)
    
    predictions = clv_model.predict_retention_and_clv(sample_rfm_dataset)
    assert "p_active" in predictions.columns
    assert "projected_clv_1y" in predictions.columns
    assert "clv_pred" in predictions.columns
    assert "churn_risk_level" in predictions.columns
    assert (predictions["p_active"] >= 0.0).all() and (predictions["p_active"] <= 1.0).all()
    assert (predictions["clv_pred"] >= 0.0).all()


def test_persona_engine(sample_rfm_dataset):
    enriched = PersonaEngine.enrich_dataframe(sample_rfm_dataset)
    assert "persona" in enriched.columns
    assert "priority_level" in enriched.columns
    assert "marketing_strategy" in enriched.columns
    assert not enriched["persona"].isnull().any()
