"""
Unit tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from customer_insights.api.main import app, pipeline


@pytest.fixture(scope="module")
def client():
    # Ensure pipeline has artifacts ready for test
    if pipeline.customer_profiles_ is None:
        pipeline.run_training_pipeline()
    
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_api_segments(client):
    response = client.get("/api/v1/segments")
    assert response.status_code == 200
    segments = response.json()
    assert isinstance(segments, list)
    assert len(segments) > 0
    assert "cluster" in segments[0]


def test_api_predict_profile(client):
    payload = {
        "recency": 25.0,
        "frequency": 6.0,
        "monetary": 350.0,
        "avg_rating": 4.8
    }
    response = client.post("/api/v1/predict/profile", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "cluster" in res
    assert "persona" in res
    assert "projected_clv_1y" in res
    assert "marketing_strategy" in res


def test_api_recommendations(client):
    # Pick a customer from profiles
    customer_id = pipeline.customer_profiles_["reviewerID"].iloc[0]
    response = client.get(f"/api/v1/customer/{customer_id}/recommendations?top_n=3")
    assert response.status_code == 200
    res = response.json()
    assert res["reviewer_id"] == customer_id
    assert len(res["recommendations"]) <= 3


def test_post_predict_clv(client):
    customer_id = pipeline.customer_profiles_["reviewerID"].iloc[0]
    payload = {"user_id": customer_id}
    response = client.post("/api/predict_clv", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == customer_id
    assert "clv_pred" in data
    assert "segment" in data
    assert "pca_coordinates" in data
    assert len(data["pca_coordinates"]) == 3
    assert "centroid_distance" in data
    assert data["clv_pred"] > 0


def test_post_recommendations(client):
    customer_id = pipeline.customer_profiles_["reviewerID"].iloc[0]
    payload = {
        "user_id": customer_id,
        "top_n": 10,
        "alpha": 0.65,
        "page": 1,
        "page_size": 10
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == customer_id
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 10
    if len(data["recommendations"]) > 0:
        first_item = data["recommendations"][0]
        assert "asin" in first_item
        assert "score" in first_item
        assert "source" in first_item
        assert "recommendation_type" in first_item

