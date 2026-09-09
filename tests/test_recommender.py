"""
Unit tests for Hybrid Recommender Engine.
"""

import pytest
import pandas as pd
from customer_insights.models.recommender import HybridRecommender


@pytest.fixture
def dummy_recsys_data():
    interactions = pd.DataFrame({
        "reviewerID": ["U1", "U1", "U1", "U2", "U2", "U3", "U3", "U4"],
        "asin": ["P1", "P2", "P5", "P2", "P3", "P1", "P3", "P4"],
        "overall": [5.0, 4.0, 4.5, 4.5, 3.0, 5.0, 4.0, 5.0]
    })
    products = pd.DataFrame({
        "asin": ["P1", "P2", "P3", "P4", "P5"],
        "title": ["Wireless Action Camera 4K", "Noise Cancelling Headphones", "Python Data Science Book", "Clean Code Book", "Smart Home Speaker"],
        "category": ["Electronics", "Electronics", "Books", "Books", "Audio"],
        "brand": ["BrandA", "BrandA", "BrandB", "BrandB", "BrandC"],
        "price": [19.99, 29.99, 14.99, 9.99, 49.99]
    })
    return interactions, products


def test_hybrid_recommender_fit_and_recommend(dummy_recsys_data):
    interactions, products = dummy_recsys_data
    rec = HybridRecommender(n_factors=2, top_n=2)
    rec.fit(interactions, products)
    
    # 1. Existing user with >= 3 interactions (Hybrid blend)
    recs_u1 = rec.recommend("U1", n=2, alpha=0.5, filter_purchased=True)
    assert len(recs_u1) > 0
    # Make sure P1, P2, P5 (already purchased) are not returned
    recommended_asins = [r["asin"] for r in recs_u1]
    assert "P1" not in recommended_asins
    assert "P2" not in recommended_asins
    assert "P5" not in recommended_asins
    assert "source" in recs_u1[0]

    # 2. Cold-start user (< 3 interactions or unknown user)
    cold_recs = rec.recommend("Unknown_User_999", n=2)
    assert len(cold_recs) == 2
    assert "Best-Sellers" in cold_recs[0]["recommendation_type"]
    assert cold_recs[0]["source"] == "Popularity Fallback"

