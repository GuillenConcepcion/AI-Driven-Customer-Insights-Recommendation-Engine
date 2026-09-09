"""
Unit tests for data generation and validation.
"""

import pytest
import pandas as pd
from customer_insights.data.generator import generate_synthetic_dataset
from customer_insights.data.validation import validate_interaction_data


def test_synthetic_data_generator():
    interactions, products = generate_synthetic_dataset(
        num_customers=50,
        num_products=20,
        num_reviews=200,
        seed=123,
        save_to_disk=False
    )
    
    assert not interactions.empty
    assert not products.empty
    assert "reviewerID" in interactions.columns
    assert "asin" in interactions.columns
    assert "overall" in interactions.columns
    assert interactions["overall"].min() >= 1.0
    assert interactions["overall"].max() <= 5.0
    assert interactions["reviewerID"].nunique() == 50


def test_data_validation_valid():
    df = pd.DataFrame({
        "reviewerID": ["U1", "U2", "U3"],
        "asin": ["P1", "P2", "P3"],
        "overall": [5.0, 4.0, 3.5],
        "unixReviewTime": [1700000000, 1700100000, 1700200000]
    })
    is_valid, report = validate_interaction_data(df)
    assert is_valid is True
    assert report["invalid_ratings_count"] == 0


def test_data_validation_invalid_rating():
    df = pd.DataFrame({
        "reviewerID": ["U1"],
        "asin": ["P1"],
        "overall": [6.5],  # Invalid rating > 5.0
        "unixReviewTime": [1700000000]
    })
    is_valid, report = validate_interaction_data(df)
    assert is_valid is False
    assert report["invalid_ratings_count"] == 1
