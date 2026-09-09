"""
Unit tests for RFM calculations and preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from customer_insights.features.rfm import calculate_rfm_metrics
from customer_insights.features.preprocessing import RFMFeaturePreprocessor


@pytest.fixture
def sample_interactions():
    return pd.DataFrame({
        "reviewerID": ["U1", "U1", "U2", "U3", "U3", "U3"],
        "asin": ["P1", "P2", "P1", "P2", "P3", "P4"],
        "overall": [5.0, 4.0, 3.0, 5.0, 5.0, 4.0],
        "unixReviewTime": [
            1704067200,  # 2024-01-01
            1704153600,  # 2024-01-02
            1706745600,  # 2024-02-01
            1709251200,  # 2024-03-01
            1711929600,  # 2024-04-01
            1714521600   # 2024-05-01
        ]
    })


def test_calculate_rfm_metrics(sample_interactions):
    rfm = calculate_rfm_metrics(sample_interactions)
    
    assert len(rfm) == 3
    assert set(rfm["reviewerID"]) == {"U1", "U2", "U3"}
    assert "recency" in rfm.columns
    assert "frequency" in rfm.columns
    assert "monetary" in rfm.columns
    assert "r_score" in rfm.columns
    
    # Check frequencies
    u1 = rfm[rfm["reviewerID"] == "U1"].iloc[0]
    u3 = rfm[rfm["reviewerID"] == "U3"].iloc[0]
    assert u1["frequency"] == 2
    assert u3["frequency"] == 3


def test_rfm_preprocessor(sample_interactions):
    rfm = calculate_rfm_metrics(sample_interactions)
    preprocessor = RFMFeaturePreprocessor()
    X_scaled = preprocessor.fit_transform(rfm)
    
    assert X_scaled.shape == (3, 4)
    # Check that scaled mean is approx 0
    assert np.allclose(X_scaled.mean(axis=0), [0, 0, 0, 0], atol=1e-7)
