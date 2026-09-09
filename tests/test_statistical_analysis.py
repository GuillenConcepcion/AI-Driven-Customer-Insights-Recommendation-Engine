"""
Unit tests for statistical analysis, hypothesis testing, and budget optimizer.
"""

import pytest
import numpy as np
import pandas as pd
from customer_insights.features.statistical_analysis import (
    compute_gini_coefficient,
    run_univariate_diagnostics,
    run_hypothesis_tests,
    optimize_retention_budget
)


@pytest.fixture
def sample_customer_df():
    data = []
    for i in range(100):
        data.append({
            "reviewerID": f"User_{i}",
            "recency": float((i * 7) % 360 + 5),
            "frequency": float((i % 10) + 1),
            "monetary": float(((i % 10) + 1) * 45.0),
            "avg_rating": float(3.0 + (i % 3) * 0.9),
            "cluster": int(i % 4),
            "persona": ["Champions (VIPs)", "Loyal Customers", "At-Risk Customers", "Potential Loyalists"][i % 4],
            "churn_probability": float((i % 10) / 10.0),
            "churn_risk_level": "Low" if (i % 10) < 3 else ("Medium" if (i % 10) < 7 else "High"),
            "projected_clv_1y": float(((i % 10) + 1) * 60.0)
        })
    return pd.DataFrame(data)


def test_compute_gini():
    # Perfectly equal distribution -> Gini 0
    equal = np.array([100.0, 100.0, 100.0, 100.0])
    assert compute_gini_coefficient(equal) == 0.0
    
    # Unequal distribution
    unequal = np.array([10.0, 20.0, 30.0, 1000.0])
    gini = compute_gini_coefficient(unequal)
    assert 0.0 < gini <= 1.0


def test_univariate_diagnostics(sample_customer_df):
    diag = run_univariate_diagnostics(sample_customer_df)
    assert "recency" in diag
    assert "monetary" in diag
    assert "gini_coefficient" in diag["monetary"]
    assert "skewness" in diag["monetary"]


def test_hypothesis_tests(sample_customer_df):
    tests = run_hypothesis_tests(sample_customer_df)
    assert "normality_test_monetary" in tests
    assert "kruskal_wallis_clusters_monetary" in tests
    assert "spearman_freq_vs_rating" in tests
    assert "chi_square_persona_vs_churn" in tests


def test_retention_budget_optimizer(sample_customer_df):
    opt = optimize_retention_budget(sample_customer_df, total_budget=500.0, intervention_cost=20.0)
    assert opt["budget_allocated"] == 500.0
    assert opt["number_of_customers_targeted"] <= 25
    assert opt["actual_campaign_cost"] <= 500.0
    assert "total_expected_net_clv_saved" in opt
    assert len(opt["top_targeted_customers"]) > 0
