"""
RFM (Recency, Frequency, Monetary) feature extraction and scoring.
Calculates customer-level behavioral metrics tailored for e-commerce and review platforms.
"""

from typing import Optional
import pandas as pd
import numpy as np


def calculate_rfm_metrics(
    interactions_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary, T (Customer Age), and Average Rating per customer.

    Args:
        interactions_df: DataFrame with ['reviewerID', 'asin', 'overall', 'unixReviewTime']
        products_df: Optional DataFrame with ['asin', 'price']
        reference_date: Optional cutoff date. Defaults to max(interaction_date) + 1 day.

    Returns:
        pd.DataFrame indexed by reviewerID with RFM features.
    """
    df = interactions_df.copy()

    # Convert timestamps if necessary
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["unixReviewTime"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Merge prices if available
    if products_df is not None and "price" in products_df.columns:
        df = df.merge(products_df[["asin", "price"]], on="asin", how="left")
        df["monetary_value"] = df["price"].fillna(29.99)
    else:
        # Fallback proxy based on rating and interaction
        df["monetary_value"] = 25.0 + (df["overall"] * 5.0)

    # Set snapshot reference date
    cutoff = reference_date if reference_date is not None else (df["timestamp"].max() + pd.Timedelta(days=1))

    # Aggregations per customer
    rfm = df.groupby("reviewerID").agg(
        first_interaction=("timestamp", "min"),
        last_interaction=("timestamp", "max"),
        frequency=("asin", "count"),
        monetary=("monetary_value", "sum"),
        avg_rating=("overall", "mean"),
        distinct_categories=("asin", "nunique")
    ).reset_index()

    # Calculate Recency and T (Customer Age in days)
    rfm["recency"] = (cutoff - rfm["last_interaction"]).dt.days
    rfm["customer_age_T"] = (cutoff - rfm["first_interaction"]).dt.days
    
    # Avoid negative or 0 days
    rfm["recency"] = rfm["recency"].clip(lower=1)
    rfm["customer_age_T"] = rfm["customer_age_T"].clip(lower=1)

    # Calculate Quartile / Quintile Scores (1 to 5)
    # Recency: Lower is better -> reverse bins
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    
    # Frequency & Monetary: Higher is better
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Combined RFM Score string and weighted numeric score
    rfm["rfm_segment"] = (
        rfm["r_score"].astype(str) + 
        rfm["f_score"].astype(str) + 
        rfm["m_score"].astype(str)
    )
    rfm["rfm_composite_score"] = (rfm["r_score"] * 0.40) + (rfm["f_score"] * 0.30) + (rfm["m_score"] * 0.30)

    return rfm
