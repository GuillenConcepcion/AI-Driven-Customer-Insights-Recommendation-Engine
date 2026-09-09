"""
Data validation and sanity checks for customer interactions and catalogs.
"""

from typing import Dict, Any, Tuple
import pandas as pd


REQUIRED_INTERACTION_COLUMNS = ["reviewerID", "asin", "overall", "unixReviewTime"]


def validate_interaction_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates interaction DataFrame integrity according to CRISP-DM / data quality standards.
    """
    report = {
        "is_valid": True,
        "total_records": len(df),
        "unique_users": 0,
        "unique_items": 0,
        "null_counts": {},
        "invalid_ratings_count": 0,
        "warnings": []
    }

    # Column existence
    missing_cols = [c for c in REQUIRED_INTERACTION_COLUMNS if c not in df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["warnings"].append(f"Missing required columns: {missing_cols}")
        return False, report

    # Nulls
    null_counts = df[REQUIRED_INTERACTION_COLUMNS].isnull().sum().to_dict()
    report["null_counts"] = null_counts
    if any(count > 0 for count in null_counts.values()):
        report["is_valid"] = False
        report["warnings"].append("Critical columns contain NULL values.")

    # Rating boundary check
    invalid_ratings = df[(df["overall"] < 1.0) | (df["overall"] > 5.0)]
    report["invalid_ratings_count"] = len(invalid_ratings)
    if len(invalid_ratings) > 0:
        report["is_valid"] = False
        report["warnings"].append(f"Found {len(invalid_ratings)} ratings outside [1.0, 5.0] range.")

    report["unique_users"] = int(df["reviewerID"].nunique())
    report["unique_items"] = int(df["asin"].nunique())

    return report["is_valid"], report
