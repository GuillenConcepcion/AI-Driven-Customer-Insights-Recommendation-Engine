"""
Feature engineering module for RFM metrics and data preprocessing.
"""

from .rfm import calculate_rfm_metrics
from .preprocessing import RFMFeaturePreprocessor

__all__ = [
    "calculate_rfm_metrics",
    "RFMFeaturePreprocessor"
]
