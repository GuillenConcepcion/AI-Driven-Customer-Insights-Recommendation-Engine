"""
Data module for AI Customer Insights & Recommendations.
"""

from .generator import generate_synthetic_dataset
from .amazon_loader import AmazonDataLoader
from .validation import validate_interaction_data

__all__ = [
    "generate_synthetic_dataset",
    "AmazonDataLoader",
    "validate_interaction_data"
]
