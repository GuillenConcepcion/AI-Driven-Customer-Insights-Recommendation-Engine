"""
Central configuration for AI Customer Insights & Recommendations Engine.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    random_seed: int = 42
    num_customers: int = 3000
    num_products: int = 250
    num_reviews: int = 25000
    date_start: str = "2024-01-01"
    date_end: str = "2025-12-31"
    categories: List[str] = field(default_factory=lambda: [
        "Electronics",
        "Home & Kitchen",
        "Video Games",
        "Books",
        "Computers & Accessories",
        "Musical Instruments",
        "Health & Personal Care"
    ])


@dataclass
class ClusteringConfig:
    k_range: tuple = (2, 9)
    default_n_clusters: int = 4
    random_state: int = 42
    pca_n_components: int = 3
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 5


@dataclass
class RecommenderConfig:
    n_factors: int = 20
    top_n: int = 5
    min_interactions_user: int = 2
    min_interactions_item: int = 2
    random_state: int = 42


@dataclass
class AppConfig:
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501


data_config = DataConfig()
clustering_config = ClusteringConfig()
recommender_config = RecommenderConfig()
app_config = AppConfig()
