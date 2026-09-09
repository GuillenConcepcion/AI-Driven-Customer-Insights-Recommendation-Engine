"""
Pydantic v2 schemas for API requests, responses, and validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerRFMInput(BaseModel):
    recency: float = Field(..., ge=0, description="Days since last purchase/interaction")
    frequency: float = Field(..., ge=1, description="Total number of distinct purchases")
    monetary: float = Field(..., ge=0, description="Total cumulative monetary spend")
    avg_rating: float = Field(default=4.0, ge=1.0, le=5.0, description="Average review rating (1.0 to 5.0)")
    customer_age_T: Optional[float] = Field(default=None, ge=1, description="Customer tenure in days since first purchase")


class CustomerPredictionResponse(BaseModel):
    cluster: int
    persona: str
    priority_level: str
    marketing_strategy: str
    preferred_channel: str
    discount_incentive: str
    projected_clv_1y: float
    churn_probability: float
    churn_risk_level: str
    pca_coordinates: List[float]


class PredictCLVRequest(BaseModel):
    user_id: str = Field(..., description="Unique customer/reviewer identifier")
    recency: Optional[float] = Field(default=None, description="Optional override: days since last interaction")
    frequency: Optional[float] = Field(default=None, description="Optional override: total order count")
    monetary: Optional[float] = Field(default=None, description="Optional override: cumulative spend")
    avg_rating: Optional[float] = Field(default=None, description="Optional override: avg rating (1.0 - 5.0)")


class PredictCLVResponse(BaseModel):
    user_id: str
    clv_pred: float = Field(..., description="Projected forward 12-month CLV ($)")
    segment: int = Field(..., description="Assigned K-Means cluster ID")
    segment_name: str = Field(..., description="Descriptive Persona name for the segment")
    persona: str
    pca_coordinates: List[float] = Field(..., description="3D coordinates in PCA space [x, y, z]")
    centroid_distance: float = Field(..., description="Euclidean distance to cluster centroid in 3D PCA space")
    churn_probability: float
    marketing_strategy: str


class RecommendationsRequest(BaseModel):
    user_id: str = Field(..., description="Customer/reviewer identifier to generate recommendations for")
    top_n: int = Field(default=10, ge=1, le=50, description="Total number of items to recommend")
    alpha: float = Field(default=0.65, ge=0.0, le=1.0, description="Hybrid blend weight: alpha * SVD + (1 - alpha) * Content")
    page: int = Field(default=1, ge=1, description="Page number for pagination")
    page_size: int = Field(default=10, ge=1, le=50, description="Page size for pagination")


class ProductRecommendationItem(BaseModel):
    asin: str
    title: str
    category: str
    brand: str
    price: float
    score: float
    recommendation_type: str
    source: Optional[str] = Field(default="Hybrid", description="Recommendation engine source (SVD, Content, Hybrid, Popularity)")


class CustomerRecommendationsResponse(BaseModel):
    reviewer_id: str
    recommendations: List[ProductRecommendationItem]


class RecommendationsAPIResponse(BaseModel):
    user_id: str
    page: int
    page_size: int
    total_recommendations: int
    recommendations: List[ProductRecommendationItem]


class SegmentSummary(BaseModel):
    cluster: int
    customer_count: int
    percentage: float
    mean_recency: float
    mean_frequency: float
    mean_monetary: float
    mean_rating: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    num_customers_indexed: int
    version: str
