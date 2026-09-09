"""
FastAPI application for AI Customer Insights, Segmentation & Recommendations.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from .schemas import (
    CustomerRFMInput,
    CustomerPredictionResponse,
    CustomerRecommendationsResponse,
    ProductRecommendationItem,
    SegmentSummary,
    HealthResponse,
    PredictCLVRequest,
    PredictCLVResponse,
    RecommendationsRequest,
    RecommendationsAPIResponse
)
from ..models.pipeline import CustomerInsightsPipeline
from ..models.persona import PersonaEngine
from contextlib import asynccontextmanager

pipeline = CustomerInsightsPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes and loads ML models on service start.
    If no artifacts exist, runs pipeline once.
    """
    loaded = pipeline.load_artifacts()
    if not loaded:
        print("Model artifacts not found. Bootstrapping training pipeline...")
        pipeline.run_training_pipeline()
    yield


app = FastAPI(
    title="AI Customer Insights & Recommender API",
    description="Production-grade API for RFM Segmentation, CLV Prediction, and Personalized Recommendations (RecSysDatasets / Amazon Reviews benchmark).",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Health check and model status inspection.
    """
    is_ready = (
        pipeline.cluster_model is not None and
        pipeline.clv_model is not None and
        pipeline.recommender is not None
    )
    count = len(pipeline.customer_profiles_) if pipeline.customer_profiles_ is not None else 0

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        models_loaded=is_ready,
        num_customers_indexed=count,
        version=__version__
    )


@app.get("/api/v1/segments", response_model=List[SegmentSummary], tags=["Segmentation"])
def get_segments():
    """
    Returns summary statistics for all customer clusters.
    """
    if pipeline.cluster_model is None or pipeline.cluster_model.cluster_profiles is None:
        raise HTTPException(status_code=503, detail="Cluster model is not yet loaded.")

    profiles = pipeline.cluster_model.cluster_profiles.to_dict(orient="records")
    return profiles


@app.post("/api/v1/predict/profile", response_model=CustomerPredictionResponse, tags=["Inference"])
def predict_customer_profile(data: CustomerRFMInput):
    """
    Calculates cluster, persona, CLV projection, churn risk, and marketing actions for raw customer metrics.
    """
    if pipeline.cluster_model is None or pipeline.clv_model is None:
        raise HTTPException(status_code=503, detail="Models are not yet loaded.")

    # 1. Cluster & PCA
    cluster_res = pipeline.cluster_model.predict_single(
        recency=data.recency,
        frequency=data.frequency,
        monetary=data.monetary,
        avg_rating=data.avg_rating
    )
    cluster_id = cluster_res["cluster"]

    # 2. Tenure
    tenure = data.customer_age_T or (data.recency + (data.frequency * 30.0))

    # 3. CLV & Churn
    clv_res = pipeline.clv_model.predict_single(
        recency=data.recency,
        frequency=data.frequency,
        monetary=data.monetary,
        customer_age_T=tenure
    )

    # 4. Persona Playbook
    persona_res = PersonaEngine.assign_persona(
        recency=data.recency,
        frequency=data.frequency,
        monetary=data.monetary,
        avg_rating=data.avg_rating,
        cluster=cluster_id
    )

    return CustomerPredictionResponse(
        cluster=cluster_id,
        persona=persona_res["persona_name"],
        priority_level=persona_res["priority_level"],
        marketing_strategy=persona_res["marketing_strategy"],
        preferred_channel=persona_res["preferred_channel"],
        discount_incentive=persona_res["discount_incentive"],
        projected_clv_1y=clv_res["projected_clv_1y"],
        churn_probability=clv_res["churn_probability"],
        churn_risk_level=clv_res["churn_risk_level"],
        pca_coordinates=cluster_res["pca_coordinates"]
    )


@app.get("/api/v1/customer/{customer_id}", tags=["Customer Management"])
def get_customer_details(customer_id: str):
    """
    Retrieves full calculated profile for an existing customer in the database.
    """
    if pipeline.customer_profiles_ is None:
        raise HTTPException(status_code=503, detail="Customer profiles database is not loaded.")

    matches = pipeline.customer_profiles_[pipeline.customer_profiles_["reviewerID"] == customer_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found.")

    record = matches.iloc[0].to_dict()
    # Convert numpy types to native python for JSON serialization
    clean_record = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
    return clean_record


@app.get(
    "/api/v1/customer/{customer_id}/recommendations",
    response_model=CustomerRecommendationsResponse,
    tags=["Recommendations"]
)
def get_recommendations(
    customer_id: str,
    top_n: int = Query(default=5, ge=1, le=20)
):
    """
    Returns personalized product recommendations for a customer using SVD collaborative filtering.
    """
    if pipeline.recommender is None:
        raise HTTPException(status_code=503, detail="Recommender model is not yet loaded.")

    recs = pipeline.recommender.recommend(reviewer_id=customer_id, n=top_n)
    
    return CustomerRecommendationsResponse(
        reviewer_id=customer_id,
        recommendations=[ProductRecommendationItem(**r) for r in recs]
    )


@app.post("/api/predict_clv", response_model=PredictCLVResponse, tags=["Inference"])
def api_predict_clv(request: PredictCLVRequest):
    """
    POST /api/predict_clv: Receives user_id (with optional RFM overrides),
    returns CLV_pred, assigned cluster segment, 3D PCA coordinates, and centroid distance.
    """
    if pipeline.cluster_model is None or pipeline.clv_model is None:
        raise HTTPException(status_code=503, detail="Models are not yet loaded.")

    user_id = request.user_id
    recency = request.recency
    frequency = request.frequency
    monetary = request.monetary
    avg_rating = request.avg_rating
    customer_age_T = None

    # Check if user exists in database if fields are omitted
    if pipeline.customer_profiles_ is not None and (recency is None or frequency is None or monetary is None):
        user_matches = pipeline.customer_profiles_[pipeline.customer_profiles_["reviewerID"] == user_id]
        if not user_matches.empty:
            row = user_matches.iloc[0]
            if recency is None:
                recency = float(row.get("recency", 30.0))
            if frequency is None:
                frequency = float(row.get("frequency", 2.0))
            if monetary is None:
                monetary = float(row.get("monetary", 50.0))
            if avg_rating is None:
                avg_rating = float(row.get("avg_rating", 4.0))
            customer_age_T = float(row.get("customer_age_T", recency + (frequency * 30.0)))

    # Fallback default values for completely cold/unseen customers
    recency = 30.0 if recency is None else float(recency)
    frequency = 1.0 if frequency is None else float(frequency)
    monetary = 35.0 if monetary is None else float(monetary)
    avg_rating = 4.0 if avg_rating is None else float(avg_rating)
    if customer_age_T is None:
        customer_age_T = recency + (frequency * 30.0)

    # 1. Cluster & 3D PCA
    cluster_res = pipeline.cluster_model.predict_single(
        recency=recency,
        frequency=frequency,
        monetary=monetary,
        avg_rating=avg_rating
    )
    cluster_id = cluster_res["cluster"]

    # 2. Predictive CLV (Poisson GLM + BTYD)
    clv_res = pipeline.clv_model.predict_single(
        recency=recency,
        frequency=frequency,
        monetary=monetary,
        customer_age_T=customer_age_T
    )

    # 3. Persona Playbook
    persona_res = PersonaEngine.assign_persona(
        recency=recency,
        frequency=frequency,
        monetary=monetary,
        avg_rating=avg_rating,
        cluster=cluster_id
    )

    return PredictCLVResponse(
        user_id=user_id,
        clv_pred=clv_res.get("clv_pred", clv_res.get("projected_clv_1y", 0.0)),
        segment=cluster_id,
        segment_name=persona_res["persona_name"],
        persona=persona_res["persona_name"],
        pca_coordinates=cluster_res["pca_coordinates"],
        centroid_distance=cluster_res.get("centroid_distance", 0.0),
        churn_probability=clv_res["churn_probability"],
        marketing_strategy=persona_res["marketing_strategy"]
    )


@app.post("/api/recommendations", response_model=RecommendationsAPIResponse, tags=["Recommendations"])
def api_recommendations(request: RecommendationsRequest):
    """
    POST /api/recommendations: Receives user_id, returns paginated top-N items
    recommended by the hybrid engine (SVD + Content) along with provenance sources.
    """
    if pipeline.recommender is None:
        raise HTTPException(status_code=503, detail="Recommender model is not yet loaded.")

    # Generate recommendations from hybrid engine
    all_recs = pipeline.recommender.recommend(
        reviewer_id=request.user_id,
        n=request.top_n,
        alpha=request.alpha
    )

    # Paginate
    page = request.page
    page_size = request.page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = all_recs[start_idx:end_idx]

    return RecommendationsAPIResponse(
        user_id=request.user_id,
        page=page,
        page_size=page_size,
        total_recommendations=len(all_recs),
        recommendations=[ProductRecommendationItem(**item) for item in page_items]
    )
