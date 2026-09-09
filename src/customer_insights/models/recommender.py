"""
Hybrid Recommendation Engine for E-Commerce:
- Matrix Factorization (TruncatedSVD) Collaborative Filtering
- NLP Content-Based Filtering (TF-IDF + Cosine Similarity over metadata)
- Weighted Hybrid Score Integration: Score_Final = alpha * Score_SVD + (1 - alpha) * Score_Content
- Cold-Start Policies:
    * New customer (< 3 reviews): Popularity / Curated Bestseller Fallback
    * New item (unrated in SVD): Pure Content-Based Filtering
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..config import recommender_config


class HybridRecommender:
    """
    Production-grade hybrid recommender combining Collaborative SVD Matrix Factorization
    and TF-IDF Content-Based filtering with linear alpha blending.
    """

    def __init__(
        self,
        n_factors: int = recommender_config.n_factors,
        top_n: int = recommender_config.top_n,
        default_alpha: float = 0.65,
        random_state: int = recommender_config.random_state
    ):
        self.n_factors = n_factors
        self.top_n = top_n
        self.default_alpha = default_alpha
        self.random_state = random_state
        
        # Collaborative filtering attributes
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        
        self.svd: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        
        # Content-Based NLP attributes
        self.tfidf: Optional[TfidfVectorizer] = None
        self.item_tfidf_matrix: Optional[Any] = None
        self.asin_to_tfidf_idx: Dict[str, int] = {}
        
        # User history and data catalog
        self.products_df: Optional[pd.DataFrame] = None
        self.user_history: Dict[str, Set[str]] = {}
        self.user_ratings: Dict[str, Dict[str, float]] = {}
        self.popular_items_cache: List[Dict[str, Any]] = []

    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame) -> "HybridRecommender":
        """
        Fits both Collaborative Filtering (SVD) and Content-Based (TF-IDF) models.

        Args:
            interactions_df (pd.DataFrame): User interaction logs with columns ['reviewerID', 'asin', 'overall'].
            products_df (pd.DataFrame): Catalog items with columns ['asin', 'title', 'category', 'brand', 'price'].

        Returns:
            HybridRecommender: The fitted recommender instance with factor matrices and TF-IDF vocabulary.
        """
        self.products_df = products_df.copy().reset_index(drop=True)
        df = interactions_df.copy()

        # Build index mappings for collaborative filtering
        unique_users = df["reviewerID"].unique()
        unique_items = df["asin"].unique()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}

        # User history and ratings tracking
        self.user_history = df.groupby("reviewerID")["asin"].apply(set).to_dict()
        user_grouped = df.groupby("reviewerID")
        self.user_ratings = {
            uid: dict(zip(group["asin"], group["overall"]))
            for uid, group in user_grouped
        }

        # Build sparse user-item interaction matrix
        row_indices = df["reviewerID"].map(self.user_to_idx).values
        col_indices = df["asin"].map(self.item_to_idx).values
        ratings = df["overall"].values

        n_users = len(unique_users)
        n_items = len(unique_items)
        sparse_matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(n_users, n_items))

        # Fit SVD factors
        actual_factors = min(self.n_factors, min(n_users, n_items) - 1)
        actual_factors = max(actual_factors, 2)
        
        self.svd = TruncatedSVD(n_components=actual_factors, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(sparse_matrix)
        self.item_factors = self.svd.components_.T

        # --- Content-Based NLP Vectorization (TF-IDF) ---
        # Concatenate textual metadata: title, category, brand, and description
        text_corpus = (
            self.products_df["title"].fillna("") + " " +
            self.products_df["category"].fillna("") + " " +
            self.products_df["brand"].fillna("")
        )
        if "description" in self.products_df.columns:
            text_corpus = text_corpus + " " + self.products_df["description"].fillna("")

        self.tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        self.item_tfidf_matrix = self.tfidf.fit_transform(text_corpus)
        self.asin_to_tfidf_idx = {asin: idx for idx, asin in enumerate(self.products_df["asin"])}

        # Popularity fallback cache
        popularity = df.groupby("asin").agg(
            review_count=("overall", "count"),
            avg_rating=("overall", "mean")
        ).reset_index()
        
        popularity = popularity.merge(self.products_df, on="asin", how="left")
        popularity["score"] = popularity["avg_rating"] * np.log1p(popularity["review_count"])
        popular_sorted = popularity.sort_values("score", ascending=False)
        
        self.popular_items_cache = popular_sorted.head(20).to_dict(orient="records")

        return self

    def _get_user_content_profile(self, reviewer_id: str) -> Optional[np.ndarray]:
        """
        Builds user interest profile by computing a rating-weighted average of the TF-IDF
        vectors of items the user has positively interacted with.
        """
        if self.item_tfidf_matrix is None or reviewer_id not in self.user_ratings:
            return None

        ratings_map = self.user_ratings[reviewer_id]
        liked_asins = [asin for asin, rating in ratings_map.items() if rating >= 3.0 and asin in self.asin_to_tfidf_idx]
        if not liked_asins:
            liked_asins = [asin for asin in ratings_map.keys() if asin in self.asin_to_tfidf_idx]
            if not liked_asins:
                return None

        weights = np.array([ratings_map[asin] for asin in liked_asins])
        weights = weights / (weights.sum() + 1e-9)

        indices = [self.asin_to_tfidf_idx[asin] for asin in liked_asins]
        user_vector = np.zeros((1, self.item_tfidf_matrix.shape[1]))
        for idx, w in zip(indices, weights):
            user_vector += w * self.item_tfidf_matrix[idx].toarray()

        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
        return user_vector

    def recommend(
        self,
        reviewer_id: str,
        n: Optional[int] = None,
        alpha: Optional[float] = None,
        filter_purchased: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generates Top-N recommendations for a customer using the Hybrid blending formula:
        Score_Final = alpha * Score_SVD + (1 - alpha) * Score_Content

        Cold Start Handling:
        1. Customer Cold Start (< 3 interactions): Fallback to Popularity Bestsellers.
        2. Item Cold Start (novel item not in SVD): Pure Content-Based Filtering.

        Args:
            reviewer_id (str): Unique customer / reviewer identifier.
            n (Optional[int], optional): Number of recommendations to retrieve. Defaults to self.top_n.
            alpha (Optional[float], optional): Convex blending weight in [0, 1]. Defaults to self.default_alpha (0.65).
            filter_purchased (bool, optional): Whether to exclude items already bought by the user. Defaults to True.

        Returns:
            List[Dict[str, Any]]: Ranked list of product recommendation dictionaries with metadata, score, and provenance.
        """
        top_k = n or self.top_n
        mix_alpha = self.default_alpha if alpha is None else float(np.clip(alpha, 0.0, 1.0))
        purchased = self.user_history.get(reviewer_id, set()) if filter_purchased else set()
        user_interaction_count = len(self.user_history.get(reviewer_id, set()))

        # 1. Customer Cold-Start check: Less than 3 historical interactions
        if reviewer_id not in self.user_to_idx or user_interaction_count < 3:
            fallback_recs = []
            for p in self.popular_items_cache:
                if p["asin"] not in purchased:
                    fallback_recs.append({
                        "asin": p["asin"],
                        "title": p.get("title", f"Product {p['asin']}"),
                        "category": p.get("category", "General"),
                        "brand": p.get("brand", "Generic"),
                        "price": float(p.get("price", 29.99)),
                        "score": round(float(p.get("score", 4.5)), 3),
                        "recommendation_type": "Curated Best-Sellers (Popularity Fallback)",
                        "source": "Popularity Fallback"
                    })
                if len(fallback_recs) >= top_k:
                    break
            return fallback_recs

        # 2. Existing Customer: Compute Collaborative SVD Scores
        user_idx = self.user_to_idx[reviewer_id]
        u_vec = self.user_factors[user_idx]
        svd_all_items = np.dot(self.item_factors, u_vec)

        # MinMax scale SVD scores to [0, 1]
        svd_min, svd_max = svd_all_items.min(), svd_all_items.max()
        svd_range = svd_max - svd_min if (svd_max - svd_min) > 1e-6 else 1.0
        svd_scores_map = {
            self.idx_to_item[idx]: float((svd_all_items[idx] - svd_min) / svd_range)
            for idx in range(len(self.idx_to_item))
        }

        # 3. Content-Based NLP Similarity Scores
        user_profile_vec = self._get_user_content_profile(reviewer_id)
        if user_profile_vec is not None and self.item_tfidf_matrix is not None:
            content_sims = cosine_similarity(user_profile_vec, self.item_tfidf_matrix)[0]
            c_min, c_max = content_sims.min(), content_sims.max()
            c_range = c_max - c_min if (c_max - c_min) > 1e-6 else 1.0
            content_scores = (content_sims - c_min) / c_range
        else:
            content_scores = np.zeros(len(self.products_df))

        # 4. Hybrid Score Aggregation over Catalog
        candidate_items = []
        for prod_idx, row in self.products_df.iterrows():
            asin = row["asin"]
            if filter_purchased and asin in purchased:
                continue

            content_score = float(content_scores[prod_idx])
            in_svd = asin in svd_scores_map

            if in_svd:
                svd_score = svd_scores_map[asin]
                # Hybrid Blend: alpha * SVD + (1 - alpha) * Content
                final_score = (mix_alpha * svd_score) + ((1.0 - mix_alpha) * content_score)
                rec_type = "Hybrid (SVD + Content)" if 0.0 < mix_alpha < 1.0 else (
                    "Collaborative (SVD)" if mix_alpha == 1.0 else "Content-Based (NLP)"
                )
                source = "Hybrid" if 0.0 < mix_alpha < 1.0 else ("SVD" if mix_alpha == 1.0 else "Content")
            else:
                # Cold Item: Unseen in SVD matrix -> purely Content-Based
                final_score = content_score
                rec_type = "Content-Based (Cold-Start Item)"
                source = "Content"

            candidate_items.append({
                "asin": asin,
                "title": str(row.get("title", f"Product {asin}")),
                "category": str(row.get("category", "General")),
                "brand": str(row.get("brand", "Generic")),
                "price": float(row.get("price", 29.99)),
                "score": round(final_score, 4),
                "recommendation_type": rec_type,
                "source": source
            })

        # Sort descending by final hybrid score
        candidate_items.sort(key=lambda x: x["score"], reverse=True)
        return candidate_items[:top_k]
