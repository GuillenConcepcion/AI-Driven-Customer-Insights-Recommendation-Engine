"""
Preprocessing pipelines for RFM features adhering to Scikit-Learn standards.
Applies log1p transformations for skewed variables and standard scaling.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["recency", "frequency", "monetary", "avg_rating"]


class RFMFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms raw RFM metrics into normalized, scaled features suitable for clustering.
    Applies log1p to Recency, Frequency, and Monetary to handle right-skewed distributions.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.scaler = StandardScaler()
        self.fitted_ = False

    def _apply_log_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        df_trans = X[self.feature_columns].copy()
        # Log1p on positive skewed metrics
        for col in ["recency", "frequency", "monetary"]:
            if col in df_trans.columns:
                df_trans[col] = np.log1p(np.maximum(df_trans[col], 0))
        return df_trans

    def fit(self, X: pd.DataFrame, y=None):
        df_trans = self._apply_log_transforms(X)
        self.scaler.fit(df_trans)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")
        df_trans = self._apply_log_transforms(X)
        return self.scaler.transform(df_trans)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)
