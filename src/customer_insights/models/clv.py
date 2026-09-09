"""
Customer Lifetime Value (CLV) and Churn Probability modeling.
Combines transaction cadence, customer tenure, recency, and spend to estimate
12-month expected forward value and retention probability.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


class CustomerLifetimeValueModel:
    """
    Probabilistic and Predictive Customer Lifetime Value model.
    Combines Poisson GLM regression on RFM behavioral features with Buy-Till-You-Die (BTYD)
    purchase velocity to estimate forward 12-month CLV and retention probability.
    """

    def __init__(self, discount_rate: float = 0.08, time_horizon_days: int = 365):
        self.discount_rate = discount_rate
        self.time_horizon_days = time_horizon_days
        self.baseline_churn_rate_ = 0.20
        self.avg_order_value_ = 45.0
        self.regressor_scaler = StandardScaler()
        self.poisson_regressor: Optional[PoissonRegressor] = None

    def fit(self, rfm_df: pd.DataFrame) -> "CustomerLifetimeValueModel":
        """
        Calibrates baseline purchase velocity and fits Poisson GLM regression on RFM features.

        Args:
            rfm_df (pd.DataFrame): Customer RFM feature dataframe containing columns:
                ['recency', 'frequency', 'monetary', 'customer_age_T'].

        Returns:
            CustomerLifetimeValueModel: Fitted model instance ready for CLV inference.
        """
        valid_orders = rfm_df["frequency"].clip(lower=1)
        self.avg_order_value_ = float((rfm_df["monetary"] / valid_orders).median())

        # Fit Poisson Regression for CLV prediction:
        # Target: Forward 12-month projected value estimated from customer velocity
        tenure = np.maximum(rfm_df["customer_age_T"], 1)
        frequency = np.maximum(rfm_df["frequency"], 1)
        recency = np.maximum(rfm_df["recency"], 1)
        avg_spend = rfm_df["monetary"] / frequency

        avg_cadence = np.maximum(tenure / frequency, 7.0)
        p_active_approx = 1.0 / (1.0 + np.exp(0.8 * ((recency / avg_cadence) - 1.5)))
        target_clv = np.maximum((365.0 / avg_cadence) * p_active_approx * avg_spend * (1.0 / (1.0 + self.discount_rate)), 5.0)

        feature_cols = ["recency", "frequency", "monetary", "customer_age_T"]
        X = rfm_df[feature_cols].copy()
        X_scaled = self.regressor_scaler.fit_transform(X)

        self.poisson_regressor = PoissonRegressor(max_iter=500, alpha=0.1)
        self.poisson_regressor.fit(X_scaled, target_clv)

        return self

    def predict_retention_and_clv(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates P(Active), churn risk, expected forward transactions, and 12-month CLV.

        Args:
            rfm_df (pd.DataFrame): Dataframe containing RFM behavioural metrics.

        Returns:
            pd.DataFrame: Augmented dataframe with ['clv_pred', 'p_active', 'churn_probability',
                'expected_transactions_1y', 'projected_clv_1y'].
        """
        df = rfm_df.copy()

        # Purchase velocity (purchases per day of active tenure)
        tenure = np.maximum(df["customer_age_T"], 1)
        recency = np.maximum(df["recency"], 1)
        frequency = np.maximum(df["frequency"], 1)

        # Average interpurchase time in days
        avg_interpurchase_days = np.maximum(tenure / frequency, 7.0)

        # P(Active): Ratio of inactivity relative to expected cadence
        # If recency is significantly longer than avg_interpurchase_days, P(Active) drops
        cadence_ratio = recency / avg_interpurchase_days
        p_active = 1.0 / (1.0 + np.exp(0.8 * (cadence_ratio - 1.5)))
        p_active = np.clip(p_active, 0.05, 0.99)

        # Expected transactions over time_horizon_days
        annual_velocity = (365.0 / avg_interpurchase_days)
        expected_future_tx = annual_velocity * p_active

        # Average monetary value per order
        avg_spend_per_tx = df["monetary"] / frequency

        # Projected 1-Year CLV with net present value discounting
        discount_factor = 1.0 / (1.0 + self.discount_rate)
        projected_clv = expected_future_tx * avg_spend_per_tx * discount_factor

        # Poisson regressor forward prediction if fitted
        if self.poisson_regressor is not None:
            feature_cols = ["recency", "frequency", "monetary", "customer_age_T"]
            X = df[feature_cols].copy()
            X_scaled = self.regressor_scaler.transform(X)
            df["clv_pred"] = self.poisson_regressor.predict(X_scaled).round(2)
        else:
            df["clv_pred"] = projected_clv.round(2)

        df["p_active"] = p_active.round(3)
        df["churn_probability"] = (1.0 - p_active).round(3)
        df["expected_transactions_1y"] = expected_future_tx.round(2)
        df["projected_clv_1y"] = projected_clv.round(2)

        # Churn risk category
        df["churn_risk_level"] = pd.cut(
            df["churn_probability"],
            bins=[0.0, 0.30, 0.65, 1.0],
            labels=["Low", "Medium", "High"]
        )

        return df

    def predict_single(
        self,
        recency: float,
        frequency: float,
        monetary: float,
        customer_age_T: float
    ) -> Dict[str, Any]:
        """
        Inference for a single customer profile. Returns both Poisson CLV prediction and BTYD cadence metrics.
        """
        tenure = max(customer_age_T, 1.0)
        rec = max(recency, 1.0)
        freq = max(frequency, 1.0)

        avg_interpurchase_days = max(tenure / freq, 7.0)
        cadence_ratio = rec / avg_interpurchase_days
        p_active = float(np.clip(1.0 / (1.0 + np.exp(0.8 * (cadence_ratio - 1.5))), 0.05, 0.99))
        churn_prob = round(1.0 - p_active, 3)

        expected_tx = round((365.0 / avg_interpurchase_days) * p_active, 2)
        avg_spend = monetary / freq
        discount_factor = 1.0 / (1.0 + self.discount_rate)
        projected_clv = round(expected_tx * avg_spend * discount_factor, 2)

        if self.poisson_regressor is not None:
            sample_df = pd.DataFrame([{
                "recency": rec,
                "frequency": freq,
                "monetary": monetary,
                "customer_age_T": tenure
            }])
            X_scaled = self.regressor_scaler.transform(sample_df)
            clv_pred_val = float(round(self.poisson_regressor.predict(X_scaled)[0], 2))
        else:
            clv_pred_val = projected_clv

        risk_level = "Low" if churn_prob < 0.30 else ("Medium" if churn_prob < 0.65 else "High")

        return {
            "clv_pred": clv_pred_val,
            "projected_clv_1y": projected_clv,
            "p_active": round(p_active, 3),
            "churn_probability": churn_prob,
            "churn_risk_level": risk_level,
            "expected_transactions_1y": expected_tx
        }
