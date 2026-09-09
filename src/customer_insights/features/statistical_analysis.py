"""
Statistical Analysis & Hypothesis Testing Module.
Performs:
- Univariate and bivariate distributional diagnostics (Gini, Skewness, Kurtosis)
- Inferential Hypothesis Testing (Shapiro-Wilk, Kruskal-Wallis, Spearman, Chi-Square)
- Prescriptive Marketing Budget Optimizer
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Computes Gini Coefficient of inequality for monetary/spend distributions.

    Args:
        values (np.ndarray): 1D array of customer monetary expenditure values.

    Returns:
        float: Gini index in [0, 1] where 0 denotes perfect equality and 1 maximal concentration.
    """
    clean_vals = np.sort(values[values >= 0])
    n = len(clean_vals)
    if n == 0 or np.sum(clean_vals) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(((2 * np.sum(index * clean_vals)) / (n * np.sum(clean_vals))) - ((n + 1) / n))


def run_univariate_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary statistics, skewness, kurtosis, and Gini coefficient.

    Args:
        df (pd.DataFrame): Dataframe containing behavioral columns (recency, frequency, monetary, avg_rating).

    Returns:
        Dict[str, Any]: Dictionary of metric diagnostics including mean, std, median, IQR, skewness, kurtosis, and Gini.
    """
    results = {}
    for col in ["recency", "frequency", "monetary", "avg_rating"]:
        if col in df.columns:
            series = df[col].dropna()
            q25, q75 = np.percentile(series, [25, 75])
            results[col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "median": round(float(series.median()), 2),
                "iqr": round(float(q75 - q25), 2),
                "q25": round(float(q25), 2),
                "q75": round(float(q75), 2),
                "skewness": round(float(stats.skew(series)), 3),
                "kurtosis": round(float(stats.kurtosis(series)), 3)
            }

    if "monetary" in df.columns:
        results["monetary"]["gini_coefficient"] = round(compute_gini_coefficient(df["monetary"].values), 3)

    return results


def run_hypothesis_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes formal statistical hypothesis tests:
    1. Shapiro-Wilk (Normality on sample)
    2. Kruskal-Wallis (Monetary differences across clusters)
    3. Spearman Rank Correlation (Frequency vs Rating)
    4. Chi-Square Test (Persona vs Churn Risk Level)

    Args:
        df (pd.DataFrame): Customer profiles dataframe with cluster, persona, and RFM metrics.

    Returns:
        Dict[str, Any]: Comprehensive test report with test statistics, p-values, rejection decisions, and conclusions.
    """
    report = {}

    # 1. Shapiro-Wilk test for normality on subsample
    subsample = df["monetary"].dropna().sample(min(500, len(df)), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(subsample)
    report["normality_test_monetary"] = {
        "test": "Shapiro-Wilk",
        "statistic_W": round(float(shapiro_stat), 4),
        "p_value": float(shapiro_p),
        "reject_H0_normal": bool(shapiro_p < 0.05),
        "conclusion": "Non-normal distribution. Non-parametric methods required." if shapiro_p < 0.05 else "Normal distribution."
    }

    # 2. Kruskal-Wallis: Difference in monetary value across clusters
    if "cluster" in df.columns and df["cluster"].nunique() > 1:
        groups = [group["monetary"].values for _, group in df.groupby("cluster")]
        kw_stat, kw_p = stats.kruskal(*groups)
        report["kruskal_wallis_clusters_monetary"] = {
            "test": "Kruskal-Wallis",
            "statistic_H": round(float(kw_stat), 2),
            "p_value": float(kw_p),
            "reject_H0_identical": bool(kw_p < 0.05),
            "conclusion": "Statistically significant differences in spend between clusters." if kw_p < 0.05 else "No significant differences."
        }

    # 3. Spearman Rank Correlation: Frequency vs Avg Rating
    if "frequency" in df.columns and "avg_rating" in df.columns:
        spearman_rho, spearman_p = stats.spearmanr(df["frequency"], df["avg_rating"])
        report["spearman_freq_vs_rating"] = {
            "test": "Spearman Rank Correlation",
            "rho": round(float(spearman_rho), 4),
            "p_value": float(spearman_p),
            "reject_H0_independence": bool(spearman_p < 0.05),
            "conclusion": "Significant monotonic correlation between frequency and satisfaction." if spearman_p < 0.05 else "Independent."
        }

    # 4. Chi-Square Test of Independence: Persona vs Churn Risk Level
    if "persona" in df.columns and "churn_risk_level" in df.columns:
        contingency = pd.crosstab(df["persona"], df["churn_risk_level"])
        chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(contingency)
        report["chi_square_persona_vs_churn"] = {
            "test": "Chi-Square Test of Independence",
            "statistic_chi2": round(float(chi2_stat), 2),
            "degrees_of_freedom": int(dof),
            "p_value": float(chi2_p),
            "reject_H0_independence": bool(chi2_p < 0.05),
            "conclusion": "Persona classification is strongly dependent on churn risk." if chi2_p < 0.05 else "Independent."
        }

    return report


def optimize_retention_budget(
    df: pd.DataFrame,
    total_budget: float = 5000.0,
    intervention_cost: float = 15.0,
    expected_retention_lift: float = 0.25
) -> Dict[str, Any]:
    """
    Prescriptive Optimization:
    Maximizes expected net retained CLV subject to budget constraint (0-1 Knapsack formulation).

    Args:
        df (pd.DataFrame): Customer profiles dataframe with churn_probability and projected_clv_1y.
        total_budget (float, optional): Total capital allocated for retention interventions in USD. Defaults to 5000.0.
        intervention_cost (float, optional): Unit cost per targeted customer intervention in USD. Defaults to 15.0.
        expected_retention_lift (float, optional): Estimated delta reduction in churn risk [0, 1]. Defaults to 0.25.

    Returns:
        Dict[str, Any]: Prescriptive budget allocation report containing:
            - budget_allocated (float)
            - actual_campaign_cost (float)
            - number_of_customers_targeted (int)
            - total_expected_net_clv_saved (float)
            - expected_roi_ratio (float)
            - top_targeted_customers (List[Dict])
    """
    # Filter candidates: High churn risk with positive CLV
    candidates = df[(df["churn_probability"] >= 0.40) & (df["projected_clv_1y"] > intervention_cost)].copy()
    
    # Expected net value gain: Lift * CLV - Cost
    candidates["expected_net_gain"] = (candidates["projected_clv_1y"] * expected_retention_lift) - intervention_cost
    
    # Sort by expected net gain descending (Knapsack greedy optimal for uniform cost)
    sorted_candidates = candidates.sort_values("expected_net_gain", ascending=False)
    
    max_interventions = int(total_budget // intervention_cost)
    selected = sorted_candidates.head(max_interventions)
    
    total_spend = len(selected) * intervention_cost
    total_projected_roi = float(selected["expected_net_gain"].sum())
    
    return {
        "budget_allocated": total_budget,
        "actual_campaign_cost": total_spend,
        "number_of_customers_targeted": len(selected),
        "total_expected_net_clv_saved": round(total_projected_roi, 2),
        "expected_roi_ratio": round((total_projected_roi + total_spend) / max(total_spend, 1.0), 2),
        "top_targeted_customers": selected[["reviewerID", "persona", "churn_probability", "projected_clv_1y", "expected_net_gain"]].head(10).to_dict(orient="records")
    }


if __name__ == "__main__":
    from ..config import PROCESSED_DATA_DIR
    profiles_path = PROCESSED_DATA_DIR / "customer_profiles.parquet"
    if profiles_path.exists():
        profiles = pd.read_parquet(profiles_path)
        diag = run_univariate_diagnostics(profiles)
        print("Univariate diagnostics:", diag)
        tests = run_hypothesis_tests(profiles)
        print("Hypothesis tests:", tests)
        opt = optimize_retention_budget(profiles)
        print("Budget optimization:", opt)
