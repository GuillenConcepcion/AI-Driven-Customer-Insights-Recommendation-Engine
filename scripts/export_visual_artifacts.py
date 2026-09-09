"""
High-Resolution Artifact Visualizer & Exporter
DS-AI Customer Insights & Lifetime Value (CLV) Engine
Author: Guillen Concepción - Senior Data Scientist & MLOps Engineer

Generates publication-grade 300 DPI figures and visual artifacts directly into images/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Ensure images directory exists
images_dir = Path("images")
images_dir.mkdir(parents=True, exist_ok=True)

data_path = Path("data/processed/customer_profiles.parquet")
if not data_path.exists():
    print(f"Error: {data_path} not found.")
    sys.exit(1)

df = pd.read_parquet(data_path)

# Dark theme palette styling matching Streamlit Ultra-HD aesthetics
plt.style.use("dark_background")
DARK_BG = "#111827"
PANEL_BG = "#1F2937"
ACCENT_BLUE = "#6366F1"
ACCENT_GREEN = "#10B981"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"
TEXT_COLOR = "#F3F4F6"
MUTED_TEXT = "#9CA3AF"

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#374151"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#374151"
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.alpha"] = 0.5

print("[INFO] Generating high-resolution visual artifacts into images/...")

# ==============================================================================
# 1. 3D PCA Topological Cluster Manifold with Centroids & Connection Vector
# ==============================================================================
fig = plt.figure(figsize=(10, 7.5), facecolor=DARK_BG)
ax = fig.add_subplot(111, projection="3d", facecolor=PANEL_BG)

# Subsample for crisp visualization
sample_df = df.sample(min(len(df), 750), random_state=42)
personas = sample_df["persona"].unique()
palette = {
    "Champions (VIPs)": ACCENT_AMBER,
    "Loyal Customers": ACCENT_BLUE,
    "Potential Loyalists": ACCENT_GREEN,
    "At-Risk Customers": ACCENT_RED,
    "Hibernating / Casuals": "#9CA3AF"
}

for persona in personas:
    sub = sample_df[sample_df["persona"] == persona]
    color = palette.get(persona, ACCENT_BLUE)
    ax.scatter(
        sub["pca_x"], sub["pca_y"], sub["pca_z"],
        c=color, label=persona, alpha=0.45, s=28, edgecolors="none"
    )

# Compute and plot centroids
centroids = df.groupby("cluster")[["pca_x", "pca_y", "pca_z"]].mean().reset_index()
for _, c in centroids.iterrows():
    ax.scatter(
        c["pca_x"], c["pca_y"], c["pca_z"],
        c="#FBBF24", marker="D", s=110, edgecolors="black", linewidth=1.5
    )
    ax.text(c["pca_x"], c["pca_y"], c["pca_z"] + 0.3, f" C#{int(c['cluster'])}", color="#FBBF24", fontsize=8, weight="bold")

# Target customer vector to centroid
sample_user = df.iloc[0]
target_c = centroids[centroids["cluster"] == sample_user["cluster"]].iloc[0]
ax.scatter(
    sample_user["pca_x"], sample_user["pca_y"], sample_user["pca_z"],
    c="#EF4444", marker="*", s=250, edgecolors="white", linewidth=1.5, label=f"User {sample_user['reviewerID']}"
)
ax.plot(
    [sample_user["pca_x"], target_c["pca_x"]],
    [sample_user["pca_y"], target_c["pca_y"]],
    [sample_user["pca_z"], target_c["pca_z"]],
    color="#F87171", linestyle="--", linewidth=2.0, label="Centroid Distance Vector"
)

ax.set_title("3D PCA Topological Space: Customer Position vs Cluster Centroids", fontsize=12, color=TEXT_COLOR, weight="bold", pad=12)
ax.set_xlabel("PC 1 (Volume & Cadence)", color=MUTED_TEXT, fontsize=9)
ax.set_ylabel("PC 2 (Monetary Variance)", color=MUTED_TEXT, fontsize=9)
ax.set_zlabel("PC 3 (Recency Drift)", color=MUTED_TEXT, fontsize=9)
ax.tick_params(colors=MUTED_TEXT, labelsize=7)
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), facecolor=PANEL_BG, edgecolor="#374151", fontsize=7.5)

fig.tight_layout()
pca_path = images_dir / "pca_3d_centroids.png"
fig.savefig(pca_path, dpi=300, facecolor=DARK_BG)
plt.close(fig)
print(f"[OK] Saved: {pca_path}")


# ==============================================================================
# 2. Economic Lorenz Curve & Gini Wealth Concentration Index (G ≈ 0.62)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

monetary_vals = np.sort(df["monetary"].values)
n = len(monetary_vals)
cum_monetary = np.cumsum(monetary_vals) / np.sum(monetary_vals)
cum_pop = np.arange(1, n + 1) / n

# Gini index
b_area = np.trapz(cum_monetary, cum_pop)
gini = 1.0 - 2.0 * b_area

ax.plot(cum_pop, cum_monetary, color=ACCENT_BLUE, linewidth=2.8, label=f"Empirical Lorenz Curve (Gini G = {gini:.2f})")
ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.5, label="Line of Perfect Equality (G = 0.0)")
ax.fill_between(cum_pop, cum_pop, cum_monetary, color=ACCENT_BLUE, alpha=0.18, label="Wealth Inequality Gap")

# Pareto 80/20 threshold mark
top_20_pop_idx = int(0.80 * n)
top_20_share = (1.0 - cum_monetary[top_20_pop_idx]) * 100

ax.axvline(0.80, color=ACCENT_AMBER, linestyle=":", linewidth=1.5)
ax.axhline(cum_monetary[top_20_pop_idx], color=ACCENT_AMBER, linestyle=":", linewidth=1.5)
ax.scatter([0.80], [cum_monetary[top_20_pop_idx]], color=ACCENT_AMBER, s=90, zorder=5)

ax.annotate(
    f"Pareto Dynamics:\nTop 20% Customers generate {top_20_share:.1f}% GMV",
    xy=(0.80, cum_monetary[top_20_pop_idx]),
    xytext=(0.42, 0.22),
    color="#FDE68A",
    fontsize=9,
    weight="bold",
    arrowprops=dict(facecolor=ACCENT_AMBER, arrowstyle="->", lw=1.5)
)

ax.set_title("Economic Concentration of Customer Spend (Lorenz Curve)", fontsize=13, color=TEXT_COLOR, weight="bold", pad=12)
ax.set_xlabel("Cumulative Share of Customers (Fraction of Population)", color=MUTED_TEXT, fontsize=10)
ax.set_ylabel("Cumulative Share of Total Revenue (GMV)", color=MUTED_TEXT, fontsize=10)
ax.tick_params(colors=MUTED_TEXT, labelsize=8)
ax.legend(loc="upper left", facecolor=PANEL_BG, edgecolor="#374151", fontsize=8.5)
ax.grid(True)

fig.tight_layout()
lorenz_path = images_dir / "lorenz_curve_gini.png"
fig.savefig(lorenz_path, dpi=300, facecolor=DARK_BG)
plt.close(fig)
print(f"[OK] Saved: {lorenz_path}")


# ==============================================================================
# 3. Action Matrix: Predicted 1-Year CLV vs Churn Risk
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

clv_target = "clv_pred" if "clv_pred" in df.columns else "projected_clv_1y"
clv_median = df[clv_target].median()

scatter = ax.scatter(
    df["churn_probability"],
    df[clv_target],
    c=df["churn_probability"],
    cmap="Spectral_r",
    alpha=0.55,
    s=35,
    edgecolors="none"
)

ax.axvline(0.50, color="#9CA3AF", linestyle="--", linewidth=1.5)
ax.axhline(clv_median, color="#9CA3AF", linestyle="--", linewidth=1.5)

# Quadrant labels
ax.text(0.12, df[clv_target].max() * 0.88, "QUADRANT I: NURTURE CHAMPIONS\n(Low Churn Risk / High CLV)\nVIP Beta programs & Early access", color=ACCENT_GREEN, fontsize=8.5, weight="bold")
ax.text(0.62, df[clv_target].max() * 0.88, "QUADRANT II: RESCUE TARGETS (CRITICAL)\n(High Churn Risk / High CLV)\nKnapsack retention budget priority", color=ACCENT_RED, fontsize=8.5, weight="bold")
ax.text(0.12, df[clv_target].min() + 30, "QUADRANT III: STEADY MAINSTREAM\n(Low Churn / Low CLV)\nCross-sell & basket uplift", color=ACCENT_BLUE, fontsize=8.5, weight="bold")
ax.text(0.62, df[clv_target].min() + 30, "QUADRANT IV: AUTOMATED LOW-TOUCH\n(High Churn / Low CLV)\nSeasonal mass retargeting", color=MUTED_TEXT, fontsize=8.5, weight="bold")

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Churn Probability P(Churn)", color=MUTED_TEXT, fontsize=9)
cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=8)

ax.set_title("Prescriptive Decision Matrix: Residual CLV vs Churn Probability", fontsize=13, color=TEXT_COLOR, weight="bold", pad=12)
ax.set_xlabel("Predicted Probability of Churn P(Churn)", color=MUTED_TEXT, fontsize=10)
ax.set_ylabel("Predicted 1-Year Forward CLV ($)", color=MUTED_TEXT, fontsize=10)
ax.tick_params(colors=MUTED_TEXT, labelsize=8)
ax.grid(True)

fig.tight_layout()
action_matrix_path = images_dir / "clv_churn_matrix.png"
fig.savefig(action_matrix_path, dpi=300, facecolor=DARK_BG)
plt.close(fig)
print(f"[OK] Saved: {action_matrix_path}")


# ==============================================================================
# 4. Statistical Publication Panel (300 DPI: Boxen Plot + Correlation Heatmap)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
ax1.set_facecolor(PANEL_BG)
ax2.set_facecolor(PANEL_BG)

# 4A. Boxen Plot of Spend across Personas
order = ["Champions (VIPs)", "Loyal Customers", "Potential Loyalists", "At-Risk Customers", "Hibernating / Casuals"]
order = [o for o in order if o in df["persona"].unique()]
sns.boxenplot(
    data=df,
    x="persona",
    y="monetary",
    order=order,
    palette=[ACCENT_AMBER, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, "#6B7280"],
    ax=ax1,
    showfliers=False
)
ax1.set_yscale("log")
ax1.set_title("Monetary Spend Dispersion Across Arquetypes", fontsize=11, color=TEXT_COLOR, weight="bold")
ax1.set_xlabel("Customer Persona Archetype", color=MUTED_TEXT, fontsize=9)
ax1.set_ylabel("Monetary Spend ($ Log Scale)", color=MUTED_TEXT, fontsize=9)
ax1.tick_params(axis="x", rotation=25, colors=MUTED_TEXT, labelsize=8)
ax1.tick_params(axis="y", colors=MUTED_TEXT, labelsize=8)
ax1.grid(True, axis="y")

# 4B. Feature Correlation Heatmap
corr_cols = [c for c in ["recency", "frequency", "monetary", "customer_age_T", "avg_rating", "clv_pred"] if c in df.columns]
corr_matrix = df[corr_cols].corr(method="spearman")

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    ax=ax2,
    cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
    annot_kws={"size": 8.5, "weight": "bold"}
)
ax2.set_title("Feature Interdependency Matrix (Spearman ρ)", fontsize=11, color=TEXT_COLOR, weight="bold")
ax2.tick_params(colors=MUTED_TEXT, labelsize=8)

fig.suptitle("Publication-Grade Statistical Validation Panels (300 DPI)", fontsize=13, color=TEXT_COLOR, weight="bold", y=1.02)
fig.tight_layout()
stat_panel_path = images_dir / "statistical_panels_300dpi.png"
fig.savefig(stat_panel_path, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"[OK] Saved: {stat_panel_path}")


# ==============================================================================
# 5. J-Shaped Review Rating Dynamics
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

rating_counts = df["avg_rating"].round().value_counts().sort_index()
stars = [f"★ {int(s)}" for s in rating_counts.index]
bar_colors = ["#EF4444", "#F97316", "#F59E0B", "#10B981", "#6366F1"]

bars = ax.bar(stars, rating_counts.values, color=bar_colors, edgecolor="black", linewidth=1.2, width=0.6)

for bar in bars:
    height = bar.get_height()
    pct = (height / len(df)) * 100
    ax.annotate(f"{height:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, color=TEXT_COLOR, weight="bold")

ax.set_title("E-Commerce J-Shaped Satisfaction Rating Dynamics", fontsize=12, color=TEXT_COLOR, weight="bold", pad=12)
ax.set_xlabel("Rounded Review Rating", color=MUTED_TEXT, fontsize=9.5)
ax.set_ylabel("Customer Cohort Count", color=MUTED_TEXT, fontsize=9.5)
ax.tick_params(colors=MUTED_TEXT, labelsize=8.5)
ax.grid(True, axis="y")

fig.tight_layout()
j_shaped_path = images_dir / "j_shaped_ratings.png"
fig.savefig(j_shaped_path, dpi=300, facecolor=DARK_BG)
plt.close(fig)
print(f"[OK] Saved: {j_shaped_path}")

print("[SUCCESS] All 5 high-resolution artifact captures generated successfully.")
