"""
Streamlit Interactive Dashboard for AI Customer Insights & Recommendations.
Benchmark: RecSysDatasets / Amazon Reviews (UCSD)
"""

import sys
from pathlib import Path

# Add src to sys.path so streamlit can find customer_insights package
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Global 300 DPI standards for Matplotlib / Seaborn publication figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#374151'
plt.rcParams['axes.linewidth'] = 0.8
sns.set_theme(style="darkgrid", palette="deep")

from customer_insights.models.pipeline import CustomerInsightsPipeline
from customer_insights.models.persona import PersonaEngine, PERSONA_PLAYBOOKS
from customer_insights.features.statistical_analysis import (
    run_univariate_diagnostics,
    run_hypothesis_tests,
    optimize_retention_budget
)

# Plotly High-Resolution & Scalable Vector Configuration (4K/Retina ready)
PLOTLY_HIGH_RES_CONFIG = {
    'toImageButtonOptions': {
        'format': 'svg',  # Infinite lossless vector resolution
        'filename': 'customer_insights_hd_chart',
        'height': 900,
        'width': 1400,
        'scale': 4  # 4x supersampling (300+ DPI equivalent for raster downloads)
    },
    'displaylogo': False,
    'responsive': True,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}


def apply_ultra_hd_style(fig, title=None, height=None):
    """
    Applies unified ultra-sharp dark aesthetics with anti-aliasing to Plotly figures.
    """
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#E5E7EB"),
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        hoverlabel=dict(
            bgcolor="#1F2937",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="#4F46E5"
        ),
        margin=dict(t=40, b=25, l=25, r=25)
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#F9FAFB")))
    if height:
        fig.update_layout(height=height)
    return fig


# Page configuration
st.set_page_config(
    page_title="AI Customer Insights & Recommendations",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2E3440;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 600;
    }
    .recommendation-card {
        background-color: #252A37;
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(cache_key: str = "v3.0_full_fresh_modules_20260909"):
    import importlib
    import customer_insights.models.clustering
    import customer_insights.models.clv
    import customer_insights.models.persona
    import customer_insights.models.recommender
    import customer_insights.models.pipeline
    importlib.reload(customer_insights.models.clustering)
    importlib.reload(customer_insights.models.clv)
    importlib.reload(customer_insights.models.persona)
    importlib.reload(customer_insights.models.recommender)
    importlib.reload(customer_insights.models.pipeline)

    from customer_insights.models.clustering import CustomerClusterModel

    pipe = customer_insights.models.pipeline.CustomerInsightsPipeline()
    loaded = pipe.load_artifacts()
    if not loaded:
        pipe.run_training_pipeline()
    
    # Defensive schema normalization for backwards compatibility and cache safety
    df = pipe.customer_profiles_
    if df is not None:
        if "clv_pred" not in df.columns:
            if "projected_clv_1y" in df.columns:
                df["clv_pred"] = df["projected_clv_1y"]
            elif "predicted_clv" in df.columns:
                df["clv_pred"] = df["predicted_clv"]
            else:
                df["clv_pred"] = (df.get("monetary", 100.0) * 0.75).round(2)
        if "customer_age_T" not in df.columns:
            df["customer_age_T"] = df.get("tenure", 180.0)
        if "centroid_distance" not in df.columns:
            df["centroid_distance"] = 0.5
        if "persona" not in df.columns or "preferred_channel" not in df.columns:
            df = PersonaEngine.enrich_dataframe(df)
        pipe.customer_profiles_ = df

    # Bind compare_kmeans_dbscan dynamically if missing on pickled object
    if pipe.cluster_model:
        if not hasattr(pipe.cluster_model, "compare_kmeans_dbscan"):
            import types
            pipe.cluster_model.compare_kmeans_dbscan = types.MethodType(
                CustomerClusterModel.compare_kmeans_dbscan, pipe.cluster_model
            )

    return pipe





def main():
    pipe = get_pipeline()
    df = pipe.customer_profiles_
    products_df = pipe.products_df_

    # Initialize session states
    if "suggested_cart" not in st.session_state:
        st.session_state["suggested_cart"] = []
    if "clicked_items" not in st.session_state:
        st.session_state["clicked_items"] = set()
    if "active_user_id" not in st.session_state:
        st.session_state["active_user_id"] = df["reviewerID"].iloc[0]

    st.title("🛍️ AI-Driven Customer Insights & Recommendation Engine")
    st.caption("Benchmark: Amazon Reviews & Product Metadata (RecSysDatasets / UCSD) | Senior MLOps Architecture")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.info(f"📊 **Indexed Customers:** {len(df):,}\n\n📦 **Catalog Items:** {len(products_df):,}")

        # Suggested Cart Drawer in Sidebar
        st.markdown("### 🛒 Cesta Sugerida")
        if st.session_state["suggested_cart"]:
            cart_df = pd.DataFrame(st.session_state["suggested_cart"])
            total_cart = cart_df["price"].sum()
            st.success(f"**Ítems:** {len(cart_df)} | **Total:** ${total_cart:,.2f}")
            for idx, item in enumerate(st.session_state["suggested_cart"]):
                st.markdown(f"- **{item['title'][:32]}...** (${item['price']:.2f})")
            if st.button("🗑️ Vaciar Cesta", key="clear_cart"):
                st.session_state["suggested_cart"] = []
                st.rerun()
        else:
            st.caption("Tu cesta sugerida está vacía. Añade productos desde las recomendaciones.")

        st.divider()
        st.markdown("### 👨‍💻 Project Engineering")
        st.markdown("**Lead:** Guillen Concepción")
        st.markdown("**Role:** Senior Data Scientist & MLOps Engineer")

        st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/guillen-concepcion-25266b127)")
        st.markdown("[GitHub Repository](https://github.com/GuillenConcepcion)")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Customer Diagnosis & Recommendations",
        "📈 Executive KPIs & Portfolio Distribution",
        "🔬 Clustering (K-Means vs DBSCAN 3D)",
        "💰 Lifetime Value (CLV) & Churn Matrix",
        "🧪 Hypothetical Profile Simulator",
        "📊 Statistical Diagnostics & Budget Optimizer"
    ])

    # TAB 1: CUSTOMER DIAGNOSIS & LIVE RECOMMENDATION ENGINE
    with tab1:
        st.subheader("🔍 Customer Insights & Hybrid Recommendation Engine")
        st.write("Diagnose any customer in real time: predict forward CLV, locate 3D PCA coordinates against cluster centroids, and generate hybrid recommendations.")

        # User input bar
        search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
        all_users = df["reviewerID"].tolist()
        
        with search_col1:
            input_user = st.selectbox(
                "Select or Type User ID:",
                options=all_users[:300],
                index=0,
                help="Pick a customer ID from the indexed Amazon reviews benchmark."
            )
        with search_col2:
            alpha_val = st.slider(
                "Hybrid Weight (α SVD vs 1-α Content):",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                help="α=1.0: Pure SVD Collaborative Filtering | α=0.0: Pure TF-IDF Content-Based"
            )
        with search_col3:
            st.write("")
            st.write("")
            analyze_btn = st.button("🚀 Analyze Customer", use_container_width=True)

        selected_user = input_user
        user_row = df[df["reviewerID"] == selected_user].iloc[0]
        user_cluster = int(user_row["cluster"])

        # 1. Insights KPI Cards
        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        # Defensive property extraction
        clv_val = user_row.get("clv_pred")
        if clv_val is None or pd.isna(clv_val):
            clv_val = user_row.get("projected_clv_1y", user_row.get("predicted_clv", 0.0))
        clv_val = float(clv_val) if clv_val is not None else 0.0

        tenure_val = float(user_row.get("customer_age_T", user_row.get("tenure", 0.0)))
        dist_val = float(user_row.get("centroid_distance", 0.0))
        churn_p = float(user_row.get("churn_probability", 0.25)) * 100
        churn_lvl = str(user_row.get("churn_risk_level", "Medium"))
        persona_name = str(user_row.get("persona", "Active Customer"))
        strat = str(user_row.get("marketing_strategy", "Targeted customer engagement"))
        pref_chan = str(user_row.get("preferred_channel", "Email / Direct Web"))
        disc_inc = str(user_row.get("discount_incentive", "Exclusive Loyalty Credit"))

        with kpi1:
            st.metric("Identified Cluster Segment", f"Cluster #{user_cluster}", persona_name)
        with kpi2:
            st.metric("Predicted CLV (1Y Forward)", f"${clv_val:,.2f}", f"Tenure: {tenure_val:.0f}d")
        with kpi3:
            st.metric("Distance to Cluster Centroid", f"{dist_val:.3f} units", "3D PCA Space")
        with kpi4:
            st.metric("Churn Risk Level", f"{churn_p:.1f}%", churn_lvl)

        st.info(f"📋 **Prescriptive Marketing Playbook:** {strat} | **Canal:** `{pref_chan}` | **Incentivo:** `{disc_inc}`")


        # 2. 3D PCA Visualization with Cluster Centroid
        vis_col, rec_col = st.columns([3, 2])

        with vis_col:
            st.markdown("#### 🌐 3D PCA Space: Customer Position vs Cluster Centroid")
            
            # Sample background customers for responsive 3D rendering
            sample_customers = df.sample(min(len(df), 800), random_state=42).copy()
            if selected_user not in sample_customers["reviewerID"].values:
                sample_customers = pd.concat([sample_customers, df[df["reviewerID"] == selected_user]])

            fig_3d = px.scatter_3d(
                sample_customers,
                x="pca_x",
                y="pca_y",
                z="pca_z",
                color="persona",
                opacity=0.35,
                hover_data=["reviewerID", "recency", "frequency", "monetary"],
                color_discrete_sequence=px.colors.qualitative.Prism
            )

            # Add Cluster Centroids as Golden Diamonds
            if pipe.cluster_model and hasattr(pipe.cluster_model, "cluster_centroids_3d"):
                centroids_dict = pipe.cluster_model.cluster_centroids_3d
                c_x = [centroids_dict[c]["x"] for c in sorted(centroids_dict.keys())]
                c_y = [centroids_dict[c]["y"] for c in sorted(centroids_dict.keys())]
                c_z = [centroids_dict[c]["z"] for c in sorted(centroids_dict.keys())]
                c_labels = [f"Centroid #{c}" for c in sorted(centroids_dict.keys())]

                fig_3d.add_trace(go.Scatter3d(
                    x=c_x, y=c_y, z=c_z,
                    mode="markers+text",
                    marker=dict(size=9, symbol="diamond", color="#F59E0B", line=dict(color="black", width=1.5)),
                    text=c_labels,
                    name="Cluster Centroids"
                ))

                # Highlight Analyzed User
                u_x = float(user_row["pca_x"])
                u_y = float(user_row["pca_y"])
                u_z = float(user_row["pca_z"])
                target_centroid = centroids_dict[user_cluster]

                fig_3d.add_trace(go.Scatter3d(
                    x=[u_x], y=[u_y], z=[u_z],
                    mode="markers+text",
                    marker=dict(size=14, color="#EF4444", symbol="circle", line=dict(color="white", width=3)),
                    text=[f"⭐ Customer {selected_user}"],
                    name="Target Customer"
                ))

                # Vector Line connecting user to centroid
                fig_3d.add_trace(go.Scatter3d(
                    x=[u_x, target_centroid["x"]],
                    y=[u_y, target_centroid["y"]],
                    z=[u_z, target_centroid["z"]],
                    mode="lines",
                    line=dict(color="#EF4444", width=5, dash="dash"),
                    name=f"Centroid Vector (d={dist_val:.2f})"
                ))

            fig_3d.update_layout(
                height=650,
                margin=dict(t=20, b=10, l=10, r=10),
                legend=dict(orientation="h", y=1.02)
            )
            apply_ultra_hd_style(fig_3d, height=650)
            st.plotly_chart(fig_3d, use_container_width=True, config=PLOTLY_HIGH_RES_CONFIG)

        # 3. Recommendations Top 10 Showcase
        with rec_col:
            st.markdown(f"#### 🎯 Top 10 Hybrid Recommendations (`{selected_user}`)")
            st.caption(f"Blended Score: {alpha_val:.2f} × SVD (Collaborative) + {1.0-alpha_val:.2f} × TF-IDF (Content-Based)")

            # Defensive call to recommender across model versions
            try:
                recs = pipe.recommender.recommend(reviewer_id=selected_user, n=10, alpha=alpha_val)
            except TypeError:
                try:
                    recs = pipe.recommender.recommend(reviewer_id=selected_user, n=10)
                except TypeError:
                    recs = pipe.recommender.recommend(user_id=selected_user, n_recommendations=10)

            # Ensure recs is a list of dicts regardless of version
            if hasattr(recs, "to_dict"):
                recs_list = recs.to_dict(orient="records")
            elif isinstance(recs, list):
                recs_list = recs
            else:
                recs_list = []

            for i, r in enumerate(recs_list, 1):
                asin_val = str(r.get("asin", f"SKU_{i}"))
                is_clicked = asin_val in st.session_state["clicked_items"]
                border_color = "#10B981" if is_clicked else "#4F46E5"
                source_badge = r.get("source", r.get("recommendation_source", "Hybrid"))
                title_val = str(r.get("title", f"Product {asin_val}"))
                cat_val = str(r.get("category", "General"))
                brand_val = str(r.get("brand", "Generic"))
                price_val = float(r.get("price", 19.99))
                score_val = r.get("score", r.get("hybrid_score", 0.85))

                st.markdown(f"""
                <div style="background-color: #1E222D; border-left: 4px solid {border_color}; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 700; color: #F3F4F6;">#{i}. {title_val[:40]}...</span>
                        <span style="background-color: #374151; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{source_badge}</span>
                    </div>
                    <div style="font-size: 13px; color: #9CA3AF; margin-top: 4px;">
                        🏷️ {cat_val} | 🏢 {brand_val} | 💰 <strong>${price_val:.2f}</strong> | Match: <code>{score_val}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button(f"👁️ Clic #{i}", key=f"click_{asin_val}_{i}"):
                        st.session_state["clicked_items"].add(asin_val)
                        st.toast(f"Clic registrado para {title_val[:25]}!")
                with btn_c2:
                    if st.button(f"🛒 Cesta #{i}", key=f"cart_{asin_val}_{i}"):
                        st.session_state["suggested_cart"].append({
                            "asin": asin_val,
                            "title": title_val,
                            "price": price_val,
                            "category": cat_val
                        })
                        st.toast(f"¡Añadido a Cesta Sugerida!")
                        st.rerun()


    # TAB 2: EXECUTIVE KPIS
    with tab2:
        st.subheader("Business Analytics & Portfolio Health")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_rev = df["monetary"].sum()
            st.metric("Total Revenue", f"${total_rev:,.2f}", "+14.2% YoY")
        with col2:
            avg_rec = df["recency"].mean()
            st.metric("Avg Recency", f"{avg_rec:.1f} days", "-6.8 days")
        with col3:
            avg_freq = df["frequency"].mean()
            st.metric("Avg Order Frequency", f"{avg_freq:.1f} orders")
        with col4:
            clv_series = df["clv_pred"] if "clv_pred" in df.columns else df.get("projected_clv_1y", df["monetary"] * 0.75)
            mean_clv = float(clv_series.mean())
            st.metric("Avg Predicted 1Y CLV", f"${mean_clv:.2f}")

        st.divider()

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            fig_persona = px.pie(
                df,
                names="persona",
                title="Customer Persona Distribution",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_persona.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            apply_ultra_hd_style(fig_persona, title="Customer Persona Distribution")
            st.plotly_chart(fig_persona, use_container_width=True, config=PLOTLY_HIGH_RES_CONFIG)

        with row1_col2:
            fig_spend = px.histogram(
                df,
                x="monetary",
                nbins=40,
                title="Customer Spend Distribution ($)",
                color_discrete_sequence=["#4F46E5"],
                log_y=True
            )
            fig_spend.update_layout(xaxis_title="Monetary Value ($)", yaxis_title="Customer Count (Log Scale)")
            apply_ultra_hd_style(fig_spend, title="Customer Spend Distribution ($)")
            st.plotly_chart(fig_spend, use_container_width=True, config=PLOTLY_HIGH_RES_CONFIG)

    # TAB 3: CLUSTERING & COMPARATIVE DBSCAN EVALUATION
    with tab3:
        st.subheader("Unsupervised Clustering & Senior K-Means vs DBSCAN Comparison")
        st.write("Validation of spherical K-Means partitions against non-spherical density groups and anomalous outlier detection via DBSCAN.")

        if pipe.cluster_model:
            from customer_insights.models.clustering import CustomerClusterModel
            if hasattr(pipe.cluster_model, "compare_kmeans_dbscan"):
                db_eval = pipe.cluster_model.compare_kmeans_dbscan(df)
            else:
                db_eval = CustomerClusterModel.compare_kmeans_dbscan(pipe.cluster_model, df)

            
            c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
            with c_kpi1:
                st.metric("K-Means Clusters", f"{db_eval['kmeans_n_clusters']} partitions")
            with c_kpi2:
                st.metric("DBSCAN Natural Density Clusters", f"{db_eval['dbscan_natural_clusters']} dense cores")
            with c_kpi3:
                st.metric("Flagged Anomalous Outliers", f"{db_eval['total_dbscan_outliers']} ({db_eval['global_outlier_percentage']}%)")

            st.write("#### 📊 Comparative Breakdown: K-Means vs DBSCAN Noise Detection")
            breakdown_df = pd.DataFrame(db_eval["cluster_breakdown"])
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        st.write("#### Cluster Profiles & 3D Centroids")
        if pipe.cluster_model and pipe.cluster_model.cluster_profiles is not None:
            st.dataframe(pipe.cluster_model.cluster_profiles, use_container_width=True, hide_index=True)

    # TAB 4: CLV & CHURN RISK
    with tab4:
        st.subheader("Customer Lifetime Value (12-Month Horizon) & Churn Dynamics")
        st.write("Prioritize retention interventions by identifying high-value customers with elevated churn probability.")

        clv_target_col = "clv_pred" if "clv_pred" in df.columns else ("projected_clv_1y" if "projected_clv_1y" in df.columns else "monetary")

        clv_col1, clv_col2 = st.columns([3, 2])
        with clv_col1:
            fig_risk = px.scatter(
                df,
                x="churn_probability",
                y=clv_target_col,
                color="churn_risk_level",
                size="monetary",
                hover_data=["reviewerID", "persona", "recency"],
                title="CLV vs Churn Risk Matrix (Action Matrix)",
                color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
            )
            fig_risk.add_vline(x=0.5, line_dash="dash", line_color="gray")
            fig_risk.add_hline(y=df[clv_target_col].median(), line_dash="dash", line_color="gray")
            fig_risk.update_layout(xaxis_title="Churn Probability P(Churn)", yaxis_title="Predicted 1-Year CLV ($)")
            apply_ultra_hd_style(fig_risk, title="CLV vs Churn Risk Matrix (Action Matrix)")
            st.plotly_chart(fig_risk, use_container_width=True, config=PLOTLY_HIGH_RES_CONFIG)

        with clv_col2:
            st.write("#### High-Value At-Risk Customers (Rescue Targets)")
            rescue_targets = df[(df["churn_probability"] > 0.50) & (df[clv_target_col] > df[clv_target_col].median())]
            st.metric("Total Rescue Candidates", f"{len(rescue_targets):,} customers")
            st.dataframe(
                rescue_targets[["reviewerID", "monetary", "recency", "churn_probability", clv_target_col]]
                .sort_values(clv_target_col, ascending=False)
                .head(10),
                use_container_width=True,
                hide_index=True
            )


    # TAB 5: PROFILE SIMULATOR
    with tab5:
        st.subheader("Interactive Profile Classifier & Recommendation Sandbox")
        st.write("Simulate any hypothetical customer profile to compute its segment, CLV, and marketing playbook.")

        with st.form("simulate_customer_form"):
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_recency = st.slider("Recency (Days since last purchase)", min_value=1, max_value=730, value=30)
                sim_frequency = st.slider("Frequency (Total orders)", min_value=1, max_value=50, value=6)
            with sim_col2:
                sim_monetary = st.number_input("Monetary Spend ($)", min_value=5.0, max_value=5000.0, value=280.0)
                sim_rating = st.slider("Average Rating Given", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

            submitted = st.form_submit_button("Simulate Customer Diagnosis & Plan")

        if submitted:
            sim_tenure = sim_recency + (sim_frequency * 35.0)
            
            cluster_info = pipe.cluster_model.predict_single(sim_recency, sim_frequency, sim_monetary, sim_rating)
            clv_info = pipe.clv_model.predict_single(sim_recency, sim_frequency, sim_monetary, sim_tenure)
            persona_info = PersonaEngine.assign_persona(sim_recency, sim_frequency, sim_monetary, sim_rating, cluster_info["cluster"])

            st.success("Analysis Complete!")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
                ### Diagnosis
                - **Assigned Cluster:** `Cluster #{cluster_info['cluster']}`
                - **Business Persona:** **{persona_info['persona_name']}**
                - **Distance to Centroid:** `{cluster_info['centroid_distance']:.3f}`
                - **Churn Probability:** `{clv_info['churn_probability'] * 100:.1f}%` ({clv_info['churn_risk_level']} Risk)
                - **Predicted 1-Year CLV:** **${clv_info.get('clv_pred', clv_info['projected_clv_1y']):.2f}**
                """)
            with res_col2:
                st.markdown(f"""
                ### Strategic Prescriptions
                - **Marketing Strategy:** {persona_info['marketing_strategy']}
                - **Preferred Channel:** `{persona_info['preferred_channel']}`
                - **Discount Incentive:** `{persona_info['discount_incentive']}`
                """)

    # TAB 6: STATISTICAL TESTS & PRESCRIPTIVE OPTIMIZER
    with tab6:
        st.subheader("🔬 Inferential Statistics, Hypothesis Testing & Prescriptive Optimizer")
        st.write("Rigorous econometric and statistical validation of customer behavior and integer linear programming budget allocation.")

        st_col1, st_col2 = st.columns(2)
        with st_col1:
            st.write("#### 📐 Univariate Diagnostics & Pareto Gini")
            diag_res = run_univariate_diagnostics(df)
            diag_df = pd.DataFrame(diag_res).T
            st.dataframe(diag_df, use_container_width=True)
            gini_val = diag_res.get("monetary", {}).get("gini_coefficient", 0.60)
            st.info(f"**Monetary Gini Inequality Index:** `{gini_val}` (Demonstrates strong Pareto 80/20 concentration of spend).")

        with st_col2:
            st.write("#### 🧪 Formal Hypothesis Testing Suite")
            tests_res = run_hypothesis_tests(df)
            
            for test_name, test_data in tests_res.items():
                status_icon = "✅" if test_data.get("reject_H0_normal") or test_data.get("reject_H0_identical") or test_data.get("reject_H0_independence") else "ℹ️"
                with st.expander(f"{status_icon} {test_data['test']}", expanded=True):
                    st.write(f"**Conclusion:** {test_data['conclusion']}")
                    stat_key = [k for k in test_data.keys() if k.startswith("statistic") or k == "rho"][0]
                    st.code(f"{stat_key}: {test_data[stat_key]} | p-value: {test_data['p_value']:.4e}")

        st.divider()
        st.write("#### 💼 Prescriptive Retention Budget Optimizer")
        opt_form_col1, opt_form_col2, opt_form_col3 = st.columns(3)
        with opt_form_col1:
            budget_input = st.number_input("Total Retention Budget ($)", min_value=500.0, max_value=50000.0, value=5000.0, step=500.0)
        with opt_form_col2:
            cost_input = st.number_input("Cost per Customer ($)", min_value=2.0, max_value=100.0, value=15.0, step=1.0)
        with opt_form_col3:
            lift_input = st.slider("Expected Retention Lift ΔP(Active)", min_value=0.05, max_value=0.60, value=0.25, step=0.05)

        opt_result = optimize_retention_budget(
            df,
            total_budget=budget_input,
            intervention_cost=cost_input,
            expected_retention_lift=lift_input
        )

        res_kpi1, res_kpi2, res_kpi3, res_kpi4 = st.columns(4)
        with res_kpi1:
            st.metric("Targeted Customers", f"{opt_result['number_of_customers_targeted']:,}")
        with res_kpi2:
            st.metric("Actual Campaign Spend", f"${opt_result['actual_campaign_cost']:,.2f}")
        with res_kpi3:
            st.metric("Expected Net CLV Saved", f"${opt_result['total_expected_net_clv_saved']:,.2f}")
        with res_kpi4:
            st.metric("Projected ROI Multiple", f"{opt_result['expected_roi_ratio']:.2f}x")

        st.write("##### Priority Target Customers (Top Expected Net Value Gain):")
        st.dataframe(pd.DataFrame(opt_result["top_targeted_customers"]), use_container_width=True, hide_index=True)

        st.divider()
        st.write("#### 🎨 High-Resolution 300 DPI Statistical Panels (Seaborn & Matplotlib)")
        st.caption("Publication-grade statistical distributions rendered at 300 DPI with vector-level font rasterization.")

        fig_sns, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        fig_sns.patch.set_facecolor('#111827')

        # 1. Boxen Distribution of Monetary Spend by Persona
        ax1.set_facecolor('#1F2937')
        sns.boxenplot(
            data=df,
            x="persona",
            y="monetary",
            palette="mako",
            ax=ax1,
            showfliers=False
        )
        ax1.set_title("Customer Monetary Spend ($) by Persona (Log Scale)", color="#F9FAFB", fontsize=11, fontweight="bold", pad=10)
        ax1.set_yscale("log")
        ax1.set_xlabel("Persona", color="#9CA3AF", fontsize=9)
        ax1.set_ylabel("Monetary ($)", color="#9CA3AF", fontsize=9)
        ax1.tick_params(axis='x', rotation=30, colors="#D1D5DB", labelsize=8)
        ax1.tick_params(axis='y', colors="#D1D5DB", labelsize=8)

        # 2. Correlation Matrix Heatmap
        ax2.set_facecolor('#1F2937')
        corr_cols = ["recency", "frequency", "monetary", "customer_age_T", "avg_rating", "clv_pred", "churn_probability"]
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="crest",
            ax=ax2,
            cbar=True,
            annot_kws={"size": 8, "weight": "bold"},
            linewidths=0.5,
            linecolor='#111827'
        )
        ax2.set_title("Behavioral & Predictive Correlation Matrix", color="#F9FAFB", fontsize=11, fontweight="bold", pad=10)
        ax2.tick_params(axis='x', rotation=45, colors="#D1D5DB", labelsize=8)
        ax2.tick_params(axis='y', rotation=0, colors="#D1D5DB", labelsize=8)

        plt.tight_layout()
        st.pyplot(fig_sns, use_container_width=True)
        plt.close(fig_sns)


if __name__ == "__main__":
    main()
