"""
Production-Grade Data Analysis Dashboard.

A professional, tab-based dashboard for machine learning analysis.
Provides data exploration, clustering, modeling, and actionable insights.

KEY FEATURES:
- Automatic: Data preprocessing, EDA, clustering, insights
- Optional: Predictive modeling (user-controlled)
- Smart UX: Detects target column, enables/disables modeling
- No forced decisions: Users choose to add modeling if needed
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import production components
from src.pipeline_orchestrator import ProductionMLPipeline
from src.visualization.dashboard_viz import DashboardVisualizer
from src.insights.business_insights import BusinessInsightGenerator


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AutoML Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# HELPER FUNCTIONS FOR ANALYSIS
# ============================================================================

def interpret_silhouette_score(score: float) -> Tuple[str, str]:
    """
    Interpret silhouette score and return message and level.
    
    Args:
        score: Silhouette score (-1 to 1)
    
    Returns:
        Tuple of (message, level) where level is 'success', 'warning', or 'error'
    """
    if score < -0.2:
        return "❌ Very weak clustering structure. Segments are not meaningful.", "error"
    elif score < 0.0:
        return "⚠️ Weak clustering structure. Segments may not be well-separated.", "warning"
    elif score < 0.3:
        return "⚠️ Weak clustering. Segmentation is barely meaningful (score < 0.3).", "warning"
    elif score < 0.5:
        return "💡 Moderate clustering quality. Segments have some coherence.", "info"
    elif score < 0.7:
        return "✅ Good clustering quality. Segments are reasonably well-separated.", "success"
    else:
        return "🌟 Excellent clustering quality. Segments are very well-separated.", "success"


def interpret_model_performance(score: float, problem_type: str = "unknown") -> Tuple[str, str]:
    """
    Interpret model performance score.
    
    Args:
        score: Performance score (0 to 1)
        problem_type: 'classification' or 'regression'
    
    Returns:
        Tuple of (message, level)
    """
    if score < 0.6:
        return f"⚠️ Model performance is weak ({score:.1%}). Predictions are unreliable and should not drive decisions yet.", "warning"
    elif score <= 0.75:
        return f"💡 Moderate performance ({score:.1%}). Model shows some predictive power but could be improved.", "info"
    else:
        return f"✅ Strong performance ({score:.1%}). Model predictions are reasonably reliable.", "success"


def render_status_message(message: str, level: str) -> None:
    """Render semantic status messages using Streamlit-native components."""
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "info":
        st.info(message)
    else:
        st.success(message)


def render_bullet_messages(messages) -> None:
    """Render a list of messages with consistent spacing and readability."""
    if not messages:
        st.info("No items available.")
        return

    for message in messages:
        st.write(f"- {message}")


# ============================================================================
# SIDEBAR: DATA UPLOAD & CONTROLS
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'pca_data' not in st.session_state:
        st.session_state.pca_data = None
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    if 'enable_modeling' not in st.session_state:
        st.session_state.enable_modeling = False
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "auto"


# ============================================================================
# SIDEBAR: DATA UPLOAD & CONTROLS
# ============================================================================

def render_sidebar():
    """Render simplified sidebar with clear UX flow."""
    with st.sidebar:
        st.title("⚙️ AutoML Setup")
        st.caption("Upload a dataset, choose whether to build a model, and run the analysis.")
        st.divider()

        st.subheader("1. Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset"
        )
        
        if uploaded_file is None:
            st.info("👆 Upload a CSV file to continue")
            return
        
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.raw_data = data
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
        
        st.divider()

        st.subheader("2. Modeling")
        st.session_state.enable_modeling = st.checkbox(
            "Build Predictive Model",
            value=False,
            help="If unchecked, only clustering analysis will run"
        )

        st.divider()

        if st.session_state.enable_modeling:
            st.subheader("3. Select Target")
            numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
            
            if not numeric_cols:
                st.error("❌ No numeric columns found. Cannot build model.")
                st.session_state.enable_modeling = False
            else:
                st.session_state.selected_target = st.selectbox(
                    "Target to predict:",
                    options=numeric_cols,
                    help="The column you want to predict"
                )
        else:
            st.subheader("3. Analysis Mode")
            st.info("Clustering and EDA will run automatically.")
            st.session_state.selected_target = None

        st.divider()
        st.subheader("4. Run Analysis")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Analysis Only", use_container_width=True):
                run_analysis_only(data)
        
        with col2:
            if st.session_state.enable_modeling:
                if st.button("🤖 Run Model", use_container_width=True, type="primary"):
                    run_modeling_only(data, st.session_state.selected_target)
            else:
                st.button("🤖 Run Model", use_container_width=True, disabled=True)


def run_analysis_only(data: pd.DataFrame):
    """Run clustering + EDA analysis WITHOUT modeling."""
    try:
        with st.spinner("📊 Running analysis..."):
            pipeline = ProductionMLPipeline(verbose=False)
            results = pipeline.run_full_pipeline(
                df=data,
                target_col=None,
                run_modeling=False,
                auto_clean=True
            )
            st.session_state.pipeline_results = results
            st.session_state.processed_data = results.get("clustering", {}).get("preprocessed_data")
            st.session_state.pca_data = results.get("clustering", {}).get("pca_data")
            st.session_state.modeling_skipped = True
            st.success("✅ Analysis complete!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Analysis error: {str(e)}", exc_info=True)


def run_modeling_only(data: pd.DataFrame, target_col: str):
    """Run ONLY modeling (skip clustering/EDA)."""
    if not target_col:
        st.error("❌ Target column required for modeling!")
        return
    
    try:
        with st.spinner("🤖 Training models..."):
            pipeline = ProductionMLPipeline(verbose=False)
            results = pipeline.run_full_pipeline(
                df=data,
                target_col=target_col,
                run_modeling=True,
                auto_clean=True
            )
            st.session_state.pipeline_results = results
            st.session_state.processed_data = results.get("clustering", {}).get("preprocessed_data")
            st.session_state.pca_data = results.get("clustering", {}).get("pca_data")
            st.session_state.modeling_skipped = False
            st.success("✅ Modeling complete!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Modeling error: {str(e)}", exc_info=True)


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

def render_overview_tab():
    """Render overview tab with dataset summary."""
    st.subheader("Dataset Overview")
    
    if not st.session_state.data_loaded:
        st.info("👈 Upload a dataset from the sidebar to get started")
        return
    
    data = st.session_state.raw_data
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Total Features", len(data.columns))
        with col3:
            numeric_count = len(data.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric", numeric_count)
        with col4:
            cat_count = len(data.select_dtypes(include=['object']).columns)
            st.metric("Categorical", cat_count)

    st.divider()

    missing_total = data.isnull().sum().sum()
    missing_pct = (missing_total / (len(data) * len(data.columns))) * 100 if len(data) > 0 else 0
    duplicates = len(data[data.duplicated()])
    high_card = len([col for col in data.columns if len(data[col].unique()) > len(data) * 0.95])

    with st.container():
        st.markdown("### Data Quality")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Missing Data",
                f"{missing_pct:.1f}%",
                delta="Healthy" if missing_pct < 10 else "Needs review",
                delta_color="normal" if missing_pct < 10 else "inverse"
            )
        with col2:
            st.metric("Duplicates", duplicates)
        with col3:
            st.metric("High Cardinality", high_card)

    st.divider()

    with st.container():
        st.markdown("### Data Sample")
        st.dataframe(data.head(10), use_container_width=True)

    st.divider()

    with st.container():
        st.markdown("### Statistical Summary")
        numeric_summary = data.select_dtypes(include=[np.number])
        if numeric_summary.empty:
            st.info("No numeric columns available for summary statistics.")
        else:
            st.dataframe(numeric_summary.describe(), use_container_width=True)


# ============================================================================
# TAB 2: DATA ANALYSIS (EDA)
# ============================================================================

def render_data_analysis_tab():
    """Render exploratory data analysis tab."""
    st.subheader("Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.info("👈 Upload a dataset to perform analysis")
        return
    
    data = st.session_state.raw_data
    
    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            analysis_type = st.radio(
                "Analysis Type",
                ["Single Feature Analysis", "Compare Features"],
                horizontal=True
            )

        with col2:
            columns = st.multiselect(
                "Select Columns",
                options=data.columns.tolist(),
                default=data.columns.tolist()[:min(3, len(data.columns))],
                max_selections=5
            )
    
    if not columns:
        st.warning("Please select at least one column")
        return
    
    st.divider()
    
    if analysis_type == "Single Feature Analysis":
        render_univariate_analysis(data, columns)
    else:
        render_bivariate_analysis(data, columns)


def render_univariate_analysis(df: pd.DataFrame, columns: list):
    """Render univariate distributions."""
    numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[columns].select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        st.markdown("### 📊 Numeric Distributions")
        cols = st.columns(min(3, len(numeric_cols)))
        for idx, col in enumerate(numeric_cols):
            with cols[idx % len(cols)]:
                fig = DashboardVisualizer.plot_numeric_distribution(df, col)
                st.plotly_chart(fig, use_container_width=True)
    
    if categorical_cols:
        st.markdown("### 📈 Categorical Distributions")
        for col in categorical_cols:
            fig = DashboardVisualizer.plot_categorical_distribution(df, col)
            st.plotly_chart(fig, use_container_width=True)


def render_bivariate_analysis(df: pd.DataFrame, columns: list):
    """Render correlation analysis."""
    numeric_data = df[columns].select_dtypes(include=[np.number])
    
    if len(numeric_data.columns) >= 2:
        st.markdown("### 📈 Correlation Analysis")
        fig = DashboardVisualizer.plot_correlation_heatmap(numeric_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least 2 numeric columns for correlation analysis")


# ============================================================================
# TAB 3: CLUSTERING
# ============================================================================

def render_clustering_tab():
    """Render clustering results with honest interpretation."""
    st.subheader("Clustering Analysis")
    
    if st.session_state.pipeline_results is None:
        st.info("👈 Run analysis from the sidebar to see clustering results")
        return
    
    results = st.session_state.pipeline_results
    clustering_info = results.get('clustering')
    
    # Defensive check: clustering_info can be None or empty dict
    if not clustering_info or not isinstance(clustering_info, dict):
        st.error("❌ Clustering analysis not available")
        return
    
    # Get evaluation metrics safely
    evaluation = clustering_info.get('evaluation')
    if not evaluation or not isinstance(evaluation, dict):
        st.error("❌ Cluster evaluation metrics not available")
        return
    
    silhouette = evaluation.get('silhouette_score', 0)
    
    st.divider()
    st.markdown("### Clustering Quality Assessment")
    interpretation, level = interpret_silhouette_score(silhouette)
    render_status_message(interpretation, level)
    
    st.divider()
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_clusters = evaluation.get('n_clusters', 0)
            st.metric("Clusters", n_clusters if n_clusters else "-")
        with col2:
            st.metric("Silhouette", f"{silhouette:.3f}" if silhouette is not None else "-",
                     help="Range: -1 to 1. Higher is better.")
        with col3:
            davies_bouldin = evaluation.get('davies_bouldin_score')
            if davies_bouldin is not None:
                st.metric("Davies-Bouldin", f"{davies_bouldin:.3f}", help="Lower is better")
            else:
                st.metric("Davies-Bouldin", "-")
        with col4:
            calinski = evaluation.get('calinski_harabasz_score')
            if calinski is not None:
                st.metric("Calinski-Harabasz", f"{calinski:.1f}", help="Higher is better")
            else:
                st.metric("Calinski-Harabasz", "-")
    
    st.divider()
    
    st.markdown("### Cluster Insights")
    try:
        insights = BusinessInsightGenerator.generate_clustering_insights(
            n_clusters=evaluation.get('n_clusters', 0),
            silhouette_score=silhouette,
            cluster_sizes={}
        )
        render_bullet_messages(insights)
    except Exception as e:
        st.warning(f"Could not generate insights: {str(e)}")
        logger.warning(f"Clustering insights error: {str(e)}")
    
    st.divider()
    
    st.markdown("### Visualizations")
    
    try:
        # Try to get pre-computed PCA data
        pca_data = clustering_info.get('pca_data')
        labels = clustering_info.get('labels')
        
        # Defensive: check both are not None and have data
        if pca_data is not None and labels is not None and len(labels) > 0:
            fig = DashboardVisualizer.plot_cluster_visualization(
                results=results,
                data_pca=pca_data,
                labels=labels,
                algorithm=clustering_info.get('algorithm', 'Auto-Selected')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: try to compute PCA from results if available
            df_with_clusters = clustering_info.get('df_with_clusters')
            if (
                df_with_clusters is not None
                and isinstance(df_with_clusters, pd.DataFrame)
                and len(df_with_clusters) > 0
                and 'cluster' in df_with_clusters.columns
            ):
                fig = DashboardVisualizer.plot_cluster_visualization(
                    df=df_with_clusters,
                    cluster_col='cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ PCA visualization data not available for this dataset")
    except Exception as e:
        st.warning(f"⚠️ Could not render visualization: {str(e)}")
        logger.warning(f"Visualization error: {str(e)}")
    
    # Cluster sizes
    try:
        labels = clustering_info.get('labels')
        if labels is not None and len(labels) > 0:
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            
            fig = DashboardVisualizer.plot_cluster_sizes(cluster_dict=cluster_sizes)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Cluster data not available")
    except Exception as e:
        st.warning(f"⚠️ Could not render cluster sizes: {str(e)}")
        logger.warning(f"Cluster sizes error: {str(e)}")


# ============================================================================
# TAB 4: MODELING (CONDITIONAL)
# ============================================================================

def render_modeling_tab():
    """Render model results with honest performance interpretation."""
    st.subheader("Predictive Modeling")
    
    if st.session_state.pipeline_results is None:
        st.info("Select a target column and click 'Run Model'.")
        return

    results = st.session_state.pipeline_results
    modeling_info = results.get('modeling')

    if modeling_info is None or not isinstance(modeling_info, dict) or not modeling_info:
        if st.session_state.get("selected_target"):
            st.info("Click 'Run Model' to train a predictive model.")
        else:
            st.info("Select a target column and click 'Run Model'.")
        return
    
    best_score = modeling_info.get('best_score', 0)
    problem_type = modeling_info.get('problem_type', 'unknown')

    st.success("Modeling complete!")
    
    st.divider()
    st.markdown("### Model Performance Assessment")
    interpretation, level = interpret_model_performance(best_score, problem_type)
    render_status_message(interpretation, level)
    
    st.divider()
    
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            best_model = modeling_info.get('best_model_name')
            st.metric("Best Model", best_model if best_model else "N/A")
        with col2:
            st.metric("Score", f"{best_score:.1%}"  if best_score is not None else "N/A",
                     help="Cross-validation average score")
        with col3:
            st.metric("Task Type", problem_type.title() if problem_type else "Unknown")
    
    st.divider()
    
    st.markdown("### Insights")
    try:
        insights = BusinessInsightGenerator.generate_model_insights(
            best_score=best_score,
            problem_type=problem_type,
            cross_val_scores=modeling_info.get('cross_val_scores', {}),
            best_model=modeling_info.get('best_model_name', '')
        )
        render_bullet_messages(insights)
    except Exception as e:
        st.warning(f"Could not generate insights: {str(e)}")
        logger.warning(f"Model insights error: {str(e)}")
    
    st.divider()
    
    st.markdown("### Model Comparison")
    try:
        cross_val = modeling_info.get('cross_val_scores')
        
        # Defensive: check for None, empty dict, and valid data
        if cross_val and isinstance(cross_val, dict) and len(cross_val) > 0:
            # Ensure all values are valid (not None, numeric)
            valid_scores = {k: v for k, v in cross_val.items() if v is not None}
            if valid_scores:
                fig = DashboardVisualizer.plot_model_comparison(valid_scores)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model comparison data not available")
        else:
            st.info("Model comparison data not available")
    except Exception as e:
        st.warning(f"Could not render model comparison: {str(e)}")
        logger.warning(f"Model comparison error: {str(e)}")
    
    st.divider()
    st.markdown("### Feature Importance")
    
    try:
        feature_importance = modeling_info.get('feature_importance')
        
        # Defensive: check for None and empty dict
        if feature_importance and isinstance(feature_importance, dict) and len(feature_importance) > 0:
            fig = DashboardVisualizer.plot_feature_importance(feature_importance, top_n=5)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Top Features")
            try:
                finsights = BusinessInsightGenerator.generate_feature_insights(
                    feature_importance=feature_importance,
                    original_features=list(feature_importance.keys())
                )
                render_bullet_messages(finsights)
            except Exception as e:
                logger.warning(f"Feature insights error: {str(e)}")
        else:
            st.info("Feature importance not available for this model")
    except Exception as e:
        st.warning(f"Could not render feature importance: {str(e)}")
        logger.warning(f"Feature importance error: {str(e)}")


# ============================================================================
# TAB 5: INSIGHTS & RECOMMENDATIONS
# ============================================================================

def render_insights_tab():
    """Render business insights with honest assessment of findings."""
    st.subheader("Actionable Insights & Recommendations")
    
    if st.session_state.pipeline_results is None:
        st.info("👈 Run analysis from the sidebar to see insights")
        return
    
    results = st.session_state.pipeline_results
    
    st.markdown("### Data Quality")
    try:
        if st.session_state.raw_data is not None:
            data_insights = BusinessInsightGenerator.generate_data_quality_insights(
                st.session_state.raw_data
            )
            render_bullet_messages(data_insights)
        else:
            st.warning("⚠️ No data available for quality insights")
    except Exception as e:
        st.warning(f"⚠️ Could not generate data quality insights: {str(e)}")
        logger.warning(f"Data quality insights error: {str(e)}")
    
    st.divider()
    
    # Clustering insights
    try:
        clustering_info = results.get('clustering')
        if clustering_info and isinstance(clustering_info, dict):
            evaluation = clustering_info.get('evaluation')
            if evaluation and isinstance(evaluation, dict):
                st.markdown("### 🎯 Segmentation Assessment")
                silhouette = evaluation.get('silhouette_score', 0)
                interpretation, level = interpret_silhouette_score(silhouette)
                render_status_message(interpretation, level)
                
                try:
                    cluster_insights = BusinessInsightGenerator.generate_clustering_insights(
                        n_clusters=evaluation.get('n_clusters', 0),
                        silhouette_score=silhouette,
                        cluster_sizes={}
                    )
                    render_bullet_messages(cluster_insights)
                except Exception as e:
                    logger.warning(f"Clustering insights error: {str(e)}")
                
                st.divider()
    except Exception as e:
        logger.warning(f"Clustering analysis error: {str(e)}")
    
    # Modeling insights (if available)
    try:
        modeling_info = results.get('modeling')
        if modeling_info and isinstance(modeling_info, dict) and modeling_info:
            st.markdown("### Model Performance Assessment")
            best_score = modeling_info.get('best_score', 0)
            problem_type = modeling_info.get('problem_type', 'unknown')
            
            interpretation, level = interpret_model_performance(best_score, problem_type)
            render_status_message(interpretation, level)
            
            try:
                model_insights = BusinessInsightGenerator.generate_model_insights(
                    best_score=best_score,
                    problem_type=problem_type,
                    cross_val_scores=modeling_info.get('cross_val_scores', {}),
                    best_model=modeling_info.get('best_model_name', '')
                )
                render_bullet_messages(model_insights)
            except Exception as e:
                logger.warning(f"Model insights error: {str(e)}")
            
            st.divider()
    except Exception as e:
        logger.warning(f"Model analysis error: {str(e)}")
    
    st.markdown("### Next Steps")
    try:
        modeling_info = results.get('modeling')
        if modeling_info is None:
            modeling_info = {}
        
        n_records = len(st.session_state.raw_data) if st.session_state.raw_data is not None else 0
        
        recommendations = BusinessInsightGenerator.generate_recommendations(
            problem_type=modeling_info.get('problem_type', 'unknown'),
            best_score=modeling_info.get('best_score', 0),
            n_records=n_records,
            missing_pct=0,
            feature_importance=modeling_info.get('feature_importance', {})
        )
        render_bullet_messages(recommendations)
    except Exception as e:
        st.warning(f"⚠️ Could not generate recommendations: {str(e)}")
        logger.warning(f"Recommendations error: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    initialize_session_state()

    st.title("📊 AutoML Data Analyzer")
    st.caption(
        "Professional machine learning analysis for dataset exploration, clustering, "
        "optional predictive modeling, and actionable recommendations."
    )
    st.divider()
    
    # Sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview",
        "🔍 Analysis",
        "🎯 Clustering",
        "🤖 Modeling",
        "💡 Insights"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_data_analysis_tab()
    
    with tab3:
        render_clustering_tab()
    
    with tab4:
        render_modeling_tab()
    
    with tab5:
        render_insights_tab()


if __name__ == "__main__":
    main()
