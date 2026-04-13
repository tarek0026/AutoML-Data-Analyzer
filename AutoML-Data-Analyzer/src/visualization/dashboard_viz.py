"""
Visualization Module for AI Data Analyst Dashboard.

Provides production-quality visualizations:
- Data overview charts
- Clustering visualizations
- Correlation heatmaps
- Missing value patterns
- Feature importance plots
"""

from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import logging

logger = logging.getLogger(__name__)


class DashboardVisualizer:
    """Production-quality visualizations for dashboard."""
    
    @staticmethod
    def plot_numeric_distribution(df: pd.DataFrame, column: str) -> go.Figure:
        """Create histogram with stats for numeric column."""
        data = df[column].dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=30,
            name='Distribution',
            marker_color='rgba(0, 150, 200, 0.7)',
            hovertemplate='<b>Value Range</b>: %{x}<br><b>Frequency</b>: %{y}<extra></extra>'
        ))
        
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_val:.2f}", 
                     annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                     annotation_text=f"Median: {median_val:.2f}",
                     annotation_position="top left")
        
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            hovermode='x unified',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_categorical_distribution(df: pd.DataFrame, column: str) -> go.Figure:
        """Create bar chart for categorical column."""
        counts = df[column].value_counts()
        
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            title=f"Distribution of {column}",
            labels={'x': column, 'y': 'Count'},
            color=counts.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_missing_values(df: pd.DataFrame) -> go.Figure:
        """Visualize missing value patterns."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
        
        if missing_pct.sum() == 0:
            # No missing values - create empty figure with message
            fig = go.Figure()
            fig.add_annotation(text="✓ No missing values detected", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(size=20, color="green"))
            fig.update_layout(height=300, template='plotly_white')
            return fig
        
        missing_data = missing_pct[missing_pct > 0]
        
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column (%)",
            labels={'x': 'Column', 'y': 'Missing %'},
            color=missing_data.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            xaxis_tickangle=-45,
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            fig = go.Figure()
            fig.add_annotation(text="Fewer than 2 numeric columns for correlation",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=300, template='plotly_white')
            return fig
        
        corr = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 9},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return fig
    
    @staticmethod
    def plot_cluster_visualization(results: Dict[str, Any] = None, 
                                  data_pca: np.ndarray = None,
                                  labels: np.ndarray = None,
                                  algorithm: str = 'KMeans',
                                  df: pd.DataFrame = None,
                                  cluster_col: str = 'cluster',
                                  pca_components: int = 2) -> go.Figure:
        """
        Visualize clusters using PCA (2D only, numeric features only).
        Safe PCA computation with defensive checks.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Case 1: PCA data already computed (new API)
            if data_pca is not None and labels is not None:
                data_pca = np.asarray(data_pca)
                labels = np.asarray(labels)

                if data_pca.ndim != 2 or len(data_pca) < 3 or data_pca.shape[1] < 2:
                    fig = go.Figure()
                    fig.add_annotation(text="Insufficient data for PCA visualization",
                                     xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(height=500, template='plotly_white')
                    return fig

                if len(data_pca) != len(labels):
                    min_len = min(len(data_pca), len(labels))
                    data_pca = data_pca[:min_len]
                    labels = labels[:min_len]
                
                # Use only 2 components
                pca_2d = data_pca[:, :2] if data_pca.shape[1] >= 2 else data_pca
                
                fig = px.scatter(
                    x=pca_2d[:, 0],
                    y=pca_2d[:, 1],
                    color=labels.astype(str),
                    title=f"Cluster Visualization ({algorithm})",
                    labels={'x': 'PC1', 'y': 'PC2'},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                fig.update_layout(
                    height=500, 
                    template='plotly_white', 
                    hovermode='closest'
                )
                return fig
            
            # Case 2: Compute PCA from dataframe (old API fallback)
            if df is not None:
                # Select only numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Remove cluster column if present
                if cluster_col in numeric_cols:
                    numeric_cols.remove(cluster_col)
                
                if len(numeric_cols) < 2:
                    fig = go.Figure()
                    fig.add_annotation(text="Insufficient numeric features for visualization",
                                     xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(height=500, template='plotly_white')
                    return fig
                
                # Safe handling of NaN values
                X = df[numeric_cols].copy()
                X = X.dropna(how='all')
                
                if len(X) < 3 or X.shape[1] < 2:
                    fig = go.Figure()
                    fig.add_annotation(text="Not enough data for PCA",
                                     xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(height=500, template='plotly_white')
                    return fig
                
                # Fill remaining NaN with median
                X = X.fillna(X.median())
                X_scaled = StandardScaler().fit_transform(X)
                
                # Use only 2 components for 2D visualization
                pca = PCA(n_components=min(2, X_scaled.shape[1]))
                components = pca.fit_transform(X_scaled)
                
                fig = px.scatter(
                    x=components[:, 0],
                    y=components[:, 1],
                    color=df.loc[X.index, cluster_col].astype(str),
                    title=f"Cluster Visualization (PCA: {pca.explained_variance_ratio_.sum():.1%})",
                    labels={'x': 'PC1', 'y': 'PC2'},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                fig.update_layout(height=500, template='plotly_white', hovermode='closest')
                return fig
            
            # No valid input
            fig = go.Figure()
            fig.add_annotation(text="No data available for visualization",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=500, template='plotly_white')
            return fig
            
        except Exception as e:
            logger.warning(f"PCA visualization failed: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Visualization failed: {str(e)[:50]}",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=500, template='plotly_white')
            return fig
    
    @staticmethod
    def plot_feature_importance(feature_importance: Dict[str, float],
                               top_n: int = 5) -> go.Figure:
        """
        Create feature importance bar chart (TOP 5 features).
        Defensive checks for None/empty data.
        """
        # Defensive check for None or empty data
        if feature_importance is None or not isinstance(feature_importance, dict):
            fig = go.Figure()
            fig.add_annotation(text="No feature importance data available",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
        
        if len(feature_importance) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No features to display",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
        
        # Limit to top_n (5)
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)[:min(top_n, len(feature_importance))]
        
        if not sorted_features:
            fig = go.Figure()
            fig.add_annotation(text="No valid features to display",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
        
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        fig = px.bar(
            x=importances,
            y=features,
            orientation='h',
            title=f"Top {len(features)} Most Important Features",
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importances,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    @staticmethod
    def plot_model_comparison(model_scores: Dict[str, float]) -> go.Figure:
        """
        Create model comparison bar chart.
        Defensive checks to ensure arrays have equal length.
        """
        # Defensive check for None or empty data
        if model_scores is None or not isinstance(model_scores, dict):
            fig = go.Figure()
            fig.add_annotation(text="No model comparison data available",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
        
        if len(model_scores) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No models to compare",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
        
        flattened_scores = {}
        for model_name, score in model_scores.items():
            if model_name is None or score is None:
                continue
            if isinstance(score, (list, tuple, np.ndarray, pd.Series)):
                numeric_values = pd.to_numeric(pd.Series(score), errors="coerce").dropna()
                if numeric_values.empty:
                    continue
                flattened_scores[str(model_name)] = float(numeric_values.mean())
            else:
                try:
                    flattened_scores[str(model_name)] = float(score)
                except (TypeError, ValueError):
                    continue

        if not flattened_scores:
            fig = go.Figure()
            fig.add_annotation(text="No valid model comparison data available",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig

        # Sort and extract models and scores
        sorted_scores = sorted(flattened_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure arrays have equal length (defensive)
        models = []
        scores = []
        for model_name, score in sorted_scores:
            if model_name is not None and score is not None:
                models.append(str(model_name))
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    scores.append(0.0)
        
        if not models or len(models) != len(scores):
            fig = go.Figure()
            fig.add_annotation(text="Invalid model data",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
        
        # Create figure with consistent colors (use go.Bar to avoid Plotly array mismatch)
        fig = go.Figure()
        
        # Add bars with colors - best model (first) is darkgreen, others light blue
        for i in range(len(models)):
            color = 'darkgreen' if i == 0 else 'lightblue'
            fig.add_trace(go.Bar(
                x=[models[i]],
                y=[scores[i]],
                marker_color=color,
                text=f"{scores[i]:.3f}",
                textposition="outside",
                hovertemplate=f"<b>{models[i]}</b><br>Score: {scores[i]:.3f}<extra></extra>",
                showlegend=False
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=400,
            template='plotly_white',
            hovermode='x unified',
            yaxis={'range': [0, max(scores) * 1.1]} if scores else [0, 1]
        )
        
        return fig
    
    @staticmethod
    def plot_cluster_sizes(cluster_dict: Dict[int, int] = None, df: pd.DataFrame = None, 
                           cluster_col: str = 'cluster') -> go.Figure:
        """
        Visualize cluster size distribution.
        Defensive checks for None data and missing cluster column.
        """
        try:
            # Case 1: Cluster dict provided (new API)
            if cluster_dict is not None and isinstance(cluster_dict, dict) and len(cluster_dict) > 0:
                cluster_sizes = pd.Series(cluster_dict)
                
                fig = px.bar(
                    x=cluster_sizes.index,
                    y=cluster_sizes.values,
                    title="Sample Distribution Across Clusters",
                    labels={'x': 'Cluster', 'y': 'Number of Samples'},
                    color=cluster_sizes.values,
                    color_continuous_scale='Blues'
                )
                
                fig.update_layout(
                    height=400,
                    template='plotly_white',
                    showlegend=False,
                    hovermode='x unified'
                )
                return fig
            
            # Case 2: DataFrame provided (old API)
            if df is not None and isinstance(df, pd.DataFrame):
                # Defensive check: column exists
                if cluster_col not in df.columns:
                    fig = go.Figure()
                    fig.add_annotation(text=f"Column '{cluster_col}' not found",
                                     xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(height=400, template='plotly_white')
                    return fig
                
                cluster_sizes = df[cluster_col].value_counts().sort_index()
                
                if len(cluster_sizes) == 0:
                    fig = go.Figure()
                    fig.add_annotation(text="No cluster data available",
                                     xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False)
                    fig.update_layout(height=400, template='plotly_white')
                    return fig
                
                fig = px.bar(
                    x=cluster_sizes.index,
                    y=cluster_sizes.values,
                    title="Sample Distribution Across Clusters",
                    labels={'x': 'Cluster', 'y': 'Number of Samples'},
                    color=cluster_sizes.values,
                    color_continuous_scale='Blues'
                )
                
                fig.update_layout(
                    height=400,
                    template='plotly_white',
                    showlegend=False,
                    hovermode='x unified'
                )
                return fig
            
            # No valid input
            fig = go.Figure()
            fig.add_annotation(text="No cluster data available",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
        
        except Exception as e:
            logger.warning(f"Cluster sizes visualization failed: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Visualization failed: {str(e)[:50]}",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400, template='plotly_white')
            return fig
    
    @staticmethod
    def create_data_summary_table(df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics table."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        summary_data = []
        
        for col in numeric_cols:
            summary_data.append({
                'Column': col,
                'Type': 'Numeric',
                'Non-Null': df[col].notna().sum(),
                'Missing': df[col].isna().sum(),
                'Mean': f"{df[col].mean():.2f}",
                'Std': f"{df[col].std():.2f}",
                'Min': f"{df[col].min():.2f}",
                'Max': f"{df[col].max():.2f}"
            })
        
        for col in categorical_cols:
            summary_data.append({
                'Column': col,
                'Type': 'Categorical',
                'Non-Null': df[col].notna().sum(),
                'Missing': df[col].isna().sum(),
                'Unique': df[col].nunique(),
                'Mode': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                'Top Category': df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A'
            })
        
        return pd.DataFrame(summary_data)
